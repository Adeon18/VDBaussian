import slangpy as spy
import numpy as np
import glfw
from OpenGL.GL import *
from pathlib import Path
import math
import ctypes 
from imgui_bundle import imgui
from slangpy.math import uint3
import random
from pathlib import Path
import openvdb as vdb
import itertools


# --- CONFIG ---
VDB_FILE = "cloud_01_variant_0000.vdb" 
VOL_SIZE = 128
SHADER_FILE = "hybrid.slang"

# ==========================================
# 1. DATA & LOGIC CLASSES
# ==========================================

class Settings:
    def __init__(self):
        # Basic
        self.step_size = 0.015
        self.density_scale = 40.0
        self.density_curve = 1.0
        self.step_count = 256
        self.smoke_color = [0.9, 0.95, 1.0]

        # Lighting & Shadows
        self.light_penetration = 0.15   # The "Fluffiness" factor
        self.phase_g = 0.4 # https://en.wikipedia.org/wiki/Henyey%E2%80%93Greenstein_phase_function - would be 0.7 - 0.9
        
        self.sun_direction = [1.0, -1.0, 0.0]
        self.sun_color_base = [1.0, 0.9, 0.7]
        self.sun_intensity = 8.0
        
        self.ambient_color_base = [0.6, 0.7, 0.8]
        self.ambient_intensity = 0.6
        
        self.shadow_steps = 6
        self.shadow_step_mult = 4.0
    
    # UNNORMALIZED!
    def get_sun_dir(self):
        return self.sun_direction

class Camera:
    def __init__(self, w, h):
        self.pos = np.array([0.0, 0.0, 2.5], dtype=np.float32)
        self.yaw = -90.0
        self.pitch = 0.0
        self.front = np.array([0, 0, -1], dtype=np.float32)
        self.right = np.array([1, 0, 0], dtype=np.float32)
        self.up = np.array([0, 1, 0], dtype=np.float32)
        self.speed = 2.0
        self.sensitivity = 0.1
        self.last_x, self.last_y = w / 2.0, h / 2.0
        self.first_mouse = True
        self.is_dragging = False
        self.update_vectors()

    def update_vectors(self):
        rad_yaw = math.radians(self.yaw)
        rad_pitch = math.radians(self.pitch)
        f = np.array([
            math.cos(rad_yaw) * math.cos(rad_pitch),
            math.sin(rad_pitch),
            math.sin(rad_yaw) * math.cos(rad_pitch)
        ], dtype=np.float32)
        self.front = f / np.linalg.norm(f)
        world_up = np.array([0, 1, 0], dtype=np.float32)
        self.right = np.cross(self.front, world_up)
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.front)
        self.up /= np.linalg.norm(self.up)

    def process_mouse(self, xpos, ypos):
        if self.first_mouse:
            self.last_x, self.last_y = xpos, ypos
            self.first_mouse = False
            return 
        
        xoffset = (xpos - self.last_x) * self.sensitivity
        yoffset = (self.last_y - ypos) * self.sensitivity 
        self.last_x, self.last_y = xpos, ypos
        self.yaw += xoffset
        self.pitch += yoffset 
        if self.pitch > 89.0: self.pitch = 89.0
        if self.pitch < -89.0: self.pitch = -89.0
        self.update_vectors()

    def get_gpu_data(self, aspect_ratio):
        data = np.zeros(16, dtype=np.float32)
        data[0:3] = self.pos
        data[4:7] = self.front
        data[8:11] = self.right * aspect_ratio * 0.5 
        data[12:15] = self.up * 0.5
        return data


def load_vdb_grid(vdb_path):
    """
    Step 1: Just load the OpenVDB grid object from disk.
    """
    print(f"Loading VDB file: {vdb_path}...")
    try:
        if not Path(vdb_path).exists():
            raise FileNotFoundError("File missing")
        
        raw = vdb.readAll(vdb_path)
        # Handle list vs single object return
        grid = raw[0][0] if isinstance(raw, (list, tuple)) else raw
        return grid
    except Exception as e:
        print(f"Error loading VDB: {e}")
        return None

def convert_grid_to_dense_volume(grid, size):
    """
    Step 2: Convert the loaded grid into a dense 3D numpy array for the Raymarcher.
    Returns: (min_world, max_world, dense_volume_data)
    """
    if grid is None:
        print("Error! Grid is None!")
        return None

    # Calculate Bounds
    bbox = grid.evalActiveVoxelBoundingBox()
    min_i, max_i = np.array(bbox[0]), np.array(bbox[1])
    center = (min_i + max_i) / 2.0
    
    # Padding logic: 1.0x extent (User changed from 1.2 previously, keeping 1.0 as per your snippet)
    extent = np.max(max_i - min_i)
    max_dim = extent 
    r = max_dim / 2.0

    min_index_bound = center - r
    max_index_bound = center + r
    
    # Convert to World Space
    transform = grid.transform
    min_w = np.array(transform.indexToWorld(tuple(min_index_bound)))
    max_w = np.array(transform.indexToWorld(tuple(max_index_bound)))
    
    # Dense Sampling
    accessor = grid.getAccessor()
    data = np.zeros((size, size, size), dtype=np.float32)
    inv_size = 1.0 / size
    
    # Optimize: Use numpy vectorized coordinates instead of itertools loop for speed
    # (Optional optimization, but sticking to your logic for safety)
    for z, y, x in itertools.product(range(size), range(size), range(size)):
        uvw = (np.array([x, y, z]) * inv_size) - 0.5
        pos = center + uvw * max_dim
        # Nearest neighbor sampling of the VDB
        val = accessor.getValue(tuple(pos.astype(int)))
        data[z, y, x] = val
        
    m = np.max(data)
    if m > 0: data /= m
    
    return min_w, max_w, np.ascontiguousarray(data, dtype=np.float32)

def convert_grid_to_gaussians(
    grid,
    probability_scale=1.0,
    max_gaussians=None,
    jitter_scale=0.5,
    sigma_scale=1.0
):
    """
    Step 3: Stochastically sample the SAME grid to create Gaussians.
    """
    if grid is None: return np.array([], dtype=np.float32)

    print("Generating Gaussians from grid...")
    accessor = grid.getAccessor()
    bbox = grid.evalActiveVoxelBoundingBox()
    min_i, max_i = np.array(bbox[0]), np.array(bbox[1])
    
    transform = grid.transform
    # Calculate voxel size in world units
    p0 = np.array(transform.indexToWorld((0, 0, 0)))
    p1 = np.array(transform.indexToWorld((1, 0, 0)))
    voxel_size = np.linalg.norm(p1 - p0)

    gaussians = []
    
    # Iterate over active voxels
    # Note: Using grid.cbeginValueOn() is faster than XYZ loops for sparse grids,
    # but loops are fine for now.
    for z in range(min_i[2], max_i[2] + 1):
        for y in range(min_i[1], max_i[1] + 1):
            for x in range(min_i[0], max_i[0] + 1):
                value = accessor.getValue((x, y, z))
                if value <= 0.0: continue

                # Stochastic spawn
                if value * probability_scale < random.random():
                    continue

                center = np.array(transform.indexToWorld((x, y, z)), dtype=np.float32)
                jitter = (np.random.rand(3) - 0.5) * voxel_size * jitter_scale
                position = center + jitter
                
                sigma = voxel_size * sigma_scale
                weight = value

                gaussians.append((
                    position[0], position[1], position[2],
                    sigma, weight
                ))
                
                if max_gaussians and len(gaussians) >= max_gaussians:
                    return np.array(gaussians, dtype=np.float32)

    print(f"Generated {len(gaussians)} gaussians.")
    return np.array(gaussians, dtype=np.float32)
    

# ==========================================
# 2. RENDERER SYSTEM
# ==========================================

class Renderer:
    def __init__(self, device, volume_data):
        self.device = device
        self.pipeline = None
        self.last_mod_time = 0
        self.error_msg = ""
        
        # Static Resources
        self.linear_sampler = device.create_sampler(
            min_filter=spy.TextureFilteringMode.linear,
            mag_filter=spy.TextureFilteringMode.linear,
            mip_filter=spy.TextureFilteringMode.linear,
            address_u=spy.TextureAddressingMode.wrap, 
            address_v=spy.TextureAddressingMode.wrap,
            address_w=spy.TextureAddressingMode.wrap
        )

        self.volume_tex = device.create_texture(
            type=spy.TextureType.texture_3d, format=spy.Format.r32_float,
            width=VOL_SIZE, height=VOL_SIZE, depth=VOL_SIZE,
            usage=spy.TextureUsage.shader_resource, label="VDBVolume"
        )
        # Upload data immediately
        cmd = device.create_command_encoder()
        cmd.upload_texture_data(self.volume_tex, [volume_data])
        device.submit_command_buffer(cmd.finish())

        self.cam_buffer = device.create_buffer(size=64, usage=spy.BufferUsage.shader_resource, memory_type=spy.MemoryType.upload)
        self.settings_buffer = device.create_buffer(size=128, usage=spy.BufferUsage.shader_resource, memory_type=spy.MemoryType.upload)

        # Dynamic Resources (Resizeable)
        self.screen_tex = None
        self.display_gl_tex = None
        self.width, self.height = 0, 0

        # Gaussian stuff
        self.gaussian_volume_tex = device.create_texture(
            type=spy.TextureType.texture_3d,
            format=spy.Format.r32_float,
            width=VOL_SIZE, height=VOL_SIZE, depth=VOL_SIZE,
            usage=spy.TextureUsage.unordered_access | spy.TextureUsage.shader_resource,
            label="GaussianVolume"
        )

        self.gaussian_buffer = None
        self.gaussian_count = 0

        self.gaussian_raster_program = device.load_program(
            "3drasterizer.slang",
            ["main"]
        )

        self.gaussian_raster_pipeline = device.create_compute_pipeline(
            program=self.gaussian_raster_program
        )

        self.use_gaussian_volume = False

        # Initial Compile
        self.check_hot_reload()

    def resize(self, w, h):
        if w == self.width and h == self.height: return
        self.width, self.height = w, h
        
        # Slang Texture
        self.screen_tex = self.device.create_texture(
            format=spy.Format.rgba32_float, width=w, height=h,
            usage=spy.TextureUsage.render_target, label="Screen"
        )
        
        # OpenGL Texture (for display)
        if self.display_gl_tex is None: self.display_gl_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.display_gl_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)

    def check_hot_reload(self):
        try:
            curr_time = Path(SHADER_FILE).stat().st_mtime
            if curr_time > self.last_mod_time:
                self.last_mod_time = curr_time
                
                # Compile
                prog = self.device.load_program(SHADER_FILE, ["vertex_main", "fragment_main"])
                new_pipe = self.device.create_render_pipeline(
                    program=prog,
                    input_layout=self.device.create_input_layout(input_elements=[], vertex_streams=[]),
                    targets=[{"format": spy.Format.rgba32_float}]
                )
                
                self.pipeline = new_pipe
                self.error_msg = ""
                print("Shader Reloaded!")
        except Exception as e:
            self.error_msg = str(e)
            print("Shader Compile Error (Safe)")

    def render(self, camera, settings):
        # Safety check
        if not self.pipeline: return

        self.cam_buffer.copy_from_numpy(camera.get_gpu_data(self.width / self.height))
        
        # PACK SETTINGS BUFFER (24 floats only!!!!)
        s_data = np.zeros(24, dtype=np.float32)
        
        # [0-6] Basic
        s_data[0] = settings.step_size
        s_data[1] = settings.density_scale
        s_data[2] = settings.density_curve
        s_data[3] = float(settings.step_count)
        s_data[4:7] = settings.smoke_color
        
        # [7-8] Params
        s_data[7] = settings.light_penetration
        s_data[8] = settings.phase_g
        
        # [9-11] Sun Dir
        s_data[9:12] = settings.get_sun_dir()
        
        # [12-14] Sun Color (Base * Intensity)
        s_data[12] = settings.sun_color_base[0] * settings.sun_intensity
        s_data[13] = settings.sun_color_base[1] * settings.sun_intensity
        s_data[14] = settings.sun_color_base[2] * settings.sun_intensity
        
        # [15-17] Ambient Color (Base * Intensity)
        s_data[15] = settings.ambient_color_base[0] * settings.ambient_intensity
        s_data[16] = settings.ambient_color_base[1] * settings.ambient_intensity
        s_data[17] = settings.ambient_color_base[2] * settings.ambient_intensity
        
        # [18-19] Shadows
        s_data[18] = float(settings.shadow_steps)
        s_data[19] = settings.shadow_step_mult

        self.settings_buffer.copy_from_numpy(s_data)

        # Encode
        cmd = self.device.create_command_encoder()
        cmd.set_texture_state(self.volume_tex, spy.ResourceState.shader_resource)

        with cmd.begin_render_pass({"color_attachments": [{"view": self.screen_tex.create_view({})}]}) as rp:
            rp.bind_pipeline(self.pipeline)
            cursor = spy.ShaderCursor(rp.bind_pipeline(self.pipeline))
            
            if self.use_gaussian_volume:
                cursor["inVolume"] = self.gaussian_volume_tex
            else:
                cursor["inVolume"] = self.volume_tex
            cursor["camera"] = self.cam_buffer
            cursor["linearSampler"] = self.linear_sampler
            cursor["settings"] = self.settings_buffer
            
            rp.set_render_state({
                "viewports": [spy.Viewport.from_size(self.width, self.height)],
                "scissor_rects": [spy.ScissorRect.from_size(self.width, self.height)]
            })
            rp.draw({"vertex_count": 3})

        self.device.submit_command_buffer(cmd.finish())

    def update_display(self):
        # Copy Slang -> OpenGL
        pixels = (np.clip(self.screen_tex.to_numpy(), 0, 1) * 255).astype(np.uint8)
        glBindTexture(GL_TEXTURE_2D, self.display_gl_tex)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE, pixels)
        
        # Blit to Framebuffer
        fb = glGenFramebuffers(1)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fb)
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.display_gl_tex, 0)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        glBlitFramebuffer(0, 0, self.width, self.height, 0, 0, self.width, self.height, GL_COLOR_BUFFER_BIT, GL_NEAREST)
        glDeleteFramebuffers(1, [fb])
    
    def rasterize_gaussians(self, gaussians, vol_min, vol_max):
        if gaussians is None or len(gaussians) == 0:
            print("No gaussians to rasterize")
            return

        self.gaussian_count = len(gaussians)

        self.gaussian_buffer = self.device.create_buffer(
            size=gaussians.nbytes,
            usage=spy.BufferUsage.shader_resource,
            memory_type=spy.MemoryType.upload
        )
        self.gaussian_buffer.copy_from_numpy(gaussians)

        # Clear volume
        cmd = self.device.create_command_encoder()
        cmd.clear_texture_float(self.gaussian_volume_tex)

        cmd.set_texture_state(self.gaussian_volume_tex, spy.ResourceState.unordered_access)

        with cmd.begin_compute_pass() as cp:
            
            cursor = spy.ShaderCursor(cp.bind_pipeline(self.gaussian_raster_pipeline))

            cursor["gGaussians"] = self.gaussian_buffer
            cursor["gGaussianVolume"] = self.gaussian_volume_tex

            cursor["GaussianParams"]["volumeMinWorld"] = tuple(vol_min)
            cursor["GaussianParams"]["volumeMaxWorld"] = tuple(vol_max)

            cursor["GaussianParams"]["gaussianCount"] = self.gaussian_count
            cursor["GaussianParams"]["volumeResolution"] = (VOL_SIZE, VOL_SIZE, VOL_SIZE)
            cursor["GaussianParams"]["voxelSize"] = 1.0

            groups = uint3(
                VOL_SIZE,
                VOL_SIZE,
                VOL_SIZE
            )
            cp.dispatch(groups)

        self.device.submit_command_buffer(cmd.finish())

        print(f"Rasterized {self.gaussian_count} gaussians")


# ==========================================
# 3. APP CLASS
# ==========================================

class App:
    def __init__(self):
        # Window
        if not glfw.init(): raise Exception("GLFW failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.width, self.height = 1024, 768
        self.window = glfw.create_window(self.width, self.height, "VDB Editor", None, None)
        glfw.make_context_current(self.window)

        # ImGui
        imgui.create_context()
        self.io = imgui.get_io()
        self.io.config_flags |= imgui.ConfigFlags_.docking_enable
        window_address = ctypes.cast(self.window, ctypes.c_void_p).value
        imgui.backends.glfw_init_for_opengl(window_address, True)
        imgui.backends.opengl3_init("#version 330")

        # Logic
        self.settings = Settings()
        self.camera = Camera(self.width, self.height)
        
        # Renderer
        example_dir = Path(__file__).parent
        self.device = spy.Device(enable_debug_layers=True, compiler_options={"include_paths": [example_dir]})
        
        # Load Data
        grid = load_vdb_grid(VDB_FILE)

        self.vol_min_world, self.vol_max_world, vol_data = convert_grid_to_dense_volume(grid, VOL_SIZE)

        self.gaussians = convert_grid_to_gaussians(
            grid,
            probability_scale=0.01,
            sigma_scale=5.0,
            jitter_scale=0.5
        )

        self.renderer = Renderer(self.device, vol_data)
        self.renderer.resize(self.width, self.height)

        self.renderer.rasterize_gaussians(self.gaussians, self.vol_min_world, self.vol_max_world)


    def run(self):
        last_time = glfw.get_time()
        
        while not glfw.window_should_close(self.window):
            curr_time = glfw.get_time()
            dt = curr_time - last_time
            last_time = curr_time
            
            glfw.poll_events()
            
            # Resize
            w, h = glfw.get_window_size(self.window)
            if w == 0 or h == 0: continue
            self.renderer.resize(w, h)

            # Hot Reload
            self.renderer.check_hot_reload()

            # ImGui Frame
            imgui.backends.opengl3_new_frame()
            imgui.backends.glfw_new_frame()
            imgui.new_frame()
            imgui.dock_space_over_viewport(0, imgui.get_main_viewport(), imgui.DockNodeFlags_.passthru_central_node)

            # Draw UI
            self.draw_ui(dt)

            # Input
            self.handle_input(dt)

            # Render Scene
            try:
                self.renderer.render(self.camera, self.settings)
                self.renderer.update_display()
            except Exception as e:
                print(f"Runtime Render Error: {e}")

            # Render UI
            imgui.render()
            imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())
            glfw.swap_buffers(self.window)

        self.cleanup()

    def handle_input(self, dt):
        if self.io.want_capture_mouse: return
        
        # Mouse
        right_down = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        if right_down and not self.camera.is_dragging:
            self.camera.is_dragging = True
            self.camera.first_mouse = True
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        elif not right_down and self.camera.is_dragging:
            self.camera.is_dragging = False
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_NORMAL)
            
        if self.camera.is_dragging:
            x, y = glfw.get_cursor_pos(self.window)
            self.camera.process_mouse(x, y)

        # Keyboard
        speed = self.camera.speed * dt
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS: self.camera.pos += self.camera.front * speed
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS: self.camera.pos -= self.camera.front * speed
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS: self.camera.pos -= self.camera.right * speed
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS: self.camera.pos += self.camera.right * speed
        if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS: self.camera.pos += self.camera.up * speed
        if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS: self.camera.pos -= self.camera.up * speed
        if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS: glfw.set_window_should_close(self.window, True)

    def draw_ui(self, dt):
        if imgui.begin("Stats"):
            imgui.text(f"FPS: {1.0/(dt+0.0001):.1f}")
            if self.renderer.error_msg:
                imgui.text_colored(imgui.ImVec4(1, 0, 0, 1), "SHADER ERROR")
        imgui.end()

        if imgui.begin("Settings"):
            if self.renderer.error_msg:
                imgui.text_colored(imgui.ImVec4(1, 0, 0, 1), f"{self.renderer.error_msg}")
            else:
                imgui.text_colored(imgui.ImVec4(0, 1, 0, 1), "Shader Active")
            
            imgui.text("Raymarcher")
            imgui.separator()
            _, self.settings.density_scale = imgui.slider_float("Density", self.settings.density_scale, 1.0, 200.0)
            _, self.settings.density_curve = imgui.slider_float("Gamma", self.settings.density_curve, 0.1, 2.0)
            _, self.settings.step_size = imgui.slider_float("Step Size", self.settings.step_size, 0.001, 0.05)
            _, self.settings.step_count = imgui.slider_int("Max Steps", self.settings.step_count, 10, 500)
            
            imgui.dummy((0, 10))
            imgui.text("Lighting")
            imgui.separator()
            
            # Sun Controls
            _, self.settings.sun_direction = imgui.slider_float3("Sun Pitch", self.settings.sun_direction, -1.0, 1.0)
            _, self.settings.sun_intensity = imgui.slider_float("Sun Intensity", self.settings.sun_intensity, 0.0, 20.0)
            _, self.settings.sun_color_base = imgui.color_edit3("Sun Color", self.settings.sun_color_base)
            
            # Ambient
            _, self.settings.ambient_intensity = imgui.slider_float("Amb Intensity", self.settings.ambient_intensity, 0.0, 2.0)
            _, self.settings.ambient_color_base = imgui.color_edit3("Amb Color", self.settings.ambient_color_base)

            imgui.dummy((0, 10))
            imgui.text("Volumetric Look")
            imgui.separator()
            
            # The Magic Sliders
            _, self.settings.light_penetration = imgui.slider_float("Fluffiness (Penetration)", self.settings.light_penetration, 0.01, 1.0)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Low = Light goes deep (White Clouds)\nHigh = Light blocked early (Dark Smoke)")
                
            _, self.settings.phase_g = imgui.slider_float("Silver Lining (Phase G)", self.settings.phase_g, -0.9, 0.9)
            
            _, self.settings.shadow_steps = imgui.slider_int("Shadow Steps", self.settings.shadow_steps, 1, 16)
            _, self.settings.shadow_step_mult = imgui.slider_float("Shadow Step Mult", self.settings.shadow_step_mult, 1.0, 10.0)

            imgui.dummy((0, 10))
            _, self.settings.smoke_color = imgui.color_edit3("Smoke Albedo", self.settings.smoke_color)
            
            changed, self.renderer.use_gaussian_volume = imgui.checkbox("Render Gaussian Volume", self.renderer.use_gaussian_volume)
            
        imgui.end()

    def cleanup(self):
        imgui.backends.opengl3_shutdown()
        imgui.backends.glfw_shutdown()
        imgui.destroy_context()
        glfw.terminate()

# ==========================================
# 4. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    app = App()
    app.run()