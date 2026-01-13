import slangpy as spy
import numpy as np
import glfw
from OpenGL.GL import *
from pathlib import Path
import math
import ctypes 
from imgui_bundle import imgui

# --- CONFIG ---
VDB_FILE = "cloud_01_variant_0000.vdb" 
VOL_SIZE = 128
SHADER_FILE = "hybrid.slang"

# ==========================================
# 1. DATA & LOGIC CLASSES
# ==========================================

class Settings:
    def __init__(self):
        self.step_size = 0.015
        self.density_scale = 40.0
        self.density_curve = 0.6
        self.step_count = 256
        self.smoke_color = [0.9, 0.95, 1.0]

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

def load_vdb_volume(filename, size):
    print(f"Loading {filename}...")
    try:
        import openvdb as vdb
        if not Path(filename).exists(): raise Exception("File missing")
        raw = vdb.readAll(filename)
        grid = raw[0][0] if isinstance(raw, (list, tuple)) else raw 
        
        bbox = grid.evalActiveVoxelBoundingBox()
        min_i, max_i = np.array(bbox[0]), np.array(bbox[1])
        center = (min_i + max_i) / 2.0
        
        # Padding fix: 1.2x largest extent
        extent = np.max(max_i - min_i)
        max_dim = extent * 1.2 
        
        accessor = grid.getAccessor()
        data = np.zeros((size, size, size), dtype=np.float32)
        inv_size = 1.0 / size
        
        import itertools
        for z, y, x in itertools.product(range(size), range(size), range(size)):
            uvw = (np.array([x, y, z]) * inv_size) - 0.5
            pos = center + uvw * max_dim
            data[z, y, x] = accessor.getValue(tuple(pos.astype(int)))
            
        m = np.max(data)
        if m > 0: data /= m
        return np.ascontiguousarray(data, dtype=np.float32)
    except Exception as e:
        print(f"Error loading VDB, using fallback noise: {e}")
        x = np.linspace(-1, 1, size)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        return np.exp(-4 * (X**2 + Y**2 + Z**2)).astype(np.float32)

# ==========================================
# 2. RENDERER SYSTEM (The Engine)
# ==========================================

class Renderer:
    def __init__(self, device, volume_data):
        self.device = device
        self.pipeline = None
        self.last_mod_time = 0
        self.error_msg = ""
        
        # 1. Static Resources
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
        self.settings_buffer = device.create_buffer(size=64, usage=spy.BufferUsage.shader_resource, memory_type=spy.MemoryType.upload)

        # 2. Dynamic Resources (Resizeable)
        self.screen_tex = None
        self.display_gl_tex = None
        self.width, self.height = 0, 0

        # 3. Initial Compile
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

        # Update Uniforms
        self.cam_buffer.copy_from_numpy(camera.get_gpu_data(self.width / self.height))
        
        s_data = np.zeros(8, dtype=np.float32)
        s_data[0] = settings.step_size
        s_data[1] = settings.density_scale
        s_data[2] = settings.density_curve
        s_data[3] = float(settings.step_count)
        s_data[4:7] = settings.smoke_color
        self.settings_buffer.copy_from_numpy(s_data)

        # Encode
        cmd = self.device.create_command_encoder()
        cmd.set_texture_state(self.volume_tex, spy.ResourceState.shader_resource)

        with cmd.begin_render_pass({"color_attachments": [{"view": self.screen_tex.create_view({})}]}) as rp:
            rp.bind_pipeline(self.pipeline)
            cursor = spy.ShaderCursor(rp.bind_pipeline(self.pipeline))
            
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

# ==========================================
# 3. APP CLASS (Window & UI)
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
        vol = load_vdb_volume(VDB_FILE, VOL_SIZE)
        self.renderer = Renderer(self.device, vol)
        self.renderer.resize(self.width, self.height)

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
            
            imgui.separator()
            _, self.settings.density_scale = imgui.slider_float("Density", self.settings.density_scale, 1.0, 200.0)
            _, self.settings.step_size = imgui.slider_float("Step Size", self.settings.step_size, 0.001, 0.05)
            _, self.settings.step_count = imgui.slider_int("Step Count", self.settings.step_count, 10, 2000)
            _, self.settings.density_curve = imgui.slider_float("Gamma", self.settings.density_curve, 0.1, 2.0)
            _, self.settings.smoke_color = imgui.color_edit3("Color", self.settings.smoke_color)
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