import slangpy as spy
import numpy as np
import glfw
from OpenGL.GL import *
from pathlib import Path
import math

# --- CONFIG ---
VDB_FILE = "cloud_01_variant_0000.vdb" 
WIDTH, HEIGHT = 1024, 768
VOL_SIZE = 128

g_width, g_height = 1024, 768
g_screen_texture = None
g_display_tex = None

# --- CAMERA CLASS (Improved) ---
class Camera:
    def __init__(self):
        self.pos = np.array([0.0, 0.0, 2.5], dtype=np.float32)
        self.yaw = -90.0
        self.pitch = 0.0
        self.front = np.array([0, 0, -1], dtype=np.float32)
        self.right = np.array([1, 0, 0], dtype=np.float32)
        self.up = np.array([0, 1, 0], dtype=np.float32)
        self.speed = 2.0
        self.sensitivity = 0.1
        
        # State tracking
        self.last_x = WIDTH / 2.0
        self.last_y = HEIGHT / 2.0
        self.first_mouse = True
        self.is_dragging = False # Track if we are currently looking around

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
        # 1. If we just started dragging, reset the anchor to avoid jumps
        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False
            return # Skip the first frame of movement

        # 2. Calculate Delta
        xoffset = (xpos - self.last_x) * self.sensitivity
        yoffset = (self.last_y - ypos) * self.sensitivity 
        
        self.last_x = xpos
        self.last_y = ypos

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

cam = Camera()

# --- VDB LOADER ---
def load_volume_data(size):
    print(f"Loading {VDB_FILE}...")
    try:
        import openvdb as vdb
        if not Path(VDB_FILE).exists(): raise Exception("File missing")
        raw = vdb.readAll(VDB_FILE)
        grid = raw[0][0] if isinstance(raw, (list, tuple)) else raw 
        
        bbox = grid.evalActiveVoxelBoundingBox()
        min_i, max_i = np.array(bbox[0]), np.array(bbox[1])
        center = (min_i + max_i) / 2.0
        max_dim = np.max(max_i - min_i) * 1.1
        
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
        print(f"Fallback Noise ({e})")
        x = np.linspace(-1, 1, size)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        return np.exp(-4 * (X**2 + Y**2 + Z**2)).astype(np.float32)

# --- SETUP ---
EXAMPLE_DIR = Path(__file__).parent
device = spy.Device(enable_debug_layers=True, compiler_options={"include_paths": [EXAMPLE_DIR]})

linear_sampler = device.create_sampler(
    min_filter=spy.TextureFilteringMode.linear,
    mag_filter=spy.TextureFilteringMode.linear,
    mip_filter=spy.TextureFilteringMode.linear,
    address_u=spy.TextureAddressingMode.wrap,
    address_v=spy.TextureAddressingMode.wrap,
    address_w=spy.TextureAddressingMode.wrap
)

volume_data = load_volume_data(VOL_SIZE)
volume_texture = device.create_texture(
    type=spy.TextureType.texture_3d, format=spy.Format.r32_float,
    width=VOL_SIZE, height=VOL_SIZE, depth=VOL_SIZE,
    usage=spy.TextureUsage.shader_resource, label="VDBVolume"
)
cmd = device.create_command_encoder()
cmd.upload_texture_data(volume_texture, [volume_data])
device.submit_command_buffer(cmd.finish())

cam_buffer = device.create_buffer(
    size=16 * 4,
    usage=spy.BufferUsage.shader_resource, 
    memory_type=spy.MemoryType.upload,
    label="CameraUniforms"
)

screen_texture = device.create_texture(
    format=spy.Format.rgba32_float, width=WIDTH, height=HEIGHT,
    usage=spy.TextureUsage.render_target, label="Screen"
)

graphics_program = device.load_program("hybrid.slang", ["vertex_main", "fragment_main"])
render_pipeline = device.create_render_pipeline(
    program=graphics_program,
    input_layout=device.create_input_layout(input_elements=[], vertex_streams=[]),
    targets=[{"format": spy.Format.rgba32_float}],
)

# --- WINDOW & INPUT ---
# --- WINDOW ---
if not glfw.init(): raise Exception("GLFW failed")
window = glfw.create_window(g_width, g_height, "Right Click to Fly | Resizeable", None, None)
glfw.make_context_current(window)

# --- RESIZE LOGIC ---
def recreate_resources(w, h):
    global g_screen_texture, g_display_tex, g_width, g_height
    g_width, g_height = w, h
    
    # 1. Slang Render Target
    g_screen_texture = device.create_texture(
        format=spy.Format.rgba32_float, width=w, height=h,
        usage=spy.TextureUsage.render_target, label="Screen"
    )
    
    # 2. OpenGL Display Texture
    if g_display_tex is None:
        g_display_tex = glGenTextures(1)
        
    glBindTexture(GL_TEXTURE_2D, g_display_tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    # Reallocate storage for new size
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)

# Initial Create
recreate_resources(g_width, g_height)

def mouse_callback(window, xpos, ypos):
    if cam.is_dragging:
        cam.process_mouse(xpos, ypos)
glfw.set_cursor_pos_callback(window, mouse_callback)

# --- MAIN LOOP ---
last_time = glfw.get_time()

while not glfw.window_should_close(window):
    current_time = glfw.get_time()
    dt = current_time - last_time
    last_time = current_time

    # 1. CHECK RESIZE
    win_w, win_h = glfw.get_window_size(window)
    # Handle minimization (size 0) by skipping
    if win_w == 0 or win_h == 0:
        glfw.poll_events()
        continue
        
    if win_w != g_width or win_h != g_height:
        recreate_resources(win_w, win_h)

    # 2. INPUT
    right_mouse_down = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
    if right_mouse_down and not cam.is_dragging:
        cam.is_dragging = True
        cam.first_mouse = True 
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    elif not right_mouse_down and cam.is_dragging:
        cam.is_dragging = False
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)

    speed = cam.speed * dt
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS: cam.pos += cam.front * speed
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS: cam.pos -= cam.front * speed
    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS: cam.pos -= cam.right * speed
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS: cam.pos += cam.right * speed
    if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS: cam.pos += cam.up * speed
    if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS: cam.pos -= cam.up * speed
    if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS: glfw.set_window_should_close(window, True)

    # 3. RENDER
    # Pass new Aspect Ratio (win_w / win_h)
    cam_data = cam.get_gpu_data(win_w / win_h)
    cam_buffer.copy_from_numpy(cam_data)

    cmd = device.create_command_encoder()
    cmd.set_texture_state(volume_texture, spy.ResourceState.shader_resource)

    with cmd.begin_render_pass({"color_attachments": [{"view": g_screen_texture.create_view({})}]}) as rp:
        shader = rp.bind_pipeline(render_pipeline)
        cursor = spy.ShaderCursor(shader)
        cursor["inVolume"] = volume_texture
        cursor["camera"] = cam_buffer
        cursor["linearSampler"] = linear_sampler
        
        # Update Viewport/Scissor to match new size
        rp.set_render_state({
            "viewports": [spy.Viewport.from_size(g_width, g_height)],
            "scissor_rects": [spy.ScissorRect.from_size(g_width, g_height)]
        })
        rp.draw({"vertex_count": 3})

    device.submit_command_buffer(cmd.finish())

    # 4. DISPLAY
    pixels = (np.clip(g_screen_texture.to_numpy(), 0, 1) * 255).astype(np.uint8)
    glBindTexture(GL_TEXTURE_2D, g_display_tex)
    # Use global width/height
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, g_width, g_height, GL_RGBA, GL_UNSIGNED_BYTE, pixels)
    
    fb = glGenFramebuffers(1)
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fb)
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g_display_tex, 0)
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
    glBlitFramebuffer(0, 0, g_width, g_height, 0, 0, g_width, g_height, GL_COLOR_BUFFER_BIT, GL_NEAREST)
    glDeleteFramebuffers(1, [fb])

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()