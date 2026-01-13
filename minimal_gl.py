# minimal_gl.py
import glfw
import numpy as np
from OpenGL.GL import *
import torch

class Simple3DViewer:
    def __init__(self, width=1024, height=768, title="Slang Prototype"):
        if not glfw.init(): raise Exception("GLFW failed")
        self.window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(self.window)
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        
        # Quad Setup
        quad = np.array([-1,1,0,1, -1,-1,0,0, 1,-1,1,0, -1,1,0,1, 1,-1,1,0, 1,1,1,1], dtype=np.float32)
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
        
        self.tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        
        # Camera State
        self.pos = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        self.yaw, self.pitch = -90.0, 0.0
        self.last_x, self.last_y = width/2, height/2
        self.first_mouse = True
        
        glfw.set_cursor_pos_callback(self.window, self.mouse_callback)
        self.width, self.height = width, height

    def mouse_callback(self, w, x, y):
        if self.first_mouse: self.last_x, self.last_y = x, y; self.first_mouse = False
        self.yaw += (x - self.last_x) * 0.1
        self.pitch = max(-89, min(89, self.pitch + (self.last_y - y) * 0.1))
        self.last_x, self.last_y = x, y

    def get_camera_matrices(self):
        # Returns (view_inv, proj_inv) tensors for Slang
        front = np.array([
            np.cos(np.radians(self.yaw)) * np.cos(np.radians(self.pitch)),
            np.sin(np.radians(self.pitch)),
            np.sin(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
        ])
        fwd = front / np.linalg.norm(front)
        right = np.cross(fwd, [0,1,0]); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        
        # WASD
        s = 0.05
        if glfw.get_key(self.window, glfw.KEY_W): self.pos += fwd * s
        if glfw.get_key(self.window, glfw.KEY_S): self.pos -= fwd * s
        if glfw.get_key(self.window, glfw.KEY_A): self.pos -= right * s
        if glfw.get_key(self.window, glfw.KEY_D): self.pos += right * s

        # View Matrix Construction
        view = np.eye(4, dtype=np.float32)
        view[:3, 0], view[:3, 1], view[:3, 2] = right, up, -fwd
        view[:3, 3] = -np.dot(view[:3,:3], self.pos)
        
        # Proj Matrix
        f = 1.0 / np.tan(np.radians(30)) # 60 fov
        proj = np.diag([f/(self.width/self.height), f, -1, 0]).astype(np.float32)
        proj[2,3], proj[3,2] = -0.1, -1
        
        return torch.tensor(np.linalg.inv(view)).cuda(), torch.tensor(np.linalg.inv(proj)).cuda()

    def render_loop(self, render_func):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            if glfw.get_key(self.window, glfw.KEY_ESCAPE): break
            
            # 1. Run user's Slang logic
            view_inv, proj_inv = self.get_camera_matrices()
            output_tensor = render_func(view_inv, proj_inv)
            
            # 2. Blit to Screen
            data = output_tensor.cpu().numpy() # Sync point
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0, GL_RGBA, GL_FLOAT, data)
            glDrawArrays(GL_TRIANGLES, 0, 6)
            glfw.swap_buffers(self.window)
        glfw.terminate()