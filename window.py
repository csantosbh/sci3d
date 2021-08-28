import threading
import queue
import time

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import icecream


class Window(object):
    def __init__(self):
        window = glfw.create_window(720, 600, f"Opengl GLFW Triangle {ver}", None, None)

        if not window:
            glfw.terminate()
            return None

        glfw.make_context_current(window)
        triangle = [
            -0.5, -0.5, 0.0,
            0.5, -0.5, 0.0,
            0.0, 0.5, 0.0
        ]

        glfw.set_cursor_pos_callback(window, cursor_cbk)

        # convert to 32bit float
        triangle = np.array(triangle, dtype=np.float32)

        VERTEX_SHADER = """
            #version 330
            in vec4 position;
            void main() {
            gl_Position = position;
        }
        """
        FRAGMENT_SHADER = f"""
            #version 330
            uniform vec4 tri_color;
            void main() {{
            gl_FragColor = 
            vec4(1.0f, {green},1.0f,1.0f) * tri_color;
            }}
        """
        # Compile The Program and shaders
        shader = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        )

        # Create Buffer object in gpu
        VBO = glGenBuffers(1)

        # Bind the buffer
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, 36, triangle, GL_STATIC_DRAW)

        # Get the position from vertex shader
        position = glGetAttribLocation(shader, 'position')
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(position)
        glUseProgram(shader)

        glClearColor(0.0, 0.0, 1.0, 1.0)
        unif = glGetUniformLocation(shader, 'tri_color')
        glUniform4f(unif, 1, 1, 1, 1)

        return window, unif
