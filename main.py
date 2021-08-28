import threading
import queue
import time

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import icecream


def cursor_cbk(window, x, y):
    q.put((window, x/720))


def main(ver, green):
    if not glfw.init():
        return

    window = glfw.create_window(720, 600, f"Opengl GLFW Triangle {ver}", None, None)

    if not window:
        glfw.terminate()
        return None

    glfw.make_context_current(window)
    triangle = [
        -0.5, -0.5, 0.0,
         0.5, -0.5, 0.0,
         0.0,  0.5, 0.0
    ]

    glfw.set_cursor_pos_callback(window, cursor_cbk)

    # convert to 32bit float
    triangle = np.array(triangle, dtype = np.float32)
   
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


def sci3d_thread():
    w1, u1 = main('1', 1)

    w2, u2 = main('2', 0.1)

    ended = False
    w1_closed = False
    w2_closed = False
    while not ended:
        ended = True

        glfw.poll_events()
        if not glfw.window_should_close(w1) and not w1_closed:
            glfw.make_context_current(w1)
            glClear(GL_COLOR_BUFFER_BIT)

            # Draw Triangle
            glDrawArrays(GL_TRIANGLES, 0, 3)
            glfw.swap_buffers(w1)
            ended = False
        elif not w1_closed:
            glfw.destroy_window(w1)
            w1_closed = True

        if not glfw.window_should_close(w2) and not w2_closed:
            glfw.make_context_current(w2)
            glClear(GL_COLOR_BUFFER_BIT)

            # Draw Triangle
            glDrawArrays(GL_TRIANGLES, 0, 3)
            glfw.swap_buffers(w2)
            ended = False
        elif not w2_closed:
            glfw.destroy_window(w2)
            w2_closed = True

        while not q.empty():
            w, v = q.get()
            glfw.make_context_current(w)
            glUniform4f(u1, v, v, v, 1)

    glfw.terminate()


if __name__ == "__main__":
    icecream.install()
    t = threading.Thread(target=sci3d_thread, daemon=False)
    t.start()
    q = queue.Queue()

    while True:
        time.sleep(0.1)

    #t.join()
