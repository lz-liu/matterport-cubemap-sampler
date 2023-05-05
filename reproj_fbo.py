import argparse
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from PIL import Image

VERTEX_SHADER = """
#version 300 es

precision highp float;

layout(location = 0) in vec3 position;
out vec3 texCoords;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    texCoords = position;
    gl_Position = projection * view * vec4(position, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 300 es

precision highp float;

in vec3 texCoords;
out vec4 fragColor;

uniform samplerCube cubemap;

void main()
{
    fragColor = texture(cubemap, texCoords);
}
"""

skybox_data = np.array(
    [
        # right
        1., -1., -1.,
        1., -1.,  1.,
        1.,  1.,  1.,
        1.,  1.,  1.,
        1.,  1., -1.,
        1., -1., -1.,
        # left
        -1., -1.,  1.,
        -1., -1., -1.,
        -1.,  1., -1.,
        -1.,  1., -1.,
        -1.,  1.,  1.,
        -1., -1.,  1.,
        # top
        -1.,  1., -1.,
        1.,  1., -1.,
        1.,  1.,  1.,
        1.,  1.,  1.,
        -1.,  1.,  1.,
        -1.,  1., -1.,
        # bottom
        -1., -1., -1.,
        -1., -1.,  1.,
        1., -1., -1.,
        1., -1., -1.,
        -1., -1.,  1.,
        1., -1.,  1.,
        # front
        -1., -1.,  1.,
        -1.,  1.,  1.,
        1.,  1.,  1.,
        1.,  1.,  1.,
        1., -1.,  1.,
        -1., -1.,  1.,
        # back
        -1.,  1., -1.,
        -1., -1., -1.,
        1., -1., -1.,
        1., -1., -1.,
        1.,  1., -1.,
        -1.,  1., -1.,
    ]
    ,dtype=np.float32
)

# Define and initialize camera rotation matrix
camera_rotation_matrix = np.identity(3, dtype=np.float32)


def load_cubemap(faces):
    """Load a cubemap texture from a list of image files.

    Args:
        faces (tuple): Tuple of image file paths for the cubemap faces in the order (right, left, top, bottom, front, back).

    Returns:
        int: Texture ID of the loaded cubemap.
    """
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_CUBE_MAP, texture_id)

    for i, face in enumerate(faces):
        img = Image.open(face)
        if i == 2:
            # Rotate the bottom image 90 degress clockwise.
            img = img.rotate(-90)
        elif i == 3:
            # Rotate the top image 90 degrees counter-clockwise.
            img = img.rotate(90)
        img_data = np.array(list(img.getdata()), np.uint8)
        glTexImage2D(
            GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
            0,
            GL_RGB,
            img.width,
            img.height,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            img_data,
        )
        img.close()
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
    return texture_id


def compile_shader(shader_type, shader_source):
    """Compile a shader from its source code.

    Args:
        shader_type (int): Type of the shader (e.g., GL_VERTEX_SHADER or GL_FRAGMENT_SHADER).
        shader_source (str): Source code of the shader.

    Returns:
        int: Compiled shader object.
    """
    shader = glCreateShader(shader_type)
    glShaderSource(shader, shader_source)
    glCompileShader(shader)

    # Check for compilation errors
    result = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if result != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))

    return shader


def create_shader_program(vertex_shader_source, fragment_shader_source):
    """Create a shader program from vertex and fragment shader source codes.

    Args:
        vertex_shader_source (str): Source code of the vertex shader.
        fragment_shader_source (str): Source code of the fragment shader.

    Returns:
        int: Compiled shader program.
    """
    vertex_shader = compile_shader(GL_VERTEX_SHADER, vertex_shader_source)
    fragment_shader = compile_shader(GL_FRAGMENT_SHADER, fragment_shader_source)

    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)

    # Check for linking errors
    result = glGetProgramiv(program, GL_LINK_STATUS)
    if result != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(program))

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return program


def render_cubemap_image(cubemap_faces):
    """Render a cubemap image using OpenGL.

    Args:
        cubemap_faces (tuple): Tuple of image file paths for the cubemap faces in the order (right, left, top, bottom, front, back).
    """
    global camera_rotation_matrix

    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(2048, 2048)
    glutCreateWindow("Matterport Visualizer")

    shader_program = create_shader_program(VERTEX_SHADER, FRAGMENT_SHADER)
    glUseProgram(shader_program)

    skyboxVAO = glGenVertexArrays(1)
    glBindVertexArray(skyboxVAO)
    skyboxVBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO)
    glBufferData(
        GL_ARRAY_BUFFER,
        size=skybox_data.size * sizeof(GLfloat),
        data=skybox_data,
        usage=GL_STATIC_DRAW,
    )
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), None)

    # Create an off-screen framebuffer object (FBO)
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)

    # Create a texture for rendering
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA, 2048, 2048, 0, GL_RGBA, GL_UNSIGNED_BYTE, None
    )
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0
    )

    cubemap_texture = load_cubemap(cubemap_faces)

    glClearColor(0, 0, 0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    projection_uniform = glGetUniformLocation(shader_program, "projection")
    projection_matrix = np.identity(4, dtype=np.float32)
    projection_matrix = perspective_projection(90, 1.0, 0.01, 100.0)
    glUniformMatrix4fv(projection_uniform, 1, GL_FALSE, projection_matrix.transpose())

    view_uniform = glGetUniformLocation(shader_program, "view")
    view_matrix = np.identity(4, dtype=np.float32)
    # Apply the camera rotation
    view_matrix[0:3, 0:3] = camera_rotation_matrix
    glUniformMatrix4fv(view_uniform, 1, GL_FALSE, view_matrix.transpose())

    cubemap_uniform = glGetUniformLocation(shader_program, "cubemap")
    glUniform1i(cubemap_uniform, 0)

    glBindVertexArray(skyboxVAO)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap_texture)
    glDrawArrays(GL_TRIANGLES, 0, 36)
    glBindVertexArray(0)

    # Read pixel data from the framebuffer
    pixels = glReadPixels(0, 0, 2048, 2048, GL_RGBA, GL_UNSIGNED_BYTE)
    # Create an image using PIL
    image = Image.frombytes("RGBA", (2048, 2048), pixels)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    # Save the image
    image.save("/images/output.png")


def perspective_projection(fov, aspect_ratio, near, far):
    """Create a perspective projection matrix.

    Args:
        fov (float): Field of view angle in degrees.
        aspect_ratio (float): Aspect ratio of the viewport.
        near (float): Distance to the near clipping plane.
        far (float): Distance to the far clipping plane.

    Returns:
        numpy.ndarray: Perspective projection matrix.
    """
    projection_matrix = np.zeros((4, 4), dtype=np.float32)
    f = 1.0 / np.tan(np.radians(fov) / 2.0)
    projection_matrix[0, 0] = f / aspect_ratio
    projection_matrix[1, 1] = f
    projection_matrix[2, 2] = (far + near) / (near - far)
    projection_matrix[2, 3] = (2.0 * far * near) / (near - far)
    projection_matrix[3, 2] = -1.0
    return projection_matrix


def look_at(eye, center, up):
    """Create a view matrix that simulates a camera looking at a specific point.

    Args:
        eye (numpy.ndarray): Position of the camera.
        center (numpy.ndarray): Point the camera is looking at.
        up (numpy.ndarray): Up vector defining the camera's orientation.

    Returns:
        numpy.ndarray: View matrix.
    """
    forward = center - eye
    forward /= np.linalg.norm(forward)

    side = np.cross(forward, up)
    side /= np.linalg.norm(side)

    new_up = np.cross(side, forward)

    view_matrix = np.eye(4)
    view_matrix[0, :3] = side
    view_matrix[1, :3] = new_up
    view_matrix[2, :3] = -forward
    view_matrix[:3, 3] = -np.dot(np.vstack((side, new_up, -forward)), eye)

    return view_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_prefix", help="Prefix for the cubemap images")
    args = parser.parse_args()

    images = [f"{args.image_prefix}_{i}.jpg" for i in range(6)]

    render_cubemap_image(
        (
            images[3],  # right
            images[1],  # left
            images[0],  # top
            images[5],  # bottom
            images[2],  # front
            images[4],  # back
        ),
    )


if __name__ == "__main__":
    main()
