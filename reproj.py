import argparse
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from PIL import Image

VERTEX_SHADER = """
#version 330 core

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
#version 330 core

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
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_CUBE_MAP, texture_id)

    for i, face in enumerate(faces):
        img = Image.open(face)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if i == 2:
            # Rotate the bottom image.
            img = img.rotate(-90)
        elif i == 3:
            # Rotate the top image.
            img = img.rotate(90)
        img = img.resize((256, 256))
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
    print("Finished loading cubemap images.")
    return texture_id


def compile_shader(shader_type, shader_source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, shader_source)
    glCompileShader(shader)

    # Check for compilation errors
    result = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if result != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))

    return shader


def create_shader_program(vertex_shader_source, fragment_shader_source):
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


def render_cubemap_image(cubemap_faces, camera_position):
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

    cubemap_texture = load_cubemap(cubemap_faces)

    def render():
        glClearColor(0, 0, 0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        projection_uniform = glGetUniformLocation(shader_program, "projection")
        projection_matrix = np.identity(4, dtype=np.float32)
        projection_matrix = perspective_projection(30, 1.0, 0.01, 10.0)
        glUniformMatrix4fv(
            projection_uniform, 1, GL_FALSE, projection_matrix.transpose()
        )

        view_uniform = glGetUniformLocation(shader_program, "view")
        view_matrix = np.identity(4, dtype=np.float32)
        # Translate the view matrix to the negative camera position
        view_matrix[3, 0:3] = -camera_position
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

        glutSwapBuffers()

    glutDisplayFunc(render)
    glutKeyboardFunc(keyboard_func)
    glutMainLoop()


def perspective_projection(fov, aspect_ratio, near, far):
    projection_matrix = np.zeros((4, 4), dtype=np.float32)
    f = 1.0 / np.tan(fov / 2.0)
    projection_matrix[0, 0] = f / aspect_ratio
    projection_matrix[1, 1] = f
    projection_matrix[2, 2] = (far + near) / (near - far)
    projection_matrix[2, 3] = (2.0 * far * near) / (near - far)
    projection_matrix[3, 2] = -1.0
    return projection_matrix


def orthographic_projection(left, right, bottom, top, near, far):
    projection_matrix = np.zeros((4, 4), dtype=np.float32)
    projection_matrix[0, 0] = 2.0 / (right - left)
    projection_matrix[1, 1] = 2.0 / (top - bottom)
    projection_matrix[2, 2] = -2.0 / (far - near)
    projection_matrix[3, 0] = -(right + left) / (right - left)
    projection_matrix[3, 1] = -(top + bottom) / (top - bottom)
    projection_matrix[3, 2] = -(far + near) / (far - near)
    projection_matrix[3, 3] = 1.0
    return projection_matrix


# Keyboard callback function
def keyboard_func(key, x, y):
    global camera_rotation_matrix

    # Rotate the camera based on keyboard input
    if key == b"a":
        rotation_angle = np.radians(5.0)
        rotation_matrix = np.array(
            [
                [np.cos(rotation_angle), 0, np.sin(rotation_angle)],
                [0, 1, 0],
                [-np.sin(rotation_angle), 0, np.cos(rotation_angle)],
            ]
        )
        camera_rotation_matrix = np.dot(rotation_matrix, camera_rotation_matrix)
    elif key == b"d":
        rotation_angle = np.radians(-5.0)
        rotation_matrix = np.array(
            [
                [np.cos(rotation_angle), 0, np.sin(rotation_angle)],
                [0, 1, 0],
                [-np.sin(rotation_angle), 0, np.cos(rotation_angle)],
            ]
        )
        camera_rotation_matrix = np.dot(rotation_matrix, camera_rotation_matrix)
    elif key == b"w":
        rotation_angle = np.radians(5.0)
        rotation_matrix = np.array(
            [
                [1, 0, 0],
                [0, np.cos(rotation_angle), -np.sin(rotation_angle)],
                [0, np.sin(rotation_angle), np.cos(rotation_angle)],
            ]
        )
        camera_rotation_matrix = np.dot(rotation_matrix, camera_rotation_matrix)
    elif key == b"s":
        rotation_angle = np.radians(-5.0)
        rotation_matrix = np.array(
            [
                [1, 0, 0],
                [0, np.cos(rotation_angle), -np.sin(rotation_angle)],
                [0, np.sin(rotation_angle), np.cos(rotation_angle)],
            ]
        )
        camera_rotation_matrix = np.dot(rotation_matrix, camera_rotation_matrix)
    # elif key == b"a":
    #     rotation_angle = np.radians(5.0)
    #     rotation_matrix = np.array(
    #         [
    #             [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
    #             [np.sin(rotation_angle), np.cos(rotation_angle), 0],
    #             [0, 0, 1],
    #         ]
    #     )
    #     camera_rotation_matrix = np.dot(rotation_matrix, camera_rotation_matrix)
    # elif key == b"d":
    #     rotation_angle = np.radians(-5.0)
    #     rotation_matrix = np.array(
    #         [
    #             [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
    #             [np.sin(rotation_angle), np.cos(rotation_angle), 0],
    #             [0, 0, 1],
    #         ]
    #     )
    #     camera_rotation_matrix = np.dot(rotation_matrix, camera_rotation_matrix)

    glutPostRedisplay()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_prefix", help="Prefix for the cubemap images")
    args = parser.parse_args()

    images = [f"{args.image_prefix}_{i}.jpg" for i in range(6)]

    camera_position = np.array([0, 0, 0], dtype=np.float32)
    # camera_rotation_matrix = np.identity(3, dtype=np.float32)

    render_cubemap_image(
        (
            images[3], # right
            images[1], # left
            images[5], # bottom
            images[0], # top
            images[2], # front
            images[4], # back
        ),
        camera_position,
    )


if __name__ == "__main__":
    main()
