#version 330 core

in vec2 in_position;
out vec2 v_uv0;

void main() {
    v_uv0 = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
