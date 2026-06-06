#version 330 core

in vec2 in_position;
out vec2 v_uv;

uniform vec2  u_offset;   // NDC translation (0 = centre)
uniform float u_scale;    // layer scale

void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position * u_scale + u_offset, 0.0, 1.0);
}
