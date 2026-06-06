#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_tex;
uniform float u_opacity;

void main() {
    // Outside the layer quad the rasterizer never invokes us; inside, sample the
    // layer FBO. Blend mode (additive / normal) is set via glBlendFunc by the host.
    vec3 c = texture(u_tex, v_uv).rgb;
    fragColor = vec4(c, u_opacity);
}
