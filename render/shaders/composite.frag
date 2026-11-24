#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_tex;
uniform float u_opacity;

void main() {
    // Outside the layer quad the rasterizer never invokes us; inside, sample the
    // layer FBO. Blend mode (additive / normal) is set via glBlendFunc by the host.
    // Premultiplied: rgb already over black, a = layer coverage (1 for most modes).
    vec4 t = texture(u_tex, v_uv);
    fragColor = vec4(t.rgb * u_opacity, t.a * u_opacity);
}
