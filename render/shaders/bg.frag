#version 330 core

in vec2 v_uv0;
out vec4 fragColor;
#define v_uv v_uv0

uniform sampler2D u_tex;
uniform int   u_has_image;
uniform float u_opacity;
uniform float u_img_aspect;     // image width / height
uniform float u_canvas_aspect;  // canvas width / height
uniform float u_zoom;           // 1.0 + reactive zoom

void main() {
    if (u_has_image == 0) {
        fragColor = vec4(0.0, 0.0, 0.0, u_opacity);
        return;
    }

    // Cover fit: keep aspect, fill canvas, crop overflow
    vec2 uv = v_uv - 0.5;
    if (u_canvas_aspect > u_img_aspect) {
        uv.y *= u_img_aspect / u_canvas_aspect;
    } else {
        uv.x *= u_canvas_aspect / u_img_aspect;
    }
    uv /= max(u_zoom, 0.0001);   // reactive zoom-in
    uv += 0.5;

    vec3 c = texture(u_tex, clamp(uv, 0.0, 1.0)).rgb;
    fragColor = vec4(c, u_opacity);
}
