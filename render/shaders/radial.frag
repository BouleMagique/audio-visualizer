#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform int   u_num_bars;
uniform float u_bars[256];
uniform float u_pulse;
uniform float u_pulse_intensity;
uniform float u_max_bar_height;
uniform float u_circle_radius;
uniform float u_bar_width;
uniform float u_aspect;
uniform sampler2D u_bg_texture;
uniform int   u_has_bg;
uniform vec3  u_pal0;
uniform vec3  u_pal1;
uniform vec3  u_pal2;
uniform vec3  u_pal3;
uniform vec3  u_pal4;
uniform float u_sensitivity;
uniform int   u_viz_type;     // 0=radial  1=miroir  2=linéaire  3=oscillo  4=halo  5=halo-bass
uniform float u_time;         // seconds, for animation
uniform float u_lissa_a;      // Lissajous X frequency
uniform float u_lissa_b;      // Lissajous Y frequency
uniform float u_lissa_energy; // Lissajous amplitude [0..1]
uniform float u_rotation;     // radians — applied to radial/halo modes
uniform sampler2D u_center_texture;
uniform int   u_has_center;   // 1 = display center image inside halo circle
uniform sampler2D u_bass_history;  // float R32F, 256×64: col=freq, row=time (0=newest)
uniform sampler2D u_bass_history2; // float R32F, EMA-smoothed (decay 0.93) — mode 7
uniform float u_halo_r_base;       // mode 6: center image radius in aspect-corrected space
uniform int   u_pal_mode;          // 0 = amplitude, 1 = fréquence
uniform int   u_bg_pulse_enabled;  // 1 = zoom BG on bass hits
uniform float u_bg_pulse_intensity;
uniform int   u_flash_enabled;     // 1 = white flash on bass hits
uniform float u_flash_intensity;
uniform int   u_mirror;           // 1 = each half covers the full spectrum
uniform int   u_force_black_bg;   // 1 = render on pure black (for additive layer compositing)
uniform float u_bass;             // mean energy, first 30 % of bars
uniform float u_mid;              // mean energy, 30–70 %
uniform float u_high;             // mean energy, 70–100 %
uniform float u_kick;             // instantaneous kick spike [0..1]
uniform float u_kick_accum;       // accumulated kick with decay [0..1]
uniform int   u_tunnel_sides;     // polygon sides (4 / 6 / 8 / 12)
uniform int   u_tunnel_rings;     // ring-line count
uniform float u_tunnel_speed;     // base advance speed
uniform float u_tunnel_kick_zoom; // zoom intensity on kick
uniform float u_tunnel_chroma;    // chromatic aberration strength
uniform float u_tunnel_bass_speed; // bass reactivity on speed

const float PI = 3.14159265;

// ── HSV → RGB ────────────────────────────────────────────────────
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// ── Tunnel Arcade helpers ─────────────────────────────────────────
float polyInradius(vec2 p, float n) {
    float k  = 2.0 * PI / n;
    float a  = atan(p.y, p.x);
    float af = floor(a / k + 0.5) * k;
    return dot(p, vec2(cos(af), sin(af)));
}

vec3 tunnelColor(vec2 uv) {
    float n   = float(u_tunnel_sides);
    float k   = 2.0 * PI / n;
    float d   = polyInradius(uv, n);
    if (d < 0.002) return vec3(0.0);

    float z     = 1.0 / d;
    float speed = u_tunnel_speed + u_bass * u_tunnel_bass_speed;

    float rt   = fract(z * float(u_tunnel_rings) - u_time * speed);
    float ring = smoothstep(0.06, 0.0, min(rt, 1.0 - rt));

    float a     = atan(uv.y, uv.x);
    float af2   = mod(a + k * 0.5, k);
    float ef    = af2 / k;
    float ew    = 0.022 + u_mid * 0.04;
    float edge  = smoothstep(ew, 0.0, min(ef, 1.0 - ef));

    float intersect = ring * edge * 2.5;

    float hue  = fract(u_time * 0.04 + u_bass * 0.25);
    vec3 bcol  = hsv2rgb(vec3(hue, 0.75 + u_high * 0.25, 1.0));

    float fog    = clamp(z * 0.35, 0.0, 1.0);
    float bright = (ring * 0.7 + edge * 0.8 + intersect) * fog;
    return bcol * bright;
}

// ── palette ──────────────────────────────────────────────────────
vec3 pal(float t) {
    t = clamp(t, 0.0, 1.0);
    if (t < 0.25) return mix(u_pal0, u_pal1, t * 4.0);
    if (t < 0.50) return mix(u_pal1, u_pal2, (t - 0.25) * 4.0);
    if (t < 0.75) return mix(u_pal2, u_pal3, (t - 0.50) * 4.0);
    return mix(u_pal3, u_pal4, (t - 0.75) * 4.0);
}

// Sélectionne la palette par amplitude ou par fréquence selon u_pal_mode
vec3 col(float amp_t, float freq_t) {
    return pal(u_pal_mode == 1 ? freq_t : amp_t);
}

// ── Catmull-Rom (scalar) ─────────────────────────────────────────
float catmull_rom(float p0, float p1, float p2, float p3, float t) {
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * t
        + (2.0*p0 - 5.0*p1 + 4.0*p2 - p3) * t * t
        + (-p0 + 3.0*p1 - 3.0*p2 + p3)    * t * t * t
    );
}

// ── helpers for flat viz types ───────────────────────────────────
bool in_bar_slot(float uv_x) {
    float frac = fract(uv_x * float(u_num_bars));
    float half_gap = (1.0 - u_bar_width) * 0.5;
    return frac > half_gap && frac < (1.0 - half_gap);
}

int bar_index_x(float uv_x) {
    return clamp(int(uv_x * float(u_num_bars)), 0, u_num_bars - 1);
}

// ────────────────────────────────────────────────────────────────
void main() {
    vec2 bg_uv = v_uv;
    if (u_bg_pulse_enabled == 1 && u_has_bg == 1) {
        float zoom = 1.0 + u_pulse * u_pulse_intensity * u_bg_pulse_intensity * 0.10;
        bg_uv = (v_uv - 0.5) / zoom + 0.5;
    }
    vec3 bg = (u_has_bg == 1) ? texture(u_bg_texture, bg_uv).rgb
                               : vec3(0.039, 0.039, 0.059);
    if (u_force_black_bg == 1) bg = vec3(0.0);
    vec3 color = bg;

    // ── Radial ──────────────────────────────────────────────────
    if (u_viz_type == 0) {
        vec2 uv = (v_uv * 2.0 - 1.0) * vec2(u_aspect, 1.0);
        float dist  = length(uv);
        float angle = atan(uv.y, uv.x) - u_rotation;
        float t_raw = fract(angle / (2.0 * PI) + 0.5);
        float t_ang = (u_mirror == 1) ? (1.0 - abs(t_raw * 2.0 - 1.0)) : t_raw;

        float pulse_r = u_circle_radius * (1.0 + 0.12 * u_pulse * u_pulse_intensity);

        float glow = smoothstep(pulse_r * 1.6, 0.0, dist) * 0.45;
        color = bg + glow * u_pal0;

        int   bar_idx = int(t_ang * float(u_num_bars)) % u_num_bars;
        float bar_val = clamp(u_bars[bar_idx] * u_sensitivity, 0.0, 1.0);
        float bar_len = bar_val * 0.55 * u_max_bar_height;
        float freq_t0 = t_ang;

        float seg_w  = (u_mirror == 1) ? (PI / float(u_num_bars)) : (2.0 * PI / float(u_num_bars));
        float half_w = seg_w * u_bar_width * 0.5;
        float bar_ang;
        if (u_mirror == 1) {
            float t_bar_raw = (t_raw < 0.5)
                ? (float(bar_idx) + 0.5) / float(u_num_bars) * 0.5
                : 1.0 - (float(bar_idx) + 0.5) / float(u_num_bars) * 0.5;
            bar_ang = (t_bar_raw - 0.5) * 2.0 * PI;
        } else {
            bar_ang = (float(bar_idx) + 0.5) / float(u_num_bars) * 2.0 * PI - PI;
        }
        float ang_diff = abs(mod(angle - bar_ang + PI, 2.0 * PI) - PI);

        if (ang_diff < half_w && dist >= pulse_r) {
            float cap_r  = half_w * (pulse_r + bar_len);
            vec2  tip    = vec2(cos(bar_ang), sin(bar_ang)) * (pulse_r + bar_len);
            bool  in_rad = dist <= pulse_r + bar_len;
            bool  in_cap = length(uv - tip) < cap_r;
            if (in_rad || in_cap) {
                float rel = (dist - pulse_r) / max(bar_len, 0.001);
                color = col(bar_val, freq_t0) * mix(1.0, 0.7, rel);
            }
        }

        if (dist < pulse_r) {
            color = mix(bg * 0.6, color, 0.3);
        }

        float sw = pulse_r * 0.04;
        float sd = dist - (pulse_r - sw);
        if (sd >= 0.0 && sd <= sw * 2.0) {
            float st = sd / (sw * 2.0);
            float sa = smoothstep(0.0, 0.4, 1.0 - abs(st * 2.0 - 1.0));
            color = mix(color, col(bar_val, freq_t0), sa * 0.95);
        }

    // ── Miroir ──────────────────────────────────────────────────
    } else if (u_viz_type == 1) {
        int   bar_idx = bar_index_x(v_uv.x);
        float bar_val = clamp(u_bars[bar_idx] * u_sensitivity, 0.0, 1.0);
        float bar_len = bar_val * u_max_bar_height;
        float y_dist  = abs(v_uv.y - 0.5) * 2.0;
        float freq_t1 = float(bar_idx) / float(max(u_num_bars - 1, 1));

        if (in_bar_slot(v_uv.x) && y_dist < bar_len) {
            float rel = y_dist / max(bar_len, 0.001);
            color = col(bar_val, freq_t1) * mix(1.0, 0.65, rel);
        }

        if (abs(v_uv.y - 0.5) < 0.002)
            color = mix(color, u_pal2, 0.6);

    // ── Linéaire ────────────────────────────────────────────────
    } else if (u_viz_type == 2) {
        int   bar_idx = bar_index_x(v_uv.x);
        float bar_val = clamp(u_bars[bar_idx] * u_sensitivity, 0.0, 1.0);
        float bar_len = bar_val * u_max_bar_height;
        float freq_t2 = float(bar_idx) / float(max(u_num_bars - 1, 1));

        if (in_bar_slot(v_uv.x) && v_uv.y < bar_len) {
            float rel = v_uv.y / max(bar_len, 0.001);
            color = col(bar_val, freq_t2) * mix(1.0, 0.55, rel);
        }

    // ── Oscilloscope / Lissajous ─────────────────────────────────
    } else if (u_viz_type == 3) {
        vec2 uv_c = (v_uv * 2.0 - 1.0) * vec2(u_aspect, 1.0);

        float min_dist = 10.0;
        float phase    = u_time * 0.5;

        // 128-sample SDF — each sample perturbed by its frequency bar
        for (int k = 0; k < 128; k++) {
            float t   = float(k) / 128.0 * 2.0 * PI;
            int bidx  = int(float(k) / 128.0 * float(u_num_bars)) % u_num_bars;
            float mod = 1.0 + clamp(u_bars[bidx] * u_sensitivity, 0.0, 1.0) * 0.35;

            float cx = u_lissa_energy * mod * sin(u_lissa_a * t + phase);
            float cy = u_lissa_energy          * sin(u_lissa_b * t);
            min_dist = min(min_dist, length(uv_c - vec2(cx, cy)));
        }

        float line_w = 0.008;
        float core   = exp(-(min_dist / line_w) * (min_dist / line_w));
        float glow   = exp(-min_dist / (line_w * 6.0)) * 0.4;

        // Tint the core with palette driven by energy, whiten the brightest part
        vec3 lissa_col = pal(clamp(u_lissa_energy * 1.2, 0.0, 1.0));
        color = bg + (core + glow) * mix(lissa_col, vec3(1.0), core * 0.85);

        // Faint background radial glow centered on the figure
        float center_d = length(uv_c);
        color += smoothstep(u_lissa_energy * 1.8, 0.0, center_d) * 0.07 * u_pal0;

    // ── Halo Waveform ────────────────────────────────────────────
    } else if (u_viz_type == 4) {
        vec2  uv_c = (v_uv * 2.0 - 1.0) * vec2(u_aspect, 1.0);
        float dist  = length(uv_c);
        float angle     = atan(uv_c.y, uv_c.x) - u_rotation;
        float theta_raw = fract(angle / (2.0 * PI) + 0.5);
        float theta     = (u_mirror == 1) ? (1.0 - abs(theta_raw * 2.0 - 1.0)) : theta_raw;

        // Catmull-Rom interpolated bar value
        float bar_f = theta * float(u_num_bars);
        int   i1    = int(bar_f) % u_num_bars;
        int   i0    = (i1 - 1 + u_num_bars) % u_num_bars;
        int   i2    = (i1 + 1) % u_num_bars;
        int   i3    = (i1 + 2) % u_num_bars;
        float t_cr  = fract(bar_f);

        float p0 = clamp(u_bars[i0] * u_sensitivity, 0.0, 1.0);
        float p1 = clamp(u_bars[i1] * u_sensitivity, 0.0, 1.0);
        float p2 = clamp(u_bars[i2] * u_sensitivity, 0.0, 1.0);
        float p3 = clamp(u_bars[i3] * u_sensitivity, 0.0, 1.0);
        float r_fft = clamp(catmull_rom(p0, p1, p2, p3, t_cr), 0.0, 1.0);

        // Pulsing base radius + FFT displacement
        float base_r  = u_circle_radius * (1.0 + 0.08 * u_pulse * u_pulse_intensity);
        float r_curve = base_r + r_fft * 0.45 * u_max_bar_height;

        float d      = abs(dist - r_curve);
        float line_w = 0.006;

        // Layered glow: tight bright core + wide soft halo
        float core = exp(-(d / line_w) * (d / line_w));
        float halo = exp(-d / (line_w * 5.0)) * 0.32;

        // Soft semi-transparent fill inside the bubble
        float fill   = smoothstep(r_curve, r_curve - 0.08, dist) * 0.07;
        // Darken the interior
        float in_bub = step(dist, r_curve);

        color = mix(bg, bg * 0.22, in_bub * 0.65);
        color += fill * u_pal0;

        // Faint central glow (echoes radial mode)
        color += smoothstep(base_r * 1.4, 0.0, dist) * 0.18 * u_pal0;

        // Line: white core, palette-tinted outer glow
        vec3 halo_col = mix(col(r_fft, theta), vec3(1.0), core * 0.9);
        color += (core + halo) * halo_col;

        // Center image (inside the base circle, smooth fade near edge)
        if (u_has_center == 1) {
            vec2  uv_ctr = uv_c / base_r * 0.5 + 0.5;
            float mask   = smoothstep(base_r, base_r * 0.72, dist);
            color = mix(color, texture(u_center_texture, clamp(uv_ctr, 0.0, 1.0)).rgb, mask);
        }

    // ── Halo Bass — waterfall polaire (N courbes Catmull-Rom) ───────
    } else if (u_viz_type == 5) {
        vec2  uv_c = (v_uv * 2.0 - 1.0) * vec2(u_aspect, 1.0);
        float dist  = length(uv_c);
        float angle = atan(uv_c.y, uv_c.x) - u_rotation;
        float angle_w = mod(angle + PI, 2.0 * PI) - PI;
        float theta_m = abs(angle_w) / PI;   // [0,1] axe fréq, miroir haut/bas

        float base_r = u_circle_radius * (1.0 + 0.10 * u_pulse * u_pulse_intensity);
        float span   = 0.55 * u_max_bar_height;
        float max_r  = base_r + span;

        // Fond sombre à l'intérieur de la couronne
        color = mix(bg, bg * 0.10, smoothstep(max_r, max_r - 0.02, dist)
                                  * smoothstep(base_r * 0.6, base_r, dist));

        // Nombre de barres basses utilisées par la spline
        int bass_n = max(u_num_bars / 3, 12);
        // UV texture : x = fréq, y = temps (0=récent, 1=ancien)
        float freq_uv  = theta_m * float(bass_n) / 256.0;
        float step_uv  = 1.0 / 256.0;   // un bin de distance dans la texture

        // Accumulation des N_RINGS courbes Catmull-Rom
        const int N_RINGS = 12;
        for (int k = 0; k < N_RINGS; k++) {
            float age_t   = float(k) / float(N_RINGS - 1);  // 0=récent, 1=ancien
            float age_w   = pow(1.0 - age_t, 0.55);          // facteur d'éclat

            // Rayon de base de cette courbe (espace régulier dans la couronne)
            float r_base  = base_r + age_t * span * 0.88;

            // Catmull-Rom en fréq depuis la texture historique (4 échantillons)
            float p0 = texture(u_bass_history, vec2(max(freq_uv - step_uv,        0.0), age_t)).r;
            float p1 = texture(u_bass_history, vec2(freq_uv,                            age_t)).r;
            float p2 = texture(u_bass_history, vec2(min(freq_uv + step_uv,        1.0), age_t)).r;
            float p3 = texture(u_bass_history, vec2(min(freq_uv + 2.0 * step_uv,  1.0), age_t)).r;
            float t_cr = fract(freq_uv * 256.0);
            float val  = clamp(catmull_rom(p0, p1, p2, p3, t_cr) * u_sensitivity, 0.0, 1.0);

            // Déplacement radial de la courbe (amplitude = 10 % de la couronne)
            float r_curve = r_base + val * span * 0.10;

            float d      = abs(dist - r_curve);
            float line_w = 0.0045;
            float core   = exp(-(d / line_w) * (d / line_w)) * age_w;
            float glow   = exp(-d / (line_w * 6.0)) * 0.28 * age_w;

            vec3 c = mix(col(val, theta_m), vec3(1.0), core * 0.85);
            color += (core + glow) * c;
        }

        // Anneau "maintenant" brillant à base_r
        float now_w = base_r * 0.012;
        color += u_pal1 * smoothstep(now_w * 2.0, 0.0, abs(dist - base_r)) * 0.55;

        // Lueur centrale douce
        color += smoothstep(base_r * 1.5, 0.0, dist) * 0.20 * u_pal0;

        // Image centre
        if (u_has_center == 1) {
            vec2  uv_ctr = uv_c / base_r * 0.5 + 0.5;
            float mask   = smoothstep(base_r, base_r * 0.72, dist);
            color = mix(color, texture(u_center_texture, clamp(uv_ctr, 0.0, 1.0)).rgb, mask);
        }

    // ── Halo Bass 2 — soft neon waterfall, EMA-smoothed data ───────────────
    } else if (u_viz_type == 7) {
        vec2  uv_c  = (v_uv * 2.0 - 1.0) * vec2(u_aspect, 1.0);
        float dist  = length(uv_c);
        float angle = atan(uv_c.y, uv_c.x) - u_rotation;
        float theta_m = abs(mod(angle + PI, 2.0 * PI) - PI) / PI;

        float base_r = u_circle_radius * (1.0 + 0.10 * u_pulse * u_pulse_intensity);
        float span   = 0.55 * u_max_bar_height;
        float max_r  = base_r + span;

        color = mix(bg, bg * 0.04,
                    smoothstep(max_r, max_r - 0.02, dist) *
                    smoothstep(base_r * 0.6, base_r, dist));

        int   bass_n  = max(u_num_bars / 3, 12);
        float freq_uv = theta_m * float(bass_n) / 256.0;
        float step_uv = 1.0 / 256.0;

        // 24 anneaux — texture LINEAR interpole les lignes intermediaires
        const int N2 = 24;
        for (int k = 0; k < N2; k++) {
            float age_t  = float(k) / float(N2 - 1);
            float age_w  = pow(1.0 - age_t, 0.30);
            float r_base_k = base_r + age_t * span * 0.88;

            // Catmull-Rom depuis les donnees EMA (texture unit 3)
            float p0 = texture(u_bass_history2, vec2(max(freq_uv - step_uv,      0.0), age_t)).r;
            float p1 = texture(u_bass_history2, vec2(freq_uv,                          age_t)).r;
            float p2 = texture(u_bass_history2, vec2(min(freq_uv + step_uv,      1.0), age_t)).r;
            float p3 = texture(u_bass_history2, vec2(min(freq_uv + 2.0*step_uv,  1.0), age_t)).r;
            float t_cr = fract(freq_uv * 256.0);
            float val  = clamp(catmull_rom(p0, p1, p2, p3, t_cr) * u_sensitivity, 0.0, 1.0);

            float r_curve = r_base_k + val * span * 0.10;
            float d       = abs(dist - r_curve);

            // Noyau Gaussien large (sigma 5x > mode 5) — look tube neon diffus
            // chevauchement ~40% entre anneaux adjacents
            float sigma = 0.022;
            float gauss = exp(-(d * d) / (sigma * sigma)) * age_w;
            float halo  = exp(-d / (sigma * 4.0)) * 0.50 * age_w;

            // Accumulation additive (equivalent GL_ONE, GL_ONE au niveau fragment)
            vec3 c = mix(col(val, theta_m), vec3(1.0), gauss * 0.70);
            color += (gauss + halo) * c * 0.80;
        }

        float now_w = base_r * 0.014;
        color += u_pal1 * smoothstep(now_w * 2.5, 0.0, abs(dist - base_r)) * 0.75;
        color += smoothstep(base_r * 1.6, 0.0, dist) * 0.22 * u_pal0;

        if (u_has_center == 1) {
            vec2  uv_ctr = uv_c / base_r * 0.5 + 0.5;
            float mask   = smoothstep(base_r, base_r * 0.72, dist);
            color = mix(color, texture(u_center_texture, clamp(uv_ctr, 0.0, 1.0)).rgb, mask);
        }

    // ── Tunnel Arcade — wireframe polygon tunnel, fragment-shader only ──
    } else if (u_viz_type == 8) {
        vec2 uv = (v_uv * 2.0 - 1.0) * vec2(u_aspect, 1.0);

        float kick  = u_kick_accum * u_tunnel_kick_zoom;
        uv = uv / (1.0 + kick * 4.5);
        uv.x += kick * 0.035 * sin(u_time * 29.3);

        float chroma = kick * u_tunnel_chroma * 0.014;
        vec3 tr = tunnelColor(uv + vec2(chroma, 0.0));
        vec3 tg = tunnelColor(uv);
        vec3 tb = tunnelColor(uv - vec2(chroma, 0.0));
        color = vec3(tr.r, tg.g, tb.b) + bg * 0.12;

        color = mix(color, vec3(0.65, 0.2, 1.0), clamp(kick * 0.55, 0.0, 0.45));
        color += vec3(1.0) * clamp(u_kick * 0.30, 0.0, 0.28);

        color *= 0.92 + 0.08 * sin(v_uv.y * 900.0 * PI);

        // Center image clipped to the tunnel polygon shape
        if (u_has_center == 1) {
            float cr   = u_halo_r_base * (1.0 + 0.08 * u_pulse * u_pulse_intensity);
            float pd   = polyInradius(uv, float(u_tunnel_sides));
            vec2  uctr = uv / cr * 0.5 + 0.5;
            float mask = smoothstep(cr, cr * 0.82, pd);
            if (uctr.x >= 0.0 && uctr.x <= 1.0 && uctr.y >= 0.0 && uctr.y <= 1.0) {
                vec4 s = texture(u_center_texture, uctr);
                color  = mix(color, s.rgb, s.a * mask);
            }
        }

    // ── Sine Plat — background only; waveform drawn by PIL ──
    } else if (u_viz_type == 9) {

    // ── Halo Sine — background only; spline, ring, and center image drawn by PIL ──
    } else if (u_viz_type == 6) {
        // Ring, glow, and center image are composited in Python so they appear
        // above the spline overlay. Nothing to add here beyond the background.

    // ── Nuclear Shockwave — cosmic noise + 4 concentric shockwaves on kick ──
    } else if (u_viz_type == 10) {
        vec2  uvc10  = (v_uv * 2.0 - 1.0) * vec2(u_aspect, 1.0);
        float r10    = length(uvc10);
        float ang10  = atan(uvc10.y, uvc10.x);

        // Two animated noise layers for organic cosmic background
        vec2  nf1    = floor(v_uv * 14.0 + u_time * vec2(0.41, 0.29));
        vec2  nf2    = floor(v_uv * 29.0 + u_time * vec2(0.17, 0.53));
        float n1     = fract(sin(dot(nf1, vec2(127.1, 311.7))) * 43758.5);
        float n2     = fract(sin(dot(nf2, vec2(269.5, 183.3))) * 43758.5);
        float nbg    = (n1 + n2) * 0.5;

        // Static star particles
        float stars  = step(0.998, fract(sin(dot(floor(v_uv * 220.0),
                            vec2(127.1, 311.7))) * 43758.5));

        // Cosmic background: noise driven by bass + faint central palette glow
        color = nbg * u_bass * u_pal0 * 0.40
              + stars * u_pal4 * 0.95
              + u_pal0 * 0.025 * (1.0 - clamp(r10 * 0.5, 0.0, 1.0));

        // Ambient background rings with sub-bass radial distortion
        float sub_r  = r10 + u_bass * 0.05 * sin(ang10 * 6.0 + u_time * 1.2);
        float rphase = mod(sub_r * 4.5 + u_time * 0.10, 1.0);
        float aring  = smoothstep(0.04, 0.0, min(rphase, 1.0 - rphase));
        color += aring * u_pal0 * 0.06 * (1.0 + u_bass * 2.5);

        // 4 shockwaves with phase offset — expand and fade driven by kick_accum
        float kpow10 = u_kick_accum;
        for (int i10 = 0; i10 < 4; i10++) {
            float fi10       = float(i10);
            float waveRadius = mod(u_time * kpow10 * 0.8 - fi10 * 0.18, 1.2);
            float waveFade   = 1.0 - clamp(waveRadius / 1.2, 0.0, 1.0);
            float wave = (1.0 - smoothstep(0.0, 0.04, abs(r10 - waveRadius)))
                         * waveFade * kpow10;
            color += wave * mix(u_pal3, u_pal1, waveFade);
        }

        // Kick traîne: radial glow emanating from center (kick_accum decay)
        color += kpow10 * smoothstep(0.55, 0.0, r10) * 0.28 * u_pal2;

        // Whiteout: pow(kick * kpow, 3) * 0.5
        float wo10 = clamp(pow(u_kick * kpow10, 3.0) * 0.5, 0.0, 1.0);
        color = mix(color, vec3(1.0), wo10);

    // ── Void Pull — abyssal spiral pulled inward on kick ──────────────────
    } else if (u_viz_type == 11) {
        vec2  uvc11  = (v_uv * 2.0 - 1.0) * vec2(u_aspect, 1.0);
        float r11    = length(uvc11);
        float ang11  = atan(uvc11.y, uvc11.x);

        // Gravity pull distortion: inward warp on kick_accum
        float pull   = u_kick_accum * 0.65;
        float r_d    = r11 * (1.0 - pull * exp(-r11 * 2.0));

        // Bass-driven rotation speed + palette hue
        float spd11  = 0.40 + u_bass * 1.5;
        float hue11  = fract(u_time * 0.06 + u_bass * 0.35);

        // 3-arm spiral rotating at spd11
        float spv    = mod(r_d * 6.0 - ang11 / (2.0 * PI) * 3.0
                           - u_time * spd11, 1.0);
        float spiral = smoothstep(0.05, 0.0, min(spv, 1.0 - spv));

        // Concentric rings scrolling inward
        float rnv    = mod(r_d * 4.0 - u_time * spd11 * 0.5, 1.0);
        float rings  = smoothstep(0.04, 0.0, min(rnv, 1.0 - rnv));

        // Void mask: smooth black hole at center
        float vmask  = smoothstep(0.0, 0.15, r11);

        // Very dark background — near-black void
        color = u_pal0 * 0.03 * (1.0 - clamp(r11 * 0.5, 0.0, 1.0));

        // Spiral + rings
        vec3 spCol = hsv2rgb(vec3(hue11, 0.85, 1.0));
        color += spiral * spCol * 0.75 * vmask;
        color += rings * mix(u_pal0, u_pal2, clamp(r11, 0.0, 1.0)) * 0.35 * vmask;

        // Radial light rays on high frequencies + kick: sin(angle*12 + time*10) * kick
        float rays = sin(ang11 * 12.0 + u_time * 10.0)
                     * u_kick * (1.0 - clamp(r11 * 0.7, 0.0, 1.0));
        color += clamp(rays * u_high * 3.0, 0.0, 1.0) * u_pal4 * vmask;

        // Whiteout radial: pow(kick, 3) * 0.4 * (1 - r * 0.5)
        float wo11 = clamp(pow(u_kick, 3.0) * 0.4 * (1.0 - r11 * 0.5) * vmask,
                           0.0, 1.0);
        color = mix(color, vec3(1.0), wo11);
    }

    if (u_flash_enabled == 1) {
        float f = clamp(u_pulse * u_pulse_intensity * u_flash_intensity, 0.0, 0.92);
        color = mix(color, vec3(1.0), f);
    }

    fragColor = vec4(color, 1.0);
}
