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
uniform float u_shock_times[8];   // mode 10: spawn time (s) of each live shockwave
uniform float u_nuke_speed;       // mode 10: shockwave expansion speed
uniform float u_nuke_life;        // mode 10: shockwave lifetime (s)
uniform float u_nuke_width;       // mode 10: ring thickness multiplier
uniform float u_nuke_flash;       // mode 10: whiteout intensity on impact
uniform float u_nuke_bg;          // mode 10: nebula/starfield brightness
uniform float u_void_speed;       // mode 11: constant rotation speed
uniform float u_void_pull;        // mode 11: contraction strength on kick/bass
uniform float u_void_rays;        // mode 11: high-frequency ray intensity
uniform int   u_void_arms;        // mode 11: spiral arm count
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

// ── Value noise (smooth, organic) ─────────────────────────────────
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
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

    // ── Nuclear Shockwave — deep space + white shockwaves fired on each kick ──
    } else if (u_viz_type == 10) {
        vec2  uvc10  = (v_uv * 2.0 - 1.0) * vec2(u_aspect, 1.0);
        float r10    = length(uvc10);

        // Organic animated nebula: two scrolling octaves of value noise
        float neb    = vnoise(uvc10 * 3.0 + vec2(u_time * 0.05, u_time * 0.03)) * 0.65
                     + vnoise(uvc10 * 6.5 - vec2(u_time * 0.07, u_time * 0.04)) * 0.35;
        // Background breathes with the bass between kicks
        float breathe = 0.55 + u_bass * 1.2 + u_kick_accum * 0.5;
        color = mix(u_pal0, u_pal1, neb) * (0.04 + neb * 0.10) * breathe * u_nuke_bg;
        color += u_pal0 * 0.05 * smoothstep(1.3, 0.0, r10) * breathe * u_nuke_bg;

        // Static starfield with slow twinkle
        float sh10  = hash21(floor(v_uv * 320.0));
        float star  = smoothstep(0.994, 1.0, sh10);
        float tw    = 0.55 + 0.45 * sin(u_time * 2.5 + sh10 * 90.0);
        color += star * tw * vec3(0.85, 0.92, 1.0) * u_nuke_bg;

        // Live shockwaves — each ripple expands from center and fades with age
        float youngest = 1000.0;
        for (int i10 = 0; i10 < 8; i10++) {
            float age = u_time - u_shock_times[i10];
            if (age < 0.0 || age > u_nuke_life) continue;
            youngest     = min(youngest, age);
            float fade   = 1.0 - clamp(age / u_nuke_life, 0.0, 1.0);
            fade         = fade * fade;
            float radius = age * u_nuke_speed;                  // expansion speed
            float thick  = (0.012 + age * 0.025) * u_nuke_width; // softens as it travels
            float ring   = (1.0 - smoothstep(0.0, thick, abs(r10 - radius))) * fade;
            // trailing echo ring — the "ripples in water" feel
            float echo   = (1.0 - smoothstep(0.0, thick, abs(r10 - radius * 0.74)))
                           * fade * 0.45;
            vec3  wcol   = mix(u_pal2, vec3(1.0), 0.7);
            color += (ring * 1.6 + echo) * wcol;
        }

        // Whiteout flash on impact — brief, decays over the first ~0.12s
        float wo10 = clamp(1.0 - youngest / 0.12, 0.0, 1.0) * u_nuke_flash;
        color = mix(color, vec3(1.0), wo10);

    // ── Void Pull — black-hole spiral, space sucked inward, gravity on kick ──
    } else if (u_viz_type == 11) {
        vec2  uvc11  = (v_uv * 2.0 - 1.0) * vec2(u_aspect, 1.0);
        float r11    = length(uvc11);
        float ang11  = atan(uvc11.y, uvc11.x);

        // Gravity contraction: only the kick/bass distorts space (pulse = sub-bass
        // envelope with fast attack / slow release → snaps in then eases back).
        float pull   = clamp(u_pulse * u_void_pull, 0.0, 0.7);
        float r_w    = r11 * (1.0 - pull);

        // Constant, slow rotation — independent of audio
        float spd11  = u_void_speed;
        float arms   = float(u_void_arms);

        // Rotating spiral (slow, hypnotic)
        float spv    = fract(r_w * 5.0 + ang11 / (2.0 * PI) * arms - u_time * spd11);
        float spiral = smoothstep(0.09, 0.0, min(spv, 1.0 - spv));

        // Concentric rings scrolling inward (toward the void)
        float rnv    = fract(r_w * 7.0 + u_time * spd11 * 0.6);
        float rings  = smoothstep(0.06, 0.0, min(rnv, 1.0 - rnv));

        // Absolute void at the center + outer falloff
        float vmask  = smoothstep(0.0, 0.22, r11);
        float outer  = smoothstep(1.45, 0.15, r11);

        // Near-black background
        color = u_pal0 * 0.02;

        vec3 armCol = mix(u_pal1, u_pal2, clamp(r11, 0.0, 1.0));
        color += spiral * armCol * 0.55 * vmask * outer;
        color += rings * u_pal2 * 0.22 * vmask * outer;

        // High-frequency light rays beaming out from the center
        float rayPat = pow(0.5 + 0.5 * sin(ang11 * 14.0 - u_time * 0.6), 6.0);
        float rays   = rayPat * u_high * u_void_rays * smoothstep(1.2, 0.0, r11) * vmask;
        color += rays * mix(u_pal3, vec3(1.0), 0.5);

        // Center flares during the gravity contraction (kick feedback)
        color += pull * smoothstep(0.55, 0.0, r11) * u_pal2 * 0.5 * vmask;
    }

    if (u_flash_enabled == 1) {
        float f = clamp(u_pulse * u_pulse_intensity * u_flash_intensity, 0.0, 0.92);
        color = mix(color, vec3(1.0), f);
    }

    fragColor = vec4(color, 1.0);
}
