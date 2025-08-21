#version 330 core

in vec2 fragTexCoord;
out vec4 finalColor;

uniform vec2 uResolution; // viewport size in pixels
uniform float uTime;       // seconds

// SDF helpers
float sdfSphere(vec3 p, float r) {
    return length(p) - r;
}

float sdfScene(vec3 p) {
    // Single sphere at origin
    return sdfSphere(p, 1.0);
}

vec3 estimateNormal(vec3 p) {
    float e = 0.001;
    vec2 h = vec2(1.0, -1.0) * 0.5773; // tetrahedral offsets to reduce samples
    return normalize(
        h.xyy * sdfScene(p + h.xyy * e) +
        h.yyx * sdfScene(p + h.yyx * e) +
        h.yxy * sdfScene(p + h.yxy * e) +
        h.xxx * sdfScene(p + h.xxx * e)
    );
}

struct Hit { bool hit; float t; };

Hit raymarch(vec3 ro, vec3 rd, int max_steps, float max_dist, float eps) {
    float t = 0.0;
    for (int i = 0; i < max_steps && t < max_dist; ++i) {
        vec3 p = ro + rd * t;
        float d = sdfScene(p);
        if (d < eps) return Hit(true, t);
        t += d;
    }
    return Hit(false, t);
}

void main() {
    vec2 uv = (gl_FragCoord.xy + vec2(0.5)) / uResolution; // [0,1]
    float aspect = uResolution.x / max(uResolution.y, 1.0);

    // Camera params
    float fov = radians(60.0);
    float tan_half = tan(0.5 * fov);

    // Map uv to NDC [-1,1]
    float u = uv.x * 2.0 - 1.0;
    float v = uv.y * 2.0 - 1.0;

    vec3 ro = vec3(0.0, 0.0, -3.0);
    vec3 rd = normalize(vec3(u * aspect * tan_half, -v * tan_half, 1.0));

    Hit h = raymarch(ro, rd, 128, 100.0, 0.001);

    vec3 col = vec3(0.7, 0.9, 1.0);
    if (h.hit) {
        vec3 p = ro + rd * h.t;
        vec3 n = estimateNormal(p);
        vec3 light_dir = normalize(vec3(0.5, 1.0, 0.3));
        float diff = max(0.0, dot(n, light_dir));
        col = vec3(0.2) * vec3(1.0) + diff * vec3(0.9, 0.6, 0.3);
    } else {
        float t = 0.5 * (rd.y + 1.0);
        col = (1.0 - t) * vec3(1.0) + t * vec3(0.5, 0.7, 1.0);
    }

    finalColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
