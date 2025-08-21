#define estimateNormal estimateNormalHex // use Tet for less samples

const vec3 shapeColor1 = vec3(0.9, 0.2, 0.3);
const vec3 shapeColor2 = vec3(0.4, 0.7, 0.9);
const vec3 shapeColor3 = vec3(0.8, 0.4, 0.6);
const vec3 light_dir = normalize(vec3(0.3, 0.6, -0.3)); // direction of the sun light source


float sdfSphere(vec3 point, float radius) {
    return length(point) - radius;
}

float sdfAABB(vec3 point, vec3 size) {
    vec3 d = abs(point) - size;
    return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0);
}

float sdfCube(vec3 point, vec3 size, quat rotation) {
    return sdfAABB(q_rotate(rotation, point), size * 0.5);
}

struct Material {
    vec3 color;
};

Material material_mix(Material a, Material b, float t) {
    return Material(
        mix(a.color, b.color, t)
    );
}

float sdfScene(vec3 point, out Material m) {
    float d1 = sdfSphere(point, 0.5);
    float d2 = sdfCube(point, vec3(2, 0.25, 2), q_euler(euler(vec3(PI * 0.1, uTime, 0), XYZ)));
    
    float blendFactor;
    float dx = smoothmin_ex(d1, d2, 0.5, blendFactor);

    float d3 = sdfSphere(point - vec3(1, 1, -1), 0.1);

    if (d3 < dx) {
        m = Material(shapeColor3);
    } else {
        m = material_mix(
            Material(shapeColor1),
            Material(shapeColor2),
            blendFactor
        );
    }

    return min(dx, d3);
}

Material getMaterial(vec3 point) {
    Material m;
    sdfScene(point, m);
    return m;
}

float softShadow(vec3 ro, vec3 rd, float maxDist, float k) {
    Material m;
    float res = 1.0;
    float t = 0.01;
    for (int i = 0; i < 64 && t < maxDist; i++) {
        float d = sdfScene(ro + rd * t, m);
        if (d < 0.001) return 0.0;
        res = min(res, k * d / t);
        t += d;
    }
    return clamp(res, 0.0, 1.0);
}

vec3 estimateNormalHex(vec3 p) {
    Material m;
    float e = 0.001;
    return normalize(vec3(
        sdfScene(p + vec3(e,0,0), m) - sdfScene(p - vec3(e,0,0), m),
        sdfScene(p + vec3(0,e,0), m) - sdfScene(p - vec3(0,e,0), m),
        sdfScene(p + vec3(0,0,e), m) - sdfScene(p - vec3(0,0,e), m)
    ));
}

vec3 estimateNormalTet(vec3 point) {
    Material m;
    float e = 0.001;
    vec2 h = vec2(1.0, -1.0) * 0.5773;
    return normalize(
        h.xyy * sdfScene(point + h.xyy * e, m) +
        h.yyx * sdfScene(point + h.yyx * e, m) +
        h.yxy * sdfScene(point + h.yxy * e, m) +
        h.xxx * sdfScene(point + h.xxx * e, m)
    );
}


struct Hit { bool hit; float t; Material material; };

Hit raymarch(vec3 ro, vec3 rd, int max_steps, float max_dist, float eps) {
    Material m;
    float t = 0.0;
    for (int i = 0; i < max_steps && t < max_dist; ++i) {
        vec3 p = ro + rd * t;
        float d = sdfScene(p, m);
        if (d < max(eps, eps * t)) return Hit(true, t, m);
        t += d;
    }
    return Hit(false, t, Material(vec3(0.0)));
}

void main() {
    vec2 uv = (gl_FragCoord.xy + vec2(0.5)) / uResolution;
    float aspect = uResolution.x / max(uResolution.y, 1.0);

    float fov = radians(60.0);
    float tan_half = tan(0.5 * fov);

    float u = uv.x * 2.0 - 1.0;
    float v = uv.y * 2.0 - 1.0;

    vec3 ro = vec3(0.0, 0.0, -3.0);
    vec3 rd = normalize(vec3(u * aspect * tan_half, -v * tan_half, 1.0));

    Hit h = raymarch(ro, rd, 128, 100.0, 0.001);

    vec3 col = vec3(0.0);
    if (h.hit) {
        vec3 p = ro + rd * h.t;
        vec3 n = estimateNormal(p);

        float diff = max(0.0, dot(n, light_dir));
        float shadow = softShadow(p + n * 0.002, light_dir, 10.0, 8.0);

        vec3 lighting = diff * h.material.color * shadow;

        col = lighting;
    } else {
        float t = 0.5 * (rd.y + 1.0);
        col = (1.0 - t) * vec3(1.0) + t * vec3(0.5, 0.7, 1.0);
    }

    finalColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}