#define estimateNormal estimateNormalHex // use Tet for less samples

const vec3 shapeColor = vec3(0.9, 0.6, 0.3); // color of the shape
const vec3 ambientColor = vec3(0.1, 0.1, 0.1); // ambient light color
const vec3 light_dir = normalize(vec3(0.5, 1.0, 0.3)); // direction of the sun light source

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

float sdfScene(vec3 point) {
    float d1 = sdfSphere(point, 0.5);
    float d2 = sdfCube(point, vec3(2, 0.25, 2), q_euler(euler(vec3(PI * 0.1, uTime, 0), XYZ)));
    float k = 0.5; // smoothness factor, increase for softer blend

    float d = smoothmin(d1, d2, k);

    return d;
}

float softShadow(vec3 ro, vec3 rd, float maxDist, float k) {
    float res = 1.0;
    float t = 0.01;
    for (int i = 0; i < 64 && t < maxDist; i++) {
        float d = sdfScene(ro + rd * t);
        if (d < 0.001) return 0.0;
        res = min(res, k * d / t);
        t += d;
    }
    return clamp(res, 0.0, 1.0);
}

float ambientOcclusion(vec3 p, vec3 n) {
    float ao = 0.0;
    float sca = 1.0;
    for (int i = 1; i <= 5; i++) {
        float h = 0.02 * float(i);   // step size
        float d = sdfScene(p + n * h);
        ao += (h - d) * sca;
        sca *= 0.5;                  // reduce weight each step
    }
    return clamp(1.0 - 3.0 * ao, 0.0, 1.0);
}

vec3 estimateNormalHex(vec3 p) {
    float e = 0.001;
    return normalize(vec3(
        sdfScene(p + vec3(e,0,0)) - sdfScene(p - vec3(e,0,0)),
        sdfScene(p + vec3(0,e,0)) - sdfScene(p - vec3(0,e,0)),
        sdfScene(p + vec3(0,0,e)) - sdfScene(p - vec3(0,0,e))
    ));
}

vec3 estimateNormalTet(vec3 point) {
    float e = 0.001;
    vec2 h = vec2(1.0, -1.0) * 0.5773;
    return normalize(
        h.xyy * sdfScene(point + h.xyy * e) +
        h.yyx * sdfScene(point + h.yyx * e) +
        h.yxy * sdfScene(point + h.yxy * e) +
        h.xxx * sdfScene(point + h.xxx * e)
    );
}

struct Hit { bool hit; float t; };

Hit raymarch(vec3 ro, vec3 rd, int max_steps, float max_dist, float eps) {
    float t = 0.0;
    for (int i = 0; i < max_steps && t < max_dist; ++i) {
        vec3 p = ro + rd * t;
        float d = sdfScene(p);
        if (d < eps * t) return Hit(true, t);
        t += d;
    }
    return Hit(false, t);
}

void main() {
    vec2 uv = (gl_FragCoord.xy + vec2(0.5)) / uResolution; // [0,1]
    float aspect = uResolution.x / max(uResolution.y, 1.0);

    // Scene params
    float fov = radians(60.0);
    float tan_half = tan(0.5 * fov);

    // Map uv to NDC [-1,1]
    float u = uv.x * 2.0 - 1.0;
    float v = uv.y * 2.0 - 1.0;

    vec3 ro = vec3(0.0, 0.0, -3.0);
    vec3 rd = normalize(vec3(u * aspect * tan_half, -v * tan_half, 1.0));

    Hit h = raymarch(ro, rd, 128, 100.0, 0.001);

    vec3 col = vec3(0.0);
    if (h.hit) {
        vec3 p = ro + rd * h.t;
        vec3 n = estimateNormal(p);

        // diffuse
        float diff = max(0.0, dot(n, light_dir));

        // soft shadow
        float shadow = softShadow(p + n * 0.002, light_dir, 10.0, 8.0);

        // ambient occlusion
        float ao = ambientOcclusion(p, n);

        // combine
        col = ambientColor * ao + diff * shapeColor * shadow;
    } else {
        float t = 0.5 * (rd.y + 1.0);
        col = (1.0 - t) * vec3(1.0) + t * vec3(0.5, 0.7, 1.0);
    }

    finalColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}