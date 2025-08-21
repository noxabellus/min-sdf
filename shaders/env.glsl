#version 330 core

// ==== Shader IO ====

in vec2 fragTexCoord;
out vec4 finalColor;

uniform vec2 uResolution; // viewport size in pixels
uniform float uTime;       // seconds


// ==== Constants ====

#define PI 3.14159265358979323846
#define pos_infinity uintBitsToFloat(uint(0x7F800000))
#define neg_infinity uintBitsToFloat(uint(0xFF800000))


// ==== Euler angles ====

const int XYZ = 0;
const int XZY = 1;
const int YXZ = 2;
const int YZX = 3;
const int ZXY = 4;
const int ZYX = 5;

struct euler {
    /// Euler angles in radians (pitch, yaw, roll)
    vec3 angles; 
    /// Order of rotations: 0=XYZ, 1=XZY, 2=YXZ, 3=YZX, 4=ZXY, 5=ZYX
    int order;
};


// ==== Quaternions ====

#define quat vec4

/// Quaternion represented as vec4: (x, y, z, w)
/// w = scalar part, (x, y, z) = vector part

/// Quaternion multiplication: q1 * q2
quat q_mul(quat q1, quat q2) {
    return quat(
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    );
}

/// Create a quaternion from axis-angle representation
quat q_axis_angle(vec3 axis, float angle) {
    float halfAngle = angle * 0.5;
    float s = sin(halfAngle);
    return normalize(vec4(axis * s, cos(halfAngle)));
}

/// Create a quaternion from Euler angles (pitch=X, yaw=Y, roll=Z)
quat q_euler(euler e) {
    vec3 a = e.angles;

    quat qx = q_axis_angle(vec3(1,0,0), a.x);
    quat qy = q_axis_angle(vec3(0,1,0), a.y);
    quat qz = q_axis_angle(vec3(0,0,1), a.z);

    switch(e.order) {
        case XYZ: return q_mul(qz, q_mul(qy, qx));
        case XZY: return q_mul(qy, q_mul(qz, qx));
        case YXZ: return q_mul(qz, q_mul(qx, qy));
        case YZX: return q_mul(qx, q_mul(qz, qy));
        case ZXY: return q_mul(qy, q_mul(qx, qz));
        case ZYX: return q_mul(qx, q_mul(qy, qz));
        default:  return q_mul(qz, q_mul(qy, qx)); // fallback XYZ
    }
}

// Rotate a vector v by quaternion q
vec3 q_rotate(quat q, vec3 v) {
    vec3 u = q.xyz;
    float s = q.w;
    return 2.0 * dot(u, v) * u
         + (s*s - dot(u, u)) * v
         + 2.0 * s * cross(u, v);
}

// Convert quaternion to a 3x3 rotation matrix
mat3 q_mat3(quat q) {
    float x = q.x;
    float y = q.y;
    float z = q.z;
    float w = q.w;

    float x2 = x + x;
    float y2 = y + y;
    float z2 = z + z;

    float xx = x * x2;
    float yy = y * y2;
    float zz = z * z2;
    float xy = x * y2;
    float xz = x * z2;
    float yz = y * z2;
    float wx = w * x2;
    float wy = w * y2;
    float wz = w * z2;

    return mat3(
        1.0 - (yy + zz), xy + wz,       xz - wy,
        xy - wz,         1.0 - (xx + zz), yz + wx,
        xz + wy,         yz - wx,       1.0 - (xx + yy)
    );
}


// ==== Utility Functions ====

float smoothmin_ex(float a, float b, float k, out float weight) {
    weight = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);

    return mix(b, a, weight) - k * weight * (1.0 - weight);
}

float smoothmin(float a, float b, float k) {
    float weight;
    return smoothmin_ex(a, b, k, weight);
}

vec3 sampleHemisphere(vec3 n, vec2 xi) {
    float phi = 2.0 * PI * xi.x;
    float cosTheta = sqrt(1.0 - xi.y);
    float sinTheta = sqrt(xi.y);

    vec3 tangent = normalize(abs(n.x) > 0.1
        ? cross(n, vec3(0,1,0))
        : cross(n, vec3(1,0,0)));
    vec3 bitangent = cross(n, tangent);

    return normalize(tangent * cos(phi) * sinTheta +
                     bitangent * sin(phi) * sinTheta +
                     n * cosTheta);
}