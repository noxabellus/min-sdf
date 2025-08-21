const std = @import("std");

const rl = @import("raylib");
const rg = @import("raygui");

const math = struct {
    pub usingnamespace std.math;
    pub usingnamespace @import("linalg.zig");

    pub fn mean(arr: []f64) f64 {
        var sum: f64 = 0;
        for (arr) |value| sum += value;
        return sum / @as(f64, @floatFromInt(arr.len));
    }
};

const vec2 = math.vec2;
const vec3 = math.vec3;
const vec4 = math.vec4;
const ivec2 = math.ivec2;
const ivec3 = math.ivec3;
const ivec4 = math.ivec4;
const uvec2 = math.uvec2;
const uvec3 = math.uvec3;
const uvec4 = math.uvec4;
const quat = math.quat;
const mat4 = math.mat4;

pub fn main() anyerror!void {
    rl.setConfigFlags(.{
        .window_resizable = true,
        .fullscreen_mode = true,
    });

    rl.initWindow(0, 0, "rtx");

    defer rl.closeWindow();

    var buffer: [256]u8 = undefined;

    var run_timer = try std.time.Timer.start();
    var frame_timer = try std.time.Timer.start();

    const frame_time_avg_count = 100;
    var frame_times = [1]f64{0} ** frame_time_avg_count;
    var frame_time_index: u64 = 0;

    const shader = try rl.loadShader(null, "shaders/raymarch.fs");
    defer rl.unloadShader(shader);

    const loc_resolution = rl.getShaderLocation(shader, "uResolution");
    const loc_time = rl.getShaderLocation(shader, "uTime");

    var width: i32 = @divFloor(rl.getScreenWidth(), 3);
    var height: i32 = @divFloor(rl.getScreenHeight(), 3);

    var target = try rl.loadRenderTexture(width, height);
    defer rl.unloadRenderTexture(target);

    while (!rl.windowShouldClose()) {
        // calculate statistics
        const dt_ns = frame_timer.lap();
        const dt_ms = @as(f64, @floatFromInt(dt_ns)) / @as(f64, @floatFromInt(std.time.ns_per_ms));

        const t_seconds = @as(f32, @floatFromInt(run_timer.read())) / @as(f32, @floatFromInt(std.time.ns_per_s));

        frame_times[frame_time_index] = dt_ms;
        frame_time_index = (frame_time_index + 1) % frame_time_avg_count;

        const frame_time_avg = math.mean(&frame_times);
        const fps = @as(f64, @floatFromInt(std.time.ms_per_s)) / frame_time_avg;

        const sw = rl.getScreenWidth();
        const sh = rl.getScreenHeight();

        const cw = @divFloor(sw, 3);
        const ch = @divFloor(sh, 3);

        // ==== Raymarcher -> RenderTarget draw ====
        {
            // keep render texture size in sync with window size
            if (cw != width or ch != height) {
                rl.unloadRenderTexture(target);
                width = cw;
                height = ch;
                target = try rl.loadRenderTexture(width, height);
            }

            // Update shader uniforms
            const res = [_]f32{ @as(f32, @floatFromInt(width)), @as(f32, @floatFromInt(height)) };
            rl.setShaderValue(shader, loc_resolution, &res, rl.ShaderUniformDataType.vec2);
            rl.setShaderValue(shader, loc_time, &t_seconds, rl.ShaderUniformDataType.float);

            // Render using shader into render texture
            rl.beginTextureMode(target);
            rl.clearBackground(.black);
            rl.beginShaderMode(shader);
            rl.drawRectangle(0, 0, width, height, .white);
            rl.endShaderMode();
            rl.endTextureMode();
        }

        // ==== Rendering to window ====
        {
            // standard setup, clear & defer

            rl.beginDrawing();
            rl.clearBackground(.white);
            defer rl.endDrawing();

            // Draw rt texture scaled to fit the window
            rl.drawTexturePro(
                target.texture,
                rl.Rectangle{
                    .x = 0,
                    .y = 0,
                    .width = @floatFromInt(cw),
                    .height = @floatFromInt(ch),
                },
                rl.Rectangle{
                    .x = 0,
                    .y = 0,
                    .width = @floatFromInt(sw),
                    .height = @floatFromInt(sh),
                },
                .{ .x = 0, .y = 0 },
                0,
                .white,
            );

            // ==== GUI ====
            {
                // show statistics
                const text = try std.fmt.bufPrintZ(
                    &buffer,
                    "Resolution: {} x {}\n(Avg. {} frames)\n  MS: {d:.3}\n  FPS: {d:.1}",
                    .{ cw, ch, frame_time_avg_count, frame_time_avg, fps },
                );
                const text_width = rl.measureText(text, 20);
                rl.drawRectangle(5, 5, text_width + 10, 100, .light_gray);
                rl.drawText(text, 10, 10, 20, .black);
            }
        }
    }
}
