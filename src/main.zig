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

const font_size: f32 = 16;
const gui_font_size: f32 = font_size * 0.6;
const margin: f32 = 16;
const half_margin: f32 = 8;

pub fn main() anyerror!void {
    rl.setConfigFlags(.{
        .window_resizable = true,
        .fullscreen_mode = true,
    });

    rl.initWindow(0, 0, "rtx");

    defer rl.closeWindow();

    var tts_buffer: [256]u8 = undefined;

    var run_timer = try std.time.Timer.start();
    var frame_timer = try std.time.Timer.start();

    const frame_time_avg_count = 100;
    var frame_times = [1]f64{0} ** frame_time_avg_count;
    var frame_time_index: u64 = 0;
    const font = try rl.loadFontEx("fonts/roboto/regular.ttf", @intFromFloat(font_size), null);
    defer rl.unloadFont(font);

    const gui_font = try rl.loadFontEx("fonts/roboto/regular.ttf", @intFromFloat(gui_font_size), null);
    defer rl.unloadFont(gui_font);

    rg.setFont(gui_font);

    const shader = try rl.loadShader(null, "shaders/raymarch.fs");
    defer rl.unloadShader(shader);

    const loc_resolution = rl.getShaderLocation(shader, "uResolution");
    const loc_time = rl.getShaderLocation(shader, "uTime");

    var scale_factor: f32 = 1.0 / 3.0;

    var width: i32 = @intFromFloat(@ceil(@as(f32, @floatFromInt(rl.getScreenWidth())) * scale_factor));
    var height: i32 = @intFromFloat(@ceil(@as(f32, @floatFromInt(rl.getScreenHeight())) * scale_factor));

    var selected_fps_limit: f32 = 144;
    var active_fps_limit: i32 = @intFromFloat(selected_fps_limit);

    rl.setTargetFPS(active_fps_limit);

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

        const cw: i32 = @intFromFloat(@ceil(@as(f32, @floatFromInt(sw)) * scale_factor));
        const ch: i32 = @intFromFloat(@ceil(@as(f32, @floatFromInt(sh)) * scale_factor));

        // ==== FPS Limiter ====
        {
            const new_limit: i32 = @intFromFloat(selected_fps_limit);
            if (active_fps_limit != new_limit) {
                active_fps_limit = new_limit;
                rl.setTargetFPS(active_fps_limit);
            }
        }

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
                var fba = std.heap.FixedBufferAllocator.init(&tts_buffer);
                const allocator = fba.allocator();

                const lines = [_][:0]const u8{
                    try std.fmt.allocPrintZ(allocator, "Screen Size: {} x {}", .{ sw, sh }),
                    try std.fmt.allocPrintZ(allocator, "Ray Resolution: {} x {}", .{ cw, ch }),
                    try std.fmt.allocPrintZ(allocator, "Avg. {} frames", .{frame_time_avg_count}),
                    try std.fmt.allocPrintZ(allocator, "  MS : {d:.3}", .{frame_time_avg}),
                    try std.fmt.allocPrintZ(allocator, "  FPS: {d:.1}", .{fps}),
                    try std.fmt.allocPrintZ(allocator, "Scale Factor: {d:.2}", .{scale_factor}),
                    try std.fmt.allocPrintZ(allocator, "Limit FPS: {}", .{active_fps_limit}),
                };

                const line_indices = .{
                    .screen_size = 0,
                    .ray_resolution = 1,
                    .avg_frames = 2,
                    .avg_ms = 3,
                    .avg_fps = 4,
                    .scale_factor = 5,
                    .fps_limit = 6,
                };

                var widths = [_]f32{0} ** lines.len;

                var max_width: f32 = 0;
                for (lines, 0..) |l, i| {
                    widths[i] = @floatFromInt(rl.measureText(l, font_size));
                    max_width = @max(max_width, widths[i]);
                }

                var box_width: f32 = 100;

                while (box_width < max_width + margin * 2) box_width += 50;

                rl.drawRectangleRec(.{
                    .x = half_margin,
                    .y = half_margin,
                    .width = box_width + margin * 2,
                    .height = font_size * (lines.len + 1) + margin * 2,
                }, .light_gray);

                var line_offset: f32 = 0;

                for (lines, 0..) |l, i| {
                    rl.drawTextPro(
                        font,
                        l,
                        .{ .x = margin, .y = margin + line_offset * font_size },
                        .{ .x = 0, .y = 0 },
                        0,
                        font_size,
                        1.0,
                        .black,
                    );

                    switch (i) {
                        line_indices.scale_factor => {
                            line_offset += 1;

                            _ = rg.sliderBar(
                                .{
                                    .x = margin,
                                    .y = margin + line_offset * font_size,
                                    .width = box_width - margin * 2,
                                    .height = font_size,
                                },
                                "",
                                "",
                                &scale_factor,
                                0.1,
                                2.0,
                            );
                        },

                        line_indices.fps_limit => {
                            line_offset += 1;

                            _ = rg.sliderBar(
                                .{
                                    .x = margin,
                                    .y = margin + line_offset * font_size,
                                    .width = box_width - margin * 2,
                                    .height = font_size,
                                },
                                "",
                                "",
                                &selected_fps_limit,
                                0,
                                240,
                            );
                        },

                        else => {},
                    }

                    line_offset += 1;
                }
            }
        }
    }
}
