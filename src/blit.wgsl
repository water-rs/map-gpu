// Copies the map surface's texture into the host's content texture, pixel
// for pixel: both hold premultiplied linear Display P3 in the working format.

@group(0) @binding(0) var map_surface: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    // One triangle covering the whole target.
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(map_surface, vec2<i32>(position.xy), 0);
}
