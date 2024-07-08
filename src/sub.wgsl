@group(0) @binding(0)
var<storage, read> lhs: array<f32>;

@group(0) @binding(1)
var<storage, read> rhs: array<f32>;

@group(0) @binding(2)
var<storage, read_write> out: array<f32>;


@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
) {
    let index = global_invocation_id.x + global_invocation_id.y * 64u;
    out[index] = lhs[index] - rhs[index];
}