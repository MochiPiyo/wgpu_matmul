

// 行列積を計算する
// Matrix<f32, N, M>
@group(0) @binding(0)
var<storage, read> lhs: array<f32>;
// Matrix<f32, M, K>
@group(0) @binding(1)
var<storage, read> rhs: array<f32>;
// Matrix<f32, N, K>
@group(0) @binding(2)
var<storage, read_write> output: array<f32>;
// メタデータ
// vec![N, M, O]
@group(0) @binding(3)
var<storage, read> sizes: vec3<u32>;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let N: u32 = sizes[0]; // Number of rows in lhs and output
    let M: u32 = sizes[1]; // Number of columns in lhs and rows in rhs
    let O: u32 = sizes[2]; // Number of columns in rhs and output

    var x = global_id.x; // Column index for output and rhs
    var y = global_id.y; // Row index for output and lhs

    var sum: f32 = 0.0;
    for(var k: u32 = 0u; k < M; k = k + 1u) {
        sum = sum + lhs[y * M + k] * rhs[k * O + x];
    }
    output[y * O + x] = sum;
}