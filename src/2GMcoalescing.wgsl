
// 行列積を計算する
// Matrix<f32, M, K>
@group(0) @binding(0)
var<storage, read> lhs: array<f32>;
// Matrix<f32, K, N>
@group(0) @binding(1)
var<storage, read> rhs: array<f32>;
// Matrix<f32, M, N>
@group(0) @binding(2)
var<storage, read_write> output: array<f32>;
// メタデータ
// vec![M, K, N]
@group(0) @binding(3)
var<storage, read> sizes: vec3<u32>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let M: u32 = sizes[0]; // Number of rows in lhs and output
    let K: u32 = sizes[1]; // Number of columns in lhs and rows in rhs
    let N: u32 = sizes[2]; // Number of columns in rhs and output

     /*global_memory_coalescing
    naiveな実装のx, yの順序を入れ替えただけのもの。
    メモリアクセスが連続的になる。
    https://siboehm.com/articles/22/CUDA-MMM
    */


    let tile_size = 16u;
    var x = global_id.x * tile_size + (local_id.x / tile_size); // Column index for output and rhs
    var y = global_id.y * tile_size + (local_id.x % tile_size); // Row index for output and lhs

    if (x < M && y < N) {
        var sum: f32 = 0.0;
        for(var i: u32 = 0u; i < K; i += 1u) {
            sum = sum + lhs[x * K + i] * rhs[i * N + y];
        }
        output[y * N + x] = sum;
    }
    
}