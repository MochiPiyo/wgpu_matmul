
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

var<workgroup> shared_lhs: array<f32, 256>;
var<workgroup> shared_rhs: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let M: u32 = sizes[0]; // Number of rows in lhs and output
    let K: u32 = sizes[1]; // Number of columns in lhs and rows in rhs
    let N: u32 = sizes[2]; // Number of columns in rhs and output

     /* Shared Memory Cache-Blocking
    yが下向き、xが右向き方向

    
    https://siboehm.com/articles/22/CUDA-MMM
    */

    

    let tile_size = 16u;



    // このworkgroup(block)が担当する場所まで飛ばすシフト
    var lhs_shift = workgroup_id.y * tile_size * K;
    var rhs_shift = workgroup_id.x * tile_size;
    var out_shift = workgroup_id.y * tile_size * N + workgroup_id.x * tile_size;

    var sum = 0.0;

    // bkIdx == tile_id
    for (var bkIdx = 0u; bkIdx < K; bkIdx += tile_size) {
        // yが横、ｘが縦
        shared_lhs[local_id.y * tile_size + local_id.x] = lhs[lhs_shift + local_id.y * K + local_id.x];
        shared_rhs[local_id.y * tile_size + local_id.x] = rhs[rhs_shift + local_id.y * N + local_id.x];
        
        workgroupBarrier();

        // 次のブロックの位置までシフト
        lhs_shift += tile_size;
        rhs_shift += tile_size * N;

        // execute the dotproduct on the currently cached block
        for (var dotIdx = 0u; dotIdx < tile_size; dotIdx += 1u) {
            sum += shared_lhs[local_id.y * tile_size + dotIdx] * shared_rhs[dotIdx * tile_size + local_id.x];
        } 

        workgroupBarrier();
    }
    output[out_shift + local_id.y * N + local_id.x] = sum;
}