
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


/*

サイズを変えるときにすること
1. BM, BN, BKを決める
2. workgroup_size.xを決める(~256)
3. TMを計算する
4. shared memoryの大きさを更新する
5. 呼び出し側に行き、tile_sizeをBM==BNに更新する

*/

// BM * BK == workgroup_size.x, lhsのブロックの要素数
// BN * BK == workgroup_size.x, rhsのブロックの要素数
const BM: u32 = 64u; // tile_size = 64
const BN: u32 = 64u;
const BK: u32 = 8u;
// TM = BM * BN / workgroup_size.x
const TM: u32 = 16u;

// array<f32, BM * BK> or BK * BN
var<workgroup> lhs_shared: array<f32, 512>;
var<workgroup> rhs_shared: array<f32, 512>;


// BM * BN / TM = workgroup_size.x, BM*BNはoutputのブロックの要素数。TMで割るとスレッド数
@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let M: u32 = sizes[0]; // Number of rows in lhs and output
    let K: u32 = sizes[1]; // Number of columns in lhs and rows in rhs
    let N: u32 = sizes[2]; // Number of columns in rhs and output

     /* 
    yが下向き、xが右向き方向

    
    https://siboehm.com/articles/22/CUDA-MMM
    */

    

    let cRow = workgroup_id.y;
    let cCol = workgroup_id.x;

    let threadCol = local_id.x % BN;
    let threadRow = local_id.x / BN;

    // ブロックのシフト
    var lhs_shift = (cRow * BM) * K;
    var rhs_shift = cCol * BN;
    var out_shift = cRow * BM * N + cCol * BN;

    // lhsとrhsのアクセス用
    let lhs_innerCol = local_id.x % BK;
    let lhs_innerRow = local_id.x / BK;
    let rhs_innerCol = local_id.x % BN;
    let rhs_innerRow = local_id.x / BN; 

    var threadResults = array<f32, TM>();

    // outer loop over block tiles
    for (var bkIdx = 0u; bkIdx < K; bkIdx += BK) {
        // populate the Shared Memory caches
        lhs_shared[lhs_innerRow * BK + lhs_innerCol] = lhs[lhs_shift + lhs_innerRow * K + lhs_innerCol];
        rhs_shared[rhs_innerRow * BN + rhs_innerCol] = rhs[rhs_shift + rhs_innerRow * N + rhs_innerCol];
        workgroupBarrier();

        // shift
        lhs_shift += BK;
        rhs_shift += BK * N;

        // calculate per-thread results
        for (var dotIdx = 0u; dotIdx < BK; dotIdx += 1u) {
            var tmp = rhs_shared[dotIdx * BN + threadCol];
            for (var resIdx = 0u; resIdx < TM; resIdx += 1u) {
                threadResults[resIdx] += lhs_shared[(threadRow * TM + resIdx) * BK + dotIdx] * tmp;
            }
        }
        workgroupBarrier();
    }

    // write out the results
    for (var resIdx = 0u; resIdx < TM; resIdx += 1u) {
        // CUDAはC += shiftでずらしていたが、こっちではできないので。
        // cRow * BM, cCol * BNはworkgroupの位置
        output[out_shift + (threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];
    }
}