
// 行列積を計算する
// Matrix<f32, M, K>
@group(0) @binding(0)
var<storage, read> lhs: array<vec4<f32>>;
// Matrix<f32, K, N>
@group(0) @binding(1)
var<storage, read> rhs: array<vec4<f32>>;
// Matrix<f32, M, N>
@group(0) @binding(2)
var<storage, read_write> output: array<vec4<f32>>;
// メタデータ
// vec![M, K, N]
@group(0) @binding(3)
var<storage, read> sizes: vec3<u32>;


/*

サイズを変えるときにすること
1. BM, BN, BKを決める
2. workgroup_size.xを決める(~256)
3. TM, TNを計算する
4. shared memoryの大きさを更新する
5. 呼び出し側に行き、tile_sizeをBM==BNに更新する

*/

// BM * BK == workgroup_size.x, lhsのブロックの要素数
// BN * BK == workgroup_size.x, rhsのブロックの要素数
const BM: u32 = 64u; // tile_size = 64
const BN: u32 = 64u;
const BK: u32 = 8u;
// TM * TN = BM * BN / workgroup_size.x
const TM: u32 = 8u; // 4の倍数（vec4のため）
const TN: u32 = 8u; // 4の倍数（vec4のため）
// TM_TN = TM * TN
const TMTN: u32 = 64u;

// resister cache用
const TM4: u32 = 2u; // TM / 4
const TN4: u32 = 2u; // TN / 4
const TMTN4: u32 = 16u; // TM * TN / 4

// array<f32, BM * BK / 4> or BK * BN / 4
var<workgroup> lhs_shared: array<vec4<f32>, 128>;
var<workgroup> rhs_shared: array<vec4<f32>, 128>;


// BM * BN / TM = workgroup_size.x, BM*BNはoutputのブロックの要素数。TMで割るとスレッド数
@compute @workgroup_size(64, 1, 1)
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

    let threadCol = local_id.x % (BN / TN);
    let threadRow = local_id.x / (BN / TN);

    // ブロックのシフト
    // 全体の要素数は1/4になっている
    var lhs_shift = (cRow * BM) * K / 4u;
    var rhs_shift = cCol * BN / 4u;
    var out_shift = cRow * BM * N + cCol * BN / 4u;

    // shared memoryのlhsとrhsのアクセス用
    let lhs_innerCol = local_id.x % (BK / 4u);
    let lhs_innerRow = local_id.x / (BK / 4u);
    let rhs_innerCol = local_id.x % (BN / 4u);
    let rhs_innerRow = local_id.x / (BN / 4u);

    // thread local cache
    var threadResults = array<vec4<f32>, TMTN4>();
    // register caches for lhs and rhs
    var regM = array<vec4<f32>, TM4>();
    var regN = array<vec4<f32>, TN4>();

    // outer loop over block tiles
    for (var bkIdx = 0u; bkIdx < K; bkIdx += BK) {
        // populate the Shared Memory caches

        // vec4
        // transpose lhs
        let lhs_vec4 = lhs[lhs_shift * lhs_innerRow * K + lhs_innerCol * 4u];
        lhs_shared[(lhs_innerCol * 4u + 0u) * BM + lhs_innerRow]

        rhs_shared[rhs_innerRow * BN + rhs_innerCol * 4u] = rhs[rhs_shift + rhs_innerRow * N + rhs_innerCol * 4u];
        workgroupBarrier();


        // shift
        lhs_shift += BK;
        rhs_shift += BK * N;

        // calculate per-thread results
        for (var dotIdx = 0u; dotIdx < BK; dotIdx += 1u) {
            // block into registers from shared memory
            for (var i = 0u; i < TM4; i += 1u) {
                // vectorized !
                regM[i] = lhs_shared[(dotIdx * BM + threadRow * TM) / 4u + i];
            }
            for (var i = 0u; i < TN4; i += 1u) {
                regN[i] = rhs_shared[(dotIdx * BN + threadCol * TN) / 4u + i];
            }

            // calculate
            for (var resIdxM = 0u; resIdxM < TM4; resIdxM += 1u) {
                for (var resIdxN = 0u; resIdxN < TN4; resIdxN += 1u) {
                    threadResults[resIdxM * TN4 + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        workgroupBarrier();
    }

    // write out the results
    for (var resIdxM = 0u; resIdxM < TM / 4u; resIdxM += 1u) {
        // vectorized! += 4u
        for (var resIdxN = 0u; resIdxN < TN / 4u; resIdxN += 1u) {
            // CUDAはC += shiftでずらしていたが、こっちではできないので。
            // cRow * BM, cCol * BNはworkgroupの位置

            output[out_shift + ((threadRow * TM + resIdxM) * N + threadCol * TN_4 + resIdxN] = threadResults[resIdxM * TN + resIdxN];
                
        }
    }
}