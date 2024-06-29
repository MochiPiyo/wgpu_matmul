


/*　最適化を入れたバージョン



*/

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


// 共有メモリの確保
var<workgroup> local_lhs: array<f32, 256>; // 16 * 16
var<workgroup> local_rhs: array<f32, 256>;


// tile size = 16
@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let N: u32 = sizes[0];
    let M: u32 = sizes[1];
    let O: u32 = sizes[2];
    let tile_size: u32 = 16u;

    var x = global_id.x;
    var y = global_id.y;
    var local_x = local_id.x;
    var local_y = local_id.y;

    
    var sum: f32 = 0.0;

    // タイリングを使用した行列積
    for (var tile = 0u; tile < (M + tile_size - 1u) / tile_size; tile++) {
        // 共有メモリにデータをロード
        let idx_lhs = y * M + (tile * tile_size + local_x);
        let idx_rhs = (tile * tile_size + local_y) * O + x;
        if (idx_lhs < N * M) {
            local_lhs[local_y * tile_size + local_x] = lhs[idx_lhs];
        } else {
            // 16*16のworkgroup_sizeで計算するのではみ出した分は0.0で埋める
            local_lhs[local_y * tile_size + local_x] = 0.0;
        }
        if (idx_rhs < M * O) {
            local_rhs[local_y * tile_size + local_x] = rhs[idx_rhs];
        } else {
            local_rhs[local_y * tile_size + local_x] = 0.0;
        }

        // バリアを使用して全スレッドがロードを完了させる
        workgroupBarrier();

        // タイル内での行列積
        for (var k = 0u; k < tile_size; k += 1u) {
            sum += local_lhs[local_y * tile_size + k] * local_rhs[k * tile_size + local_x];
            // ループアンローリングは有意差なし。パイプラインとかないからかな
        }

        // 再びバリアを使用して計算を同期
        workgroupBarrier();
    }

    // 結果を書き出す
    if (x < O && y < N) {
        output[y * O + x] = sum;
    }
}