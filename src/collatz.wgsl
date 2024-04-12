// input and output
@group(0) @binding(0) var<storage, read_write> v_indices: array<u32>;

// the Collatz Conjecture states that for any integer n:
// if n is even, n = n/2
// if n is odd, n = 3n+1
// and repeat this process for eatch new n, you will ways eventurally reach 1.

// thie function returns how many times this recurrence needs to be applied to reach 1.
fn collatz_iterations(n_base: u32) -> u32 {
    var n: u32 = n_base;
    var i: u32 = 0u;

    loop {
        if (n <= 1u) {
            break;
        }
        if (n % 2u == 0u) {
            n = n / 2u;
        } else {
            if (n >= 0x55555555u) {
                return 0xffffffffu;
            }
             n = 3u * n + 1u;
        }
        // 反復に要した回数
        i = i + 1u;
    }
    return i;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    v_indices[global_id.x] = collatz_iterations(v_indices[global_id.x]);
}
