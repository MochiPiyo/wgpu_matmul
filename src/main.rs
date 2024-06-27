
mod collatz;
mod matmul;
mod matmul_structured2;

fn main() {
    std::env::set_var("RUST_LOG", "warn");
    env_logger::init();
    //pollster::block_on(collatz::run());

    /*
    let nvec: Vec<usize> = vec![500, 800, 1000, 2000, 3000, 4000];
    for &n in nvec.iter() {
        println!("n = {}", n);
        pollster::block_on(matmul::run(n));
    }
    */

   // pollster::block_on(matmul::run());
   matmul_structured2::run();
    
}
