
mod collatz;
mod matmul;

fn main() {
    //std::env::set_var("RUST_LOG", "info");
    env_logger::init();
    pollster::block_on(collatz::run());
    pollster::block_on(matmul::run());
}
