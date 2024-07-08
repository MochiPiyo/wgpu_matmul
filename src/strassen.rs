use crate::matmul_structured2::{RawGf32, Shape};


struct Strassen4 {
    bodys: [RawGf32; 4],
}
impl Strassen4 {
    fn new_fill_with(size: usize, fill_with: f32) -> Self {
        let bodys = [
            RawGf32::new_init(Shape::D2(size, size), &vec![fill_with; size*size], None),
            RawGf32::new_init(Shape::D2(size, size), &vec![fill_with; size*size], None),
            RawGf32::new_init(Shape::D2(size, size), &vec![fill_with; size*size], None),
            RawGf32::new_init(Shape::D2(size, size), &vec![fill_with; size*size], None),
        ];
        Self { bodys }
    }

    fn new_from(bodys: [RawGf32; 4]) -> Self {
        Self { bodys }
    }

    fn matmul(&self, other: &Self) -> Self {
        // シュトラッセンのアルゴリズム
        let a11 = &self.bodys[0];
        let a12 = &self.bodys[1];
        let a21 = &self.bodys[2];
        let a22 = &self.bodys[3];

        let b11 = &other.bodys[0];
        let b12 = &other.bodys[1];
        let b21 = &other.bodys[2];
        let b22 = &other.bodys[3];

        let p1 = a11.add(a22).matmul(&b11.add(b22));
        let p2 = a21.add(a22).matmul(b11);
        let p3 = a11.matmul(&b12.sub(b22));
        let p4 = a22.matmul(&b21.sub(b11));
        let p5 = a11.add(a12).matmul(b22);
        let p6 = a21.sub(a11).matmul(&b11.add(b12));
        let p7 = a12.sub(a22).matmul(&b21.add(b22));

        let c11 = p1.add(&p4).sub(&p5).add(&p7);
        let c12 = p3.add(&p5);
        let c21 = p2.add(&p4);
        let c22 = p1.add(&p3).sub(&p2).add(&p6);

        Strassen4::new_from([c11, c12, c21, c22])
    }

    fn print_1(&self) {
        for body in &self.bodys {
            body.print_1();
        }
    }
}

pub fn run() {
    let sizes = vec![1, 2, 4, 8];
    let mut results = vec![];

    for &size in sizes.iter() {
        
        // 1回計算
        let size = size * 512;
        let a = Strassen4::new_fill_with(size, 1.0);
        let b = Strassen4::new_fill_with(size, 2.0);

        let s = std::time::Instant::now();
        let c = a.matmul(&b);
        let c = c.matmul(&a);

        println!("result of c is: ");
        c.print_1();
        let time = s.elapsed();
        results.push(time);
        println!("連続１回, size = {}, time = {:?}", size, time);
    }
    for r in results.iter() {
        let micro2: String = r.as_micros().to_string().chars().take(2).collect();
        println!("{:?}.{}", r.as_millis(), micro2);
    }
}