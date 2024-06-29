use std::{borrow::{BorrowMut, Cow}, collections::HashMap, ops::Deref, sync::{Arc, RwLock, RwLockReadGuard}};
use flume::r#async;
use wgpu::{util::DeviceExt, Buffer, ShaderModule};
use lazy_static::lazy_static;



/*
こっちが完成版
2024_6_10


*/

// 1 スレッド，1 デバイス，1 configの原則
thread_local! {
    static DEVICE: Wgpu = {
        let instance = wgpu::Instance::default();

        // thread_localの中ではawaitは使えないのでpollsterを使う。
        let a = pollster::block_on(
            instance.request_adapter(&wgpu::RequestAdapterOptions::default())
        );
        let adapter = a.unwrap();

        let d = pollster::block_on(
            adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
        );
        let (device, queue) = d.unwrap();

        Wgpu {
            device,
            queue,
            shader_cache: RwLock::new(HashMap::new()),
        }
    };
}


// thread_local!は
lazy_static! {
    static ref WGPU_SERVER: WgpuServer = WgpuServer::new();
}
struct Wgpu {
    // type
    // id
    device: wgpu::Device,
    queue: wgpu::Queue,

    shader_cache: RwLock<HashMap<String, wgpu::ShaderModule>>
}

// 操作を集約して，RwLockの中身を外部に送信しなくていいようにしたい
struct WgpuServer {} // 中身ないのでmodでもいいが一応structの形をとらせる
impl WgpuServer {
    fn new() -> Self {
        Self {}
    }
    fn create_buffer(size: usize, label: Option<&str>) -> wgpu::Buffer {
        DEVICE.with(|w| {
            let b = w.device.create_buffer(&wgpu::BufferDescriptor {
                label,
                size: size as wgpu::BufferAddress,
    
                /*
                emptyeは
                １．途中の計算結果を書き込んだあと，他のシェーダで読む
                ２．計算結果を書き込んだあと，ステージングバッファにコピーしてCPU側に読み出す
                なので，COPY_SRCをつけとけばよいのでは。
                STORAGEはシェーダ側，COPY_XXXはコマンドエンコーダから操作するための特性（なのでは）
                 */
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            b
        })
    }
    fn create_buffer_init<T: bytemuck::Pod>(contents: &Vec<T>, label: Option<&str>) -> wgpu::Buffer {
        DEVICE.with(|w| {
            let b = w.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label,
                contents: bytemuck::cast_slice(contents),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            b   
        })
    }
    fn execute_4(
        buf1: &wgpu::Buffer,
        buf2: &wgpu::Buffer,
        buf3: &wgpu::Buffer,
        buf4: &wgpu::Buffer,
        shader_name: &str,
        shader_str: &str, // include_str!して実行ファイルを１つにするために必要
        dispatch: (u32, u32, u32),
    ) {
        DEVICE.with(|w| {
            // まずキャッシュを読み取り専用で確認
            /*let shader_module = {
                {
                    let mut shader_cache = w.shader_cache.write().unwrap();
                    if shader_cache.contains_key(shader_name) {
                        // キャッシュに存在する場合、何もせずスコープを抜ける
                    
                    } else {
                        // キャッシュに存在しない場合、新しいShaderModuleを作成
                        
                        // 作成したShaderModuleをキャッシュに挿入
                        shader_cache.insert(shader_name.to_string(), shader_module);
                    }
                }
                {
                    let shader_cache: RwLockReadGuard<HashMap<String, ShaderModule>> = w.shader_cache.read().unwrap();
                    shader_cache.get(shader_name).unwrap()
                }
            };*/
            let shader_module = w.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_str)),
            });

            let compute_pileline = w.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader_module,
                entry_point: "main",
                // このバージョンではないっぽい
                // constantas: &Default::default(),
            });
    
            let bind_group_layout = compute_pileline.get_bind_group_layout(0);
            let bind_group = w.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                // 
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        // wgslからはbinding(0)で取れる
                        binding: 0,
                        // binding(0)の中身をさっき作成したbufferに指定
                        resource: buf1.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf2.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buf3.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buf4.as_entire_binding(),
                    }
                ],
            });
    
            // comand encoderは一つか複数のパイプラインを実行する
            let mut encoder = w.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: None,
            });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    // ない
                    // timestamp_writes: None,
                });
                cpass.set_pipeline(&compute_pileline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.insert_debug_marker("compute collatz iterations");
                cpass.dispatch_workgroups(dispatch.0, dispatch.1, dispatch.2);
            }
            
            // encoderの中身を送信
            w.queue.submit(Some(encoder.finish()));
    
            
        })

    }
    fn get(src: &wgpu::Buffer) -> Vec<f32> {
        DEVICE.with(|w| {
            // 
            let staging_buffer = w.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging buffer"),
                size: src.size() as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let mut encoder = w.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: None,
            });
            // エンコーダにコピーを指示。たぶん前のbigin_compute_passが終わったら行われる。
            // 処理結果が詰まったstorage_bufferはVRAM上にあり，それをCPUから見えるstaging_bufferに移す。
            encoder.copy_buffer_to_buffer(&src, 0, &staging_buffer, 0, src.size() as u64);

            // encoderの中身を送信
            w.queue.submit(Some(encoder.finish()));


            let buffer_slice = staging_buffer.slice(..);
            let (sender, receiver) = flume::bounded(1);
            buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

            // ブロッキング方式でデバイスポール
            // ポールは実際にはイベントループか別スレッドでやるべきらしい
            w.device.poll(wgpu::Maintain::Wait);

            // buffer_futureが読み出し可能になるまでawait
            let result: Vec<f32> = if let Ok(Ok(())) = pollster::block_on(receiver.recv_async()) {
                // get contents of buffer
                let buffer_view = buffer_slice.get_mapped_range();
                // bytes to u32
                let result = bytemuck::cast_slice(&buffer_view).to_vec();

                // 現在のインタフェースでは，bufferをunmapする前に全てのviewがドロップしている必要がある。
                drop(buffer_view); // delete pointer;
                staging_buffer.unmap(); // pointer = NULL;

                result
            } else {
                panic!("failed to run compute on gpu!")
            };

            result
        })
        
    }
}



pub enum Shape {
    //D1(usize),
    D2(usize, usize),
}
impl Shape {
    fn to_string(&self) -> String {
        if let Self::D2(i, j) = self {
            let s = format!("Shape::D2({}, {})",i, j);
            return s;
        } else {
            unimplemented!()
        };
    }
    fn size(&self) -> usize {
        if let Self::D2(i, j) = self {
            return i*j;
        } else {
            unimplemented!()
        };
    }
}


struct RawGf32 {
    label: Option<String>,
    shape: Shape,
    buffer: wgpu::Buffer,
}
impl RawGf32 {
    // internal
    fn _new_empty(shape: Shape, label: Option<&str>) -> Self {
        let size = if let Shape::D2(i, j) = shape {
            i * j * 4 // f32 is 4 Byte
        } else {
            panic!();
        };

        let buffer = WgpuServer::create_buffer(size, label);
        
        Self {
            label: label.map(|str| str.to_string()),
            shape,
            buffer,
        }
    }

    pub fn new_init(shape: Shape, values: &Vec<f32>, label: Option<&str>) -> Self {
        let buffer = WgpuServer::create_buffer_init(values, label);
        
        Self {
            label: label.map(|str| str.to_string()),
            shape,
            buffer,
        }
    }

    pub fn size(&self) -> usize {
        // f32 is 4 Byte
        self.shape.size() * 4
    }

    pub fn matmul(&self, other: &Self) -> Self {
        // 行列積の結果のサイズとシェーダのためのサイズ情報
        let (reuslt_shape, sizes_info) = {
            if let Shape::D2(i, j) = self.shape {
                if let Shape::D2(k, l) = other.shape {
                    if j != k {
                        panic!("incompatible matrix size");
                    }
                    (Shape::D2(i, l), vec![i as u32, j as u32, l as u32])
                } else {
                    panic!()
                }
            } else {
                panic!()
            }
        };
        // サイズ情報のバッファ [u32; 3]
        let size_info_buffer = DEVICE.with(|w| {
            let b = w.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sizes info"),
                contents: bytemuck::cast_slice(&sizes_info),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            b
        });
        // 結果のバッファ確保
        let result = Self::_new_empty(reuslt_shape, Some("result"));

        WgpuServer::execute_4(
            &self.buffer,
            &other.buffer,
            &result.buffer,
            &size_info_buffer,
            //"matmul.wgsl",
            //include_str!("./matmul.wgsl"),
            //(sizes_info[0] as u32, sizes_info[2] as u32, 1)
            "matmul2.wgsl",
            include_str!("./matmul2.wgsl"),
            // tile size = 16
            (sizes_info[0] as u32 / 16, sizes_info[2] as u32/ 16, 1)
        );

        return  result;
    }

    pub fn print_1(&self) {
        let result = WgpuServer::get(&self.buffer);

        // 出力する
        println!("shape: {}, body[0]: {:?}", self.shape.to_string(), result[0]);
    }
}




pub fn run() {

    // 計算データをバッファに確保
    let s = std::time::Instant::now();
    let a = RawGf32::new_init(Shape::D2(2, 2), &vec![1.0; 4], Some("a"));
    let b = RawGf32::new_init(Shape::D2(2, 2), &vec![2.0; 4], Some("b"));
    println!("1, {:?}", s.elapsed());

    // gpuで計算（シェーダコンパイル，パイプライン，ディスパッチ）
    let s = std::time::Instant::now();
    let c = a.matmul(&b);
    println!("2, {:?} // async", s.elapsed());

    // staging bufferを利用してデータを読み出し
    println!("result of c is: ");
    let s = std::time::Instant::now();
    c.print_1();
    println!("3, {:?}", s.elapsed());

    // 連続した計算
    let d = RawGf32::new_init(Shape::D2(2, 2), &vec![3.0; 4], Some("d"));
    // staging bufferを利用してデータを読み出し
    let e = c.matmul(&d);

    println!("result of e is: ");
    e.print_1();




    // --------------------------------------
    // データ転送の時間を特定
    // 結果はspeed_result.text(.gitginore)に記載
    
    let size = 1024*4;

    // 1回計算
    let s = std::time::Instant::now();
    let a = RawGf32::new_init(Shape::D2(size, size), &vec![1.0; size*size], Some("a"));
    let b = RawGf32::new_init(Shape::D2(size, size), &vec![2.0; size*size], Some("b"));

    let c = a.matmul(&b);

    println!("result of e is: ");
    c.print_1();
    println!("連続１回，{:?}", s.elapsed());

    // 2回計算
    let s = std::time::Instant::now();
    let a = RawGf32::new_init(Shape::D2(size, size), &vec![1.0; size*size], Some("a"));
    let b = RawGf32::new_init(Shape::D2(size, size), &vec![2.0; size*size], Some("b"));

    let b2 = RawGf32::new_init(Shape::D2(size, size), &vec![2.0; size*size], Some("b"));

    let c = a.matmul(&b);
    let e = c.matmul(&b2);

    println!("result of e is: ");
    e.print_1();
    println!("連続２回，{:?}", s.elapsed());
    
}