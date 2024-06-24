use std::{borrow::{BorrowMut, Cow}, collections::HashMap, ops::Deref, sync::{Arc, RwLock, RwLockReadGuard}};
use flume::r#async;
use wgpu::{util::DeviceExt, Buffer, ShaderModule};
use lazy_static::lazy_static;



/*

wgpu::DeviceをRwLockで引き回すのに限界を感じ，WgpuServerを採用して
matmul_stuctured2.rsに続きを書く
これはRwLockのアンチパターンのサンプルとして残しておく
そういえば，未踏採択おめでとう
2024_6_10



将来的には，これはモジュールに隠蔽したい

1. thread_local! DEVICE: Device
2. enum DeviceEnum and struct Device
// ここから使え
mod wgpu_device {
    get_device() // DEVICE.withを隠蔽するラッパ
}




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

        Wgpu {device, queue}
    };
}



/* このthread_local DEVICEを使うのはWgpuバックエンドのRawTensorだけだからenumはいらない

enum DeviceEnum {
    Cpu, // default
    Wgpu(Wgpu),
    // Cuda
}
impl DeviceEnum {
    fn get_wgpu_device(&self) -> &wgpu::Device {
        if let Self::Wgpu(wgpu) = self {
            &wgpu.device
        } else {
            panic!()
        }
    }
    fn get_wgpu_queue(&self) -> &wgpu::Queue {
        if let Self::Wgpu(wgpu) = self {
            &wgpu.queue
        } else {
            panic!()
        }
    }
}*/

// thread_local!は
lazy_static! {
    static ref WGPU_SERVER: WgpuServer = WgpuServer::new();
}
// 操作を集約して，RwLockの中身を外部に送信しなくていいようにしたい
struct WgpuServer {
}
impl WgpuServer {
    fn new() -> Self {
        Self {}
    }
    fn execute_4(
        buf1: wgpu::Buffer,
        buf2: wgpu::Buffer,
        buf3: wgpu::Buffer,
        buf4: wgpu::Buffer,
        shader: &str,
        dispatch: (u32, u32, u32),
    ) {

    }
    fn get<T>(src: wgpu::Buffer) -> Vec<T> {

    }
}

struct Wgpu {
    // type
    // id
    device: wgpu::Device,
    queue: wgpu::Queue,
}
struct Device {
    device: RwLock<DeviceEnum>,
    shader_cache: RwLock<HashMap<String, wgpu::ShaderModule>>,
}
impl Device {
    /* 
    fn new_cpu() -> Self {
        Self {
            device: RwLock::new(DeviceEnum::Cpu),
            shader_cache: RwLock::new(HashMap::new()),
        }
    }
    
    fn set_wgpu(&self, wgpu: Wgpu) {
        let mut write = self.device.write().unwrap();
        *write = DeviceEnum::Wgpu(wgpu);
        // デバイスが変わったらシェーダのビルド結果は変わるので廃棄しなければならない
        let mut shader_cache_write = self.shader_cache.write().unwrap();
        *shader_cache_write = HashMap::new();
    }*/
    fn new_wgpu(wgpu: Wgpu) -> Self {
        Self {
            device: RwLock::new(DeviceEnum::Wgpu(wgpu)),
            shader_cache: RwLock::new(HashMap::new()),
        }
    }
    fn get_device_read(&self) -> RwLockReadGuard<DeviceEnum> {
        self.device.read().unwrap()
    }
    fn get_shader_module(&mut self, shader_name: &str, shader_str: &str) -> ShaderModule {
        // まずキャッシュを読み取り専用で確認
        let mut shader_cache = self.shader_cache.write().unwrap();
        if shader_cache.contains_key(shader_name) {
            // none
        } else {
            // キャッシュに存在しない場合、新しいShaderModuleを作成
            let device_lock: RwLockReadGuard<DeviceEnum> = self.get_device_read();
            let device: &wgpu::Device = device_lock.get_wgpu_device();
            let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_str)),
            });
            // 作成したShaderModuleをキャッシュに挿入
            shader_cache.insert(shader_name.to_string(), shader_module);
        }
        // キャッシュに挿入したShaderModuleを返す
        shader_cache.get(shader_name).unwrap()
    }
}

mod wgpu_device {

    use std::sync::RwLockReadGuard;

    // create_buffer_init()はこのtraitをimportしないと使えない
    use wgpu::util::DeviceExt;

    use super::DeviceEnum;

    pub fn get_shader<'a>(shader_name: &str, shader_str: &str) -> &'a wgpu::ShaderModule {
        super::DEVICE.with(|d| {
            d.get_shader_module(shader_name, shader_str)
        })
    }

    pub fn get_device_read<'a>() -> RwLockReadGuard<'a, DeviceEnum> {
        super::DEVICE.with(|d| {
            d.get_device_read()
        })
    }

    // empty
    pub fn create_buffer(size: usize, label: Option<&str>) -> wgpu::Buffer {
        super::DEVICE.with(|d| {
            let b = d.get_device_read().get_wgpu_device().create_buffer(&wgpu::BufferDescriptor {
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
    pub fn create_buffer_init(values: &Vec<f32>, label: Option<&str>) -> wgpu::Buffer {
        super::DEVICE.with(|f| {
            let b = f.get_device_read().get_wgpu_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label,
                contents: bytemuck::cast_slice(values),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            b
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
        // thread_localにアクセスする.with(Fn)
        let buffer = wgpu_device::create_buffer(size, label);
        
        Self {
            label: label.map(|str| str.to_string()),
            shape,
            buffer,
        }
    }

    pub fn new_init(shape: Shape, values: &Vec<f32>, label: Option<&str>) -> Self {
        let buffer = wgpu_device::create_buffer_init(values, label);
        
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
        let size_info_buffer = DEVICE.with(|f| {
            let b = f.get_device_read().get_wgpu_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sizes info"),
                contents: bytemuck::cast_slice(&sizes_info),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            b
        });
        // 結果のバッファ確保
        let result = Self::_new_empty(reuslt_shape, Some("result"));

        // DEVICEのshader_cacheからシェーダを入手（ない場合はソースからビルド）
        let shader_module = wgpu_device::get_shader("matmul.wgsl", include_str!("./matmul.wgsl"));
        
        let device_lock = wgpu_device::get_device_read();
        let device = device_lock.get_wgpu_device();
        let compute_pileline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader_module,
            entry_point: "main",
            // このバージョンではないっぽい
            // constantas: &Default::default(),
        });

        let bind_group_layout = compute_pileline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            // 
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    // wgslからはbinding(0)で取れる
                    binding: 0,
                    // binding(0)の中身をさっき作成したbufferに指定
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: other.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: size_info_buffer.as_entire_binding(),
                }
            ],
        });

        // comand encoderは一つか複数のパイプラインを実行する
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
            cpass.dispatch_workgroups(sizes_info[0] as u32, sizes_info[2] as u32, 1);
        }
        
        // encoderの中身を送信
        wgpu_device::get_device_read().get_wgpu_queue().submit(Some(encoder.finish()));

        return  result;
    }

    pub fn print(&self) {
        let device_lock = wgpu_device::get_device_read();
        let device = device_lock.get_wgpu_device();
        // 
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging buffer"),
            size: self.size() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None,
        });
        // エンコーダにコピーを指示。たぶん前のbigin_compute_passが終わったら行われる。
        // 処理結果が詰まったstorage_bufferはVRAM上にあり，それをCPUから見えるstaging_bufferに移す。
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, self.size() as u64);

        // encoderの中身を送信
        wgpu_device::get_device_read().get_wgpu_queue().submit(Some(encoder.finish()));


        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        // ブロッキング方式でデバイスポール
        // ポールは実際にはイベントループか別スレッドでやるべきらしい
        device.poll(wgpu::Maintain::Wait);

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

        // 出力する
        print!("shape: {}, body: {:?}", self.shape.to_string(), result);
    }
}




pub async fn run() {

    // 計算データをバッファに確保
    let a = RawGf32::new_init(Shape::D2(2, 2), &vec![1.0; 4], Some("a"));
    let b = RawGf32::new_init(Shape::D2(2, 2), &vec![2.0; 4], Some("b"));
    
    // gpuで計算（シェーダコンパイル，パイプライン，ディスパッチ）
    let c = a.matmul(&b);

    // staging bufferを利用してデータを読み出し
    c.print();

    // 連続した計算
    let d = RawGf32::new_init(Shape::D2(2, 2), &vec![3.0; 4], Some("d"));
    // staging bufferを利用してデータを読み出し
    let e = c.matmul(&d);

    e.print();
}