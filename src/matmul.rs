use std::borrow::Cow;
use wgpu::util::DeviceExt;

/*

1. @8565u
let n = 100;
let m = 100;
let o = 100;
の条件においてGPU: 5msec, CPU: 60msecなのではやい

get device: 113msec
buffer allocate and binding, shader compile: 3msec
execute and get back: 5msec
gpu: 200.0
cpu execute: 60msec
cpu: 200.0


*/


pub async fn run() {

    
    let n = 100;
    let m = 100;
    let o = 100;
    


    let lhs: Vec<f32> = vec![2.0; n*m];
    let rhs: Vec<f32> = vec![1.0; m*o];
    let output_mem_size = std::mem::size_of::<f32>() * n * o;
    let sizes: Vec<u32> = vec![n as u32, m as u32, o as u32];
    
    let start = std::time::Instant::now();
    // instance -> adapter -> device
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await
        .unwrap();

    println!("get device: {}msec", start.elapsed().as_millis());
    let start = std::time::Instant::now();

    // device, queue, numbers
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("matmul.wgsl"))),
    });

    
    // buffer with data
    let lhs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage Buffer1"),
        contents: bytemuck::cast_slice(&lhs),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let rhs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage Buffer2"),
        contents: bytemuck::cast_slice(&rhs),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    // buffer without data
    // ステージングバッファはCPU RAMとVRAMの中継のためのバッファ。データをGPUから取り出すために使う。
    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: output_mem_size as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let size_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sizes info"),
        contents: bytemuck::cast_slice(&sizes),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging buffer"),
        size: output_mem_size as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });



    let compute_pileline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader_module,
        entry_point: "main",
        // このバージョンではないっぽい
        //constantas: &Default::default(),
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
                resource: lhs_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rhs_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: result_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: size_buffer.as_entire_binding(),
            }
        ],
    });

    println!("buffer allocate and binding, shader compile: {}msec", start.elapsed().as_millis());
    let start = std::time::Instant::now();
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
        cpass.dispatch_workgroups(n as u32, o as u32, 1);
    }
    // エンコーダにコピーを指示。たぶん前のbigin_compute_passが終わったら行われる。
    // 処理結果が詰まったstorage_bufferはVRAM上にあり，それをCPUから見えるstaging_bufferに移す。
    encoder.copy_buffer_to_buffer(&result_buffer, 0, &staging_buffer, 0, output_mem_size as u64);

    // encoderの中身を送信
    queue.submit(Some(encoder.finish()));

    

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // ブロッキング方式でデバイスポール
    // ポールは実際にはイベントループか別スレッドでやるべきらしい
    device.poll(wgpu::Maintain::Wait);

    // buffer_futureが読み出し可能になるまでawait
    let result: Vec<f32> = if let Ok(Ok(())) = receiver.recv_async().await {
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

    println!("execute and get back: {}msec", start.elapsed().as_millis());

    // print result
    println!("gpu: {:?}", result[0]);

    let mut cpu_result = vec![0.0; n*o];
    let start = std::time::Instant::now();
    for i in 0..n {
        for j in 0..m {
            for k in 0..o {
                cpu_result[i * o + k] += lhs[i * m + j] * rhs[j * o + k]
            }
        }
    }
    println!("cpu execute: {}msec", start.elapsed().as_millis());
    println!("cpu: {:?}", cpu_result[0]);

    let mut cpu_result = vec![0.0; n*o];
    let start = std::time::Instant::now();
    for i in 0..n {
        for k in 0..o {
            for j in 0..m {
                cpu_result[i * o + k] += lhs[i * m + j] * rhs[j * o + k]
            }
        }
    }
    println!("cpu execute(ikj): {}msec", start.elapsed().as_millis());
    println!("cpu: {:?}", cpu_result[0]);
}