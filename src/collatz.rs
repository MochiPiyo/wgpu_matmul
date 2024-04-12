use std::borrow::Cow;
use wgpu::util::DeviceExt;

// Indicates a u32 overflow in an intermediate Collatz value
const OVERFLOW: u32 = 0xffffffff;

pub async fn run() {
    let n = 10;
    let numbers: Vec<u32> = (1..n).collect();

    
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

    // device, queue, numbers
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("collatz.wgsl"))),
    });

    // get the size in bytes of the buffer
    let size = std::mem::size_of_val(&numbers) as wgpu::BufferAddress;

    // buffer without data
    // ステージングバッファはCPU RAMとVRAMの中継のためのバッファ。データをGPUから取り出すために使う。
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // buffer with data
    let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage Buffer"),
        contents: bytemuck::cast_slice(&numbers),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
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
        entries: &[wgpu::BindGroupEntry {
            // wgslからはbinding(0)で取れる
            binding: 0,
            // binding(0)の中身をさっき作成したbufferに指定
            resource: storage_buffer.as_entire_binding(),
        }],
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
        cpass.dispatch_workgroups(numbers.len() as u32, 1, 1);
    }

    // エンコーダにコピーを指示。たぶん前のbigin_compute_passが終わったら行われる。
    // 処理結果が詰まったstorage_bufferはVRAM上にあり，それをCPUから見えるstaging_bufferに移す。
    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, size);

    // encoderの中身を送信
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // ブロッキング方式でデバイスポール
    // ポールは実際にはイベントループか別スレッドでやるべきらしい
    device.poll(wgpu::Maintain::Wait);

    // buffer_futureが読み出し可能になるまでawait
    let result: Vec<u32> = if let Ok(Ok(())) = receiver.recv_async().await {
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


    // print result
    let disp_steps: Vec<String> = result.iter().map(|&n| match n {
        OVERFLOW => "OVERFLOW".to_string(),
        _ => n.to_string(),
    }).collect();

    println!("Steps: [{}]", disp_steps.join(", "));
}