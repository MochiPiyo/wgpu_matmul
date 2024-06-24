use std::borrow::Cow;
use wgpu::util::DeviceExt;

/*

1. @8565u

500
execute and get back: 352msec
cpu execute: 50msec

800
execute and get back: 2141msec
cpu execute: 215msec

1000
execute and get back: 5010msec
cpu execute: 850msec

2000からGPUの計算結果が0になるのadapter limitsのこれが原因では
max_compute_workgroup_size_x: 1024,
max_compute_workgroup_size_y: 1024,
dviceは256になってるからdeviceのlimitより大きくても大丈夫らしい。
1024よりでかくできないのか？
Error in Adapter::request_device: Limit 'max_compute_workgroup_size_y' value 2048 is better than allowed 1024

adapter info: AdapterInfo {
    name: "Intel(R) UHD Graphics 620",
    vendor: 32902,
    device: 16032,
    device_type: IntegratedGpu,
    driver: "Intel Corporation",
    driver_info: "Intel driver",
    backend: Vulkan,
}
adapter limits: Limits {
    max_texture_dimension_1d: 16384,
    max_texture_dimension_2d: 16384,
    max_texture_dimension_3d: 2048,
    max_texture_array_layers: 2048,
    max_bind_groups: 8,
    max_bindings_per_bind_group: 1000,
    max_dynamic_uniform_buffers_per_pipeline_layout: 16,
    max_dynamic_storage_buffers_per_pipeline_layout: 16,
    max_sampled_textures_per_shader_stage: 200,
    max_samplers_per_shader_stage: 64,
    max_storage_buffers_per_shader_stage: 200,
    max_storage_textures_per_shader_stage: 16,
    max_uniform_buffers_per_shader_stage: 200,
    max_uniform_buffer_binding_size: 134217724,
    max_storage_buffer_binding_size: 1073741820,
    max_vertex_buffers: 16,
    max_buffer_size: 18446744073709551615,
    max_vertex_attributes: 32,
    max_vertex_buffer_array_stride: 4095,
    min_uniform_buffer_offset_alignment: 64,
    min_storage_buffer_offset_alignment: 64,
    max_inter_stage_shader_components: 128,
    max_compute_workgroup_storage_size: 32768,
    max_compute_invocations_per_workgroup: 1024,
    max_compute_workgroup_size_x: 1024,
    max_compute_workgroup_size_y: 1024,
    max_compute_workgroup_size_z: 64,
    max_compute_workgroups_per_dimension: 65536,
    max_push_constant_size: 256,
}
hardware limits: Limits {
    max_texture_dimension_1d: 2048,
    max_texture_dimension_2d: 2048,
    max_texture_dimension_3d: 256,
    max_texture_array_layers: 256,
    max_bind_groups: 4,
    max_bindings_per_bind_group: 1000,
    max_dynamic_uniform_buffers_per_pipeline_layout: 8,
    max_dynamic_storage_buffers_per_pipeline_layout: 4,
    max_sampled_textures_per_shader_stage: 16,
    max_samplers_per_shader_stage: 16,
    max_storage_buffers_per_shader_stage: 4,
    max_storage_textures_per_shader_stage: 4,
    max_uniform_buffers_per_shader_stage: 12,
    max_uniform_buffer_binding_size: 16384,
    max_storage_buffer_binding_size: 134217728,
    max_vertex_buffers: 8,
    max_buffer_size: 268435456,
    max_vertex_attributes: 16,
    max_vertex_buffer_array_stride: 2048,
    min_uniform_buffer_offset_alignment: 256,
    min_storage_buffer_offset_alignment: 256,
    max_inter_stage_shader_components: 60,
    max_compute_workgroup_storage_size: 16352,
    max_compute_invocations_per_workgroup: 256,
    max_compute_workgroup_size_x: 256,
    max_compute_workgroup_size_y: 256,
    max_compute_workgroup_size_z: 64,
    max_compute_workgroups_per_dimension: 65535,
    max_push_constant_size: 0,
}
*/


pub async fn run(x: usize) {


    let n = x;//1000;
    let m = x;//1000;
    let o = x;//1000;
    


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


    println!("adapter info: {:#?}", adapter.get_info());
    println!("adapter limits: {:#?}", adapter.limits());
    println!("hardware limits: {:#?}", device.limits());
    


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
    //println!("gpu: {:?}", result[0]);

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
    //println!("cpu: {:?}", cpu_result[0]);

    for (c, g) in cpu_result.iter().zip(result.iter()) {
        if c != g {
            panic!("cpu: {}, gpu: {}", c, g);
        }
    }

    /*
    
    
    
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
     */
}