use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Tell cargo to rerun if proto files change
    println!("cargo:rerun-if-changed=proto/maiia/worker/v1/worker.proto");
    println!("cargo:rerun-if-changed=proto/maiia/common/v1/common.proto");
    println!("cargo:rerun-if-changed=proto/grpc/health/v1/health.proto");
    println!("cargo:rerun-if-changed=proto");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);

    // Compile all proto files with file descriptor set for reflection
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .file_descriptor_set_path(out_dir.join("inference_descriptor.bin"))
        .compile_protos(
            &[
                "proto/maiia/worker/v1/worker.proto",
                "proto/maiia/common/v1/common.proto",
                "proto/grpc/health/v1/health.proto",
            ],
            &["proto"],
        )?;

    Ok(())
}
