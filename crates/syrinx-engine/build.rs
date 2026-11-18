//! Compiles the CUDA kernels (`kernels/*.cu`) to PTX, which the engine embeds and the
//! NVIDIA driver compiles for the GPU it finds when the engine starts. So the engine
//! ships no CUDA libraries: the driver is all it needs.
//!
//! PTX for compute capability 7.5 runs on every NVIDIA GPU from the RTX 20 series on
//! (CUDA 13's oldest target); a second build for 8.0 adds the tensor-core kernels. Without the CUDA Toolkit, the engine builds without the
//! kernels and reports the GPU as unavailable.
//!
//! nvcc is found from the NVCC or CUDA_PATH variables, or the newest toolkit in its
//! default place. On Windows nvcc needs Visual Studio's C++ compiler even for PTX; it's
//! found with vswhere, so no developer command prompt is needed.

use std::path::{Path, PathBuf};
use std::process::Command;

const PTX_TARGET: &str = "compute_75";
/// The tensor cores' 8-bit matrix multiply (mma.m16n8k16 on s8) needs 8.0.
const PTX_TARGET_MMA: &str = "compute_80";

fn main() {
    println!("cargo::rustc-check-cfg=cfg(syrinx_cuda)");
    println!("cargo::rerun-if-changed=kernels");
    println!("cargo::rerun-if-env-changed=NVCC");
    println!("cargo::rerun-if-env-changed=CUDA_PATH");
    // On the Mac the GPU backend is Metal (src/metal), compiled when the GPU is opened.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        return;
    }

    let Some(nvcc) = find_nvcc() else {
        println!("cargo::warning=CUDA Toolkit not found: building without GPU kernels");
        return;
    };
    // Two builds of the same kernels: one every supported GPU runs, and one for compute
    // capability 8.0 on (RTX 30 series and later) with the tensor-core kernels too.
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    for (target, name) in [
        (PTX_TARGET, "vireo.ptx"),
        (PTX_TARGET_MMA, "syrinx_sm80.ptx"),
    ] {
        let mut command = Command::new(&nvcc);
        command
            .args(["-ptx", "-O3", "-std=c++17"])
            .arg(format!("-arch={target}"))
            .arg("-o")
            .arg(out_dir.join(name))
            .arg("kernels/syrinx.cu");
        if cfg!(windows) {
            match find_msvc() {
                Some(cl) => {
                    command.arg("-ccbin").arg(cl);
                }
                None => {
                    println!(
                        "cargo::warning=Visual Studio's C++ tools not found: building without GPU kernels"
                    );
                    return;
                }
            }
        }
        let output = command.output().expect("couldn't run nvcc");
        if !output.status.success() {
            panic!(
                "nvcc failed to compile the kernels for {target}:
{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
    println!("cargo::rustc-cfg=syrinx_cuda");
}

fn find_nvcc() -> Option<PathBuf> {
    let exe = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
    if let Some(path) = std::env::var_os("NVCC") {
        return Some(path.into());
    }
    if let Some(root) = std::env::var_os("CUDA_PATH") {
        let path = Path::new(&root).join("bin").join(exe);
        if path.exists() {
            return Some(path);
        }
    }
    let roots: &[&str] = if cfg!(windows) {
        &[r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"]
    } else {
        &["/usr/local"]
    };
    let mut found: Vec<PathBuf> = roots
        .iter()
        .filter_map(|root| std::fs::read_dir(root).ok())
        .flatten()
        .flatten()
        .map(|entry| entry.path().join("bin").join(exe))
        .filter(|path| path.exists())
        .collect();
    // "v13.5" after "v12.4"; "cuda-13.5" after "cuda-12.4".
    found.sort_by_key(|nvcc| version_of(nvcc.parent().and_then(Path::parent)));
    found.pop()
}

/// The numbers in a folder's name, for sorting versions: "v13.5" is [13, 5].
fn version_of(folder: Option<&Path>) -> Vec<u32> {
    let name = folder
        .and_then(Path::file_name)
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    name.split(|c: char| !c.is_ascii_digit())
        .filter_map(|part| part.parse().ok())
        .collect()
}

/// The folder holding Visual Studio's 64-bit `cl.exe`.
fn find_msvc() -> Option<PathBuf> {
    let vswhere =
        Path::new(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe");
    let output = Command::new(vswhere)
        .args(["-latest", "-products", "*", "-requires"])
        .arg("Microsoft.VisualStudio.Component.VC.Tools.x86.x64")
        .args(["-property", "installationPath"])
        .output()
        .ok()?;
    let install = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let tools = Path::new(&install).join(r"VC\Tools\MSVC");
    let mut versions: Vec<PathBuf> = std::fs::read_dir(&tools)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .collect();
    versions.sort_by_key(|p| version_of(Some(p)));
    let cl = versions.pop()?.join(r"bin\Hostx64\x64");
    cl.join("cl.exe").exists().then_some(cl)
}
