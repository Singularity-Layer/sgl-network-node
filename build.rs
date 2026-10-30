// macOS only: compile src/enclave/se.swift (Secure Enclave identity key) into a static library
// and link it in, so the node stays one binary. Every other target builds exactly as before.
//
// Escape hatch: SGL_NO_SWIFT=1 skips the Swift build; the node then compiles the stub in
// src/se_identity.rs and simply never reports an enclave key (nothing depends on it).

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/enclave/se.swift");
    println!("cargo:rerun-if-env-changed=SGL_NO_SWIFT");
    println!("cargo:rustc-check-cfg=cfg(sgl_se)");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    if env::var("SGL_NO_SWIFT").as_deref() == Ok("1") {
        println!("cargo:warning=SGL_NO_SWIFT=1: building without the Secure Enclave identity key");
        return;
    }

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let arch = match env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
        Ok("aarch64") => "arm64",
        Ok("x86_64") => "x86_64",
        other => {
            println!("cargo:warning=unknown macOS arch {other:?}: building without the Secure Enclave key");
            return;
        }
    };
    let min_os = env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| "11.0".to_string()); // = Rust's aarch64-apple-darwin default
    let lib = out.join("libsglenclave.a");

    let status = Command::new("xcrun")
        .args(["swiftc", "-parse-as-library", "-emit-library", "-static", "-O"])
        .args(["-module-name", "sglenclave"])
        .args(["-target", &format!("{arch}-apple-macos{min_os}")])
        .arg("src/enclave/se.swift")
        .arg("-o")
        .arg(&lib)
        .status();
    match status {
        Ok(s) if s.success() => {}
        other => {
            // Never fail the whole node build over this optional key: warn and use the stub.
            println!("cargo:warning=swiftc failed ({other:?}): building without the Secure Enclave key");
            return;
        }
    }

    // link-arg (not rustc-link-lib): Cargo hands rustc-link-lib only to the LIBRARY target when
    // a package has one, and the enclave code lives in the `sgl` binary (src/se_identity.rs), so
    // the bin failed to link with undefined _sgl_se_* symbols. link-arg reaches bins and tests.
    println!("cargo:rustc-link-arg={}", lib.display());
    for fw in ["CryptoKit", "Foundation", "Security"] {
        println!("cargo:rustc-link-arg=-framework");
        println!("cargo:rustc-link-arg={fw}");
    }
    // Swift runtime: in the OS on macOS 12+ (/usr/lib/swift); the toolchain dir provides the
    // swiftCompatibility shims the static library may reference.
    println!("cargo:rustc-link-arg=-L/usr/lib/swift");
    if let Ok(o) = Command::new("xcrun").args(["--find", "swiftc"]).output() {
        if let Ok(p) = String::from_utf8(o.stdout) {
            let swiftc = PathBuf::from(p.trim());
            if let Some(toolchain) = swiftc.parent().and_then(|b| b.parent()) {
                let dir = toolchain.join("lib/swift/macosx");
                println!("cargo:rustc-link-arg=-L{}", dir.display());
            }
        }
    }
    if let Ok(o) = Command::new("xcrun").args(["--show-sdk-path"]).output() {
        if let Ok(p) = String::from_utf8(o.stdout) {
            println!("cargo:rustc-link-arg=-L{}/usr/lib/swift", p.trim());
        }
    }
    println!("cargo:rustc-cfg=sgl_se");
}
