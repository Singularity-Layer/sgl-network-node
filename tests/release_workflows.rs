use std::fs;
use std::path::PathBuf;

fn workflow(name: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(root.join(".github/workflows").join(name))
        .unwrap_or_else(|error| panic!("read {name}: {error}"))
}

fn repository_file(path: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(root.join(path)).unwrap_or_else(|error| panic!("read {path}: {error}"))
}

#[test]
fn windows_probe_and_release_use_the_same_cpu_runtime_profile() {
    let windows = workflow("windows.yml");
    let release = workflow("release.yml");
    let required = [
        "RUSTFLAGS: -C target-feature=+crt-static",
        "CMAKE_MSVC_RUNTIME_LIBRARY: MultiThreaded",
        "CMAKE_POLICY_DEFAULT_CMP0091: NEW",
        "CFLAGS: /MT",
        "CXXFLAGS: /MT",
    ];

    for (name, contents) in [("windows.yml", &windows), ("release.yml", &release)] {
        for setting in required {
            assert_eq!(
                contents
                    .lines()
                    .filter(|line| line.trim() == setting)
                    .count(),
                1,
                "{name} must define exactly one shared Windows setting: {setting}"
            );
        }
        assert!(
            !contents.contains("+avx2") && !contents.contains("/arch:AVX2"),
            "{name} must not build a probe for a stronger CPU than the release binary"
        );
    }

    let binary =
        "cargo build --release --target x86_64-pc-windows-msvc --features inprocess,vision";
    let probe = format!("{binary} --example tool_probe");
    for (name, contents) in [("windows.yml", &windows), ("release.yml", &release)] {
        assert_eq!(contents.matches(binary).count(), 2, "{name}");
        assert_eq!(contents.matches(&probe).count(), 1, "{name}");
        assert_eq!(
            contents
                .matches("./.github/scripts/run-windows-inference-probe.ps1")
                .count(),
            1,
            "{name} must run the shared proof exactly once"
        );
    }
}

#[test]
fn windows_probe_model_is_immutable_and_verified_before_native_load() {
    let script = repository_file(".github/scripts/run-windows-inference-probe.ps1");
    assert!(!script.contains("resolve/main"));
    for required in [
        "067b946cf014b7c697f3654f621d577a3e3afd1c",
        "6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83",
        "$modelSize = 807694464",
        "Get-FileHash -Algorithm SHA256",
        "if ($actualSha256 -ne $modelSha256)",
        "if ($actualSize -ne $modelSize)",
    ] {
        assert!(
            script.contains(required),
            "missing model integrity gate: {required}"
        );
    }
    let verify = script.find("Get-FileHash -Algorithm SHA256").unwrap();
    let invoke = script.find("$out = & $probe model.gguf").unwrap();
    assert!(
        verify < invoke,
        "model hash must be checked before native loading"
    );
}
