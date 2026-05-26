use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct TeeCapabilities {
    pub tee_type: String,
    pub secure_enclave_available: bool,
    pub chip: String,
    pub cpu_cores: u32,
    pub memory_gb: f64,
    pub gpu: String,
    pub metal_support: bool,
}

pub fn detect() -> TeeCapabilities {
    let cpu_cores = std::thread::available_parallelism()
        .map(|p| p.get() as u32)
        .unwrap_or(1);

    let memory_gb = detect_memory_gb();
    let chip = detect_chip_name();
    let secure_enclave = detect_secure_enclave();
    let metal = detect_metal();

    TeeCapabilities {
        tee_type: if secure_enclave { "apple_se".to_string() } else { "none".to_string() },
        secure_enclave_available: secure_enclave,
        chip,
        cpu_cores,
        memory_gb,
        gpu: "apple_metal".to_string(),
        metal_support: metal,
    }
}

fn detect_memory_gb() -> f64 {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output();

    match output {
        Ok(out) => {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            s.parse::<u64>().map(|b| b as f64 / (1024.0 * 1024.0 * 1024.0)).unwrap_or(16.0)
        }
        Err(_) => 16.0,
    }
}

fn detect_chip_name() -> String {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output();

    match output {
        Ok(out) => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        Err(_) => "Apple Silicon".to_string(),
    }
}

fn detect_secure_enclave() -> bool {
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("ioreg")
            .args(["-l", "-p", "IODeviceTree"])
            .output();

        if let Ok(out) = output {
            let text = String::from_utf8_lossy(&out.stdout);
            return text.contains("AppleSEP") || text.contains("sep");
        }
    }
    false
}

fn detect_metal() -> bool {
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("system_profiler")
            .args(["SPDisplaysDataType"])
            .output();

        if let Ok(out) = output {
            let text = String::from_utf8_lossy(&out.stdout);
            return text.contains("Metal");
        }
    }
    false
}

pub fn print_capabilities(caps: &TeeCapabilities) {
    println!("=== Hardware Capabilities ===");
    println!("Chip:             {}", caps.chip);
    println!("CPU cores:        {}", caps.cpu_cores);
    println!("Memory:           {:.1} GB", caps.memory_gb);
    println!("GPU:              {}", caps.gpu);
    println!("Metal:            {}", if caps.metal_support { "Yes" } else { "No" });
    println!("Secure Enclave:   {}", if caps.secure_enclave_available { "Available" } else { "Not detected" });
    println!("TEE type:         {}", caps.tee_type);
}
