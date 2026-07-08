use serde::Serialize;
use sha2::{Digest, Sha256};

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

#[derive(Debug, Clone, Serialize)]
pub struct HardwareAttestationReport {
    pub tee_type: String,
    pub chip_model: String,
    pub cpu_cores: u32,
    pub memory_gb: f64,
    pub secure_enclave: bool,
    pub sip_enabled: bool,
    pub os_version: String,
    pub kernel_version: String,
    pub boot_uuid: String,
    pub hardware_uuid: String,
    pub firmware_version: String,
    pub serial_hash: String,
    /// sha256 of the running sgl binary — lets the orchestrator gate on a known,
    /// hardened build (allowlist) so a tampered binary can't serve.
    pub binary_hash: String,
    pub report_hash: String,
}

impl HardwareAttestationReport {
    pub fn compute_hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.tee_type.as_bytes());
        hasher.update(self.chip_model.as_bytes());
        hasher.update(self.cpu_cores.to_le_bytes());
        hasher.update(self.hardware_uuid.as_bytes());
        hasher.update(self.firmware_version.as_bytes());
        hasher.update(self.boot_uuid.as_bytes());
        hasher.update(self.kernel_version.as_bytes());
        hasher.update(if self.secure_enclave { &[1u8] } else { &[0u8] });
        hasher.update(if self.sip_enabled { &[1u8] } else { &[0u8] });
        hasher.update(self.binary_hash.as_bytes());
        hex::encode(hasher.finalize())
    }
}

pub fn detect() -> TeeCapabilities {
    let cpu_cores = std::thread::available_parallelism()
        .map(|p| p.get() as u32)
        .unwrap_or(1);

    let memory_gb = detect_memory_gb();
    let chip = detect_chip_name();
    let secure_enclave = detect_secure_enclave();
    let metal = detect_metal();
    let gpu = detect_gpu_name();

    TeeCapabilities {
        tee_type: if secure_enclave {
            "apple_se".to_string()
        } else {
            "none".to_string()
        },
        secure_enclave_available: secure_enclave,
        chip,
        cpu_cores,
        memory_gb,
        gpu,
        metal_support: metal,
    }
}

pub fn generate_attestation_report() -> HardwareAttestationReport {
    let caps = detect();
    let sip = detect_sip_status();
    let os_version = detect_os_version();
    let kernel_version = run_cmd("uname", &["-r"]);
    let boot_uuid = detect_boot_uuid();
    let hw_uuid = detect_hardware_uuid();
    let firmware = detect_firmware_version();
    let serial_hash = detect_serial_hash();
    let binary_hash = detect_binary_hash();

    let mut report = HardwareAttestationReport {
        tee_type: caps.tee_type,
        chip_model: caps.chip,
        cpu_cores: caps.cpu_cores,
        memory_gb: caps.memory_gb,
        secure_enclave: caps.secure_enclave_available,
        sip_enabled: sip,
        os_version,
        kernel_version,
        boot_uuid,
        hardware_uuid: hw_uuid,
        firmware_version: firmware,
        serial_hash,
        binary_hash,
        report_hash: String::new(),
    };
    report.report_hash = report.compute_hash();
    report
}

/// sha256 of the currently-running sgl binary. The orchestrator can require
/// this to be on an allowlist of known-hardened builds.
pub fn detect_binary_hash() -> String {
    match std::env::current_exe()
        .ok()
        .and_then(|p| std::fs::read(p).ok())
    {
        Some(bytes) => {
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            hex::encode(hasher.finalize())
        }
        None => String::new(),
    }
}

fn detect_memory_gb() -> f64 {
    #[cfg(target_os = "macos")]
    {
        let s = run_cmd("sysctl", &["-n", "hw.memsize"]);
        return s.parse::<u64>()
            .map(|b| b as f64 / (1024.0 * 1024.0 * 1024.0))
            .unwrap_or(16.0);
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    let kb = rest.split_whitespace().next()
                        .and_then(|v| v.parse::<u64>().ok())
                        .unwrap_or(0);
                    if kb > 0 {
                        return kb as f64 / (1024.0 * 1024.0);
                    }
                }
            }
        }
        16.0
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        16.0
    }
}

fn detect_chip_name() -> String {
    #[cfg(target_os = "macos")]
    {
        let s = run_cmd("sysctl", &["-n", "machdep.cpu.brand_string"]);
        return if s.is_empty() { "Apple Silicon".to_string() } else { s };
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            for key in ["model name", "Hardware", "Processor"] {
                for line in cpuinfo.lines() {
                    if let Some((k, v)) = line.split_once(':') {
                        if k.trim() == key {
                            let value = v.trim();
                            if !value.is_empty() {
                                return value.to_string();
                            }
                        }
                    }
                }
            }
        }
        return run_cmd("uname", &["-m"]);
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        "unknown".to_string()
    }
}

fn detect_gpu_name() -> String {
    #[cfg(target_os = "macos")]
    {
        return "apple_metal".to_string();
    }
    #[cfg(target_os = "linux")]
    {
        let nvidia = run_cmd("nvidia-smi", &["--query-gpu=name", "--format=csv,noheader"]);
        if let Some(first) = nvidia.lines().map(str::trim).find(|line| !line.is_empty()) {
            return first.to_string();
        }
        let lspci = run_cmd("lspci", &[]);
        for line in lspci.lines() {
            let lower = line.to_ascii_lowercase();
            if lower.contains("vga compatible controller")
                || lower.contains("3d controller")
                || lower.contains("display controller")
            {
                if let Some((_, value)) = line.split_once(": ") {
                    return value.trim().to_string();
                }
                return line.trim().to_string();
            }
        }
        return "unknown".to_string();
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        "unknown".to_string()
    }
}

fn detect_secure_enclave() -> bool {
    #[cfg(target_os = "macos")]
    {
        let text = run_cmd("ioreg", &["-l", "-p", "IODeviceTree"]);
        text.contains("AppleSEP") || text.contains("sep")
    }
    #[cfg(not(target_os = "macos"))]
    false
}

fn detect_metal() -> bool {
    #[cfg(target_os = "macos")]
    {
        let text = run_cmd("system_profiler", &["SPDisplaysDataType"]);
        text.contains("Metal")
    }
    #[cfg(not(target_os = "macos"))]
    false
}

fn detect_sip_status() -> bool {
    #[cfg(target_os = "macos")]
    {
        let text = run_cmd("csrutil", &["status"]);
        text.contains("enabled")
    }
    #[cfg(not(target_os = "macos"))]
    false
}

fn detect_os_version() -> String {
    #[cfg(target_os = "macos")]
    {
        return run_cmd("sw_vers", &["-productVersion"]);
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(os_release) = std::fs::read_to_string("/etc/os-release") {
            for key in ["PRETTY_NAME", "VERSION_ID", "NAME"] {
                for line in os_release.lines() {
                    if let Some((k, v)) = line.split_once('=') {
                        if k == key {
                            let value = v.trim().trim_matches('"');
                            if !value.is_empty() {
                                return value.to_string();
                            }
                        }
                    }
                }
            }
        }
        return run_cmd("uname", &["-sr"]);
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        String::new()
    }
}

fn detect_boot_uuid() -> String {
    #[cfg(target_os = "macos")]
    {
        run_cmd("sysctl", &["-n", "kern.bootsessionuuid"])
    }
    #[cfg(not(target_os = "macos"))]
    String::new()
}

fn detect_hardware_uuid() -> String {
    #[cfg(target_os = "macos")]
    {
        let text = run_cmd("system_profiler", &["SPHardwareDataType"]);
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("Hardware UUID:") || trimmed.starts_with("Provisioning UDID:") {
                if let Some(val) = trimmed.split(':').nth(1) {
                    return val.trim().to_string();
                }
            }
        }
        String::new()
    }
    #[cfg(not(target_os = "macos"))]
    {
        read_first_existing(&[
            "/sys/class/dmi/id/product_uuid",
            "/sys/devices/virtual/dmi/id/product_uuid",
            "/etc/machine-id",
        ])
    }
}

fn detect_firmware_version() -> String {
    #[cfg(target_os = "macos")]
    {
        let text = run_cmd("system_profiler", &["SPHardwareDataType"]);
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("System Firmware Version:")
                || trimmed.starts_with("OS Loader Version:")
            {
                if let Some(val) = trimmed.split(':').nth(1) {
                    return val.trim().to_string();
                }
            }
        }
        run_cmd("sysctl", &["-n", "kern.osversion"])
    }
    #[cfg(not(target_os = "macos"))]
    {
        read_first_existing(&[
            "/sys/class/dmi/id/bios_version",
            "/sys/devices/virtual/dmi/id/bios_version",
        ])
    }
}

fn detect_serial_hash() -> String {
    #[cfg(target_os = "macos")]
    {
        let text = run_cmd("system_profiler", &["SPHardwareDataType"]);
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("Serial Number") {
                if let Some(val) = trimmed.split(':').nth(1) {
                    let serial = val.trim();
                    let mut hasher = Sha256::new();
                    hasher.update(serial.as_bytes());
                    return hex::encode(hasher.finalize());
                }
            }
        }
        String::new()
    }
    #[cfg(not(target_os = "macos"))]
    {
        let serial = read_first_existing(&[
            "/sys/class/dmi/id/product_serial",
            "/sys/devices/virtual/dmi/id/product_serial",
            "/sys/class/dmi/id/board_serial",
            "/sys/devices/virtual/dmi/id/board_serial",
        ]);
        if serial.is_empty() {
            String::new()
        } else {
            let mut hasher = Sha256::new();
            hasher.update(serial.as_bytes());
            hex::encode(hasher.finalize())
        }
    }
}

fn run_cmd(cmd: &str, args: &[&str]) -> String {
    match std::process::Command::new(cmd).args(args).output() {
        Ok(out) => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        Err(_) => String::new(),
    }
}

#[cfg(not(target_os = "macos"))]
fn read_first_existing(paths: &[&str]) -> String {
    for path in paths {
        if let Ok(value) = std::fs::read_to_string(path) {
            let trimmed = value.trim();
            if !trimmed.is_empty() && trimmed != "None" && trimmed != "Not Specified" {
                return trimmed.to_string();
            }
        }
    }
    String::new()
}

pub fn print_capabilities(caps: &TeeCapabilities) {
    println!("=== Hardware Capabilities ===");
    println!("Chip:             {}", caps.chip);
    println!("CPU cores:        {}", caps.cpu_cores);
    println!("Memory:           {:.1} GB", caps.memory_gb);
    println!("GPU:              {}", caps.gpu);
    println!(
        "Metal:            {}",
        if caps.metal_support { "Yes" } else { "No" }
    );
    println!(
        "Secure Enclave:   {}",
        if caps.secure_enclave_available {
            "Available"
        } else {
            "Not detected"
        }
    );
    println!("TEE type:         {}", caps.tee_type);
}
