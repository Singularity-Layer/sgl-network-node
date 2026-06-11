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
        gpu: "apple_metal".to_string(),
        metal_support: metal,
    }
}

pub fn generate_attestation_report() -> HardwareAttestationReport {
    let caps = detect();
    let sip = detect_sip_status();
    let os_version = run_cmd("sw_vers", &["-productVersion"]);
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
    let s = run_cmd("sysctl", &["-n", "hw.memsize"]);
    s.parse::<u64>()
        .map(|b| b as f64 / (1024.0 * 1024.0 * 1024.0))
        .unwrap_or(16.0)
}

fn detect_chip_name() -> String {
    let s = run_cmd("sysctl", &["-n", "machdep.cpu.brand_string"]);
    if s.is_empty() {
        "Apple Silicon".to_string()
    } else {
        s
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
    String::new()
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
    String::new()
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
    String::new()
}

fn run_cmd(cmd: &str, args: &[&str]) -> String {
    match std::process::Command::new(cmd).args(args).output() {
        Ok(out) => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        Err(_) => String::new(),
    }
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
