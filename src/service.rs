//! Background-service installer for the SGL node.
//!
//! Turns `sgl start ...` into a managed OS service so an operator's machine
//! keeps serving across reboots, logout, crashes, and (on macOS) idle sleep —
//! without the operator hand-writing a plist/unit.
//!
//!   macOS  → launchd LaunchAgent (~/Library/LaunchAgents), wraps start in
//!            `caffeinate -i` to block idle sleep while serving.
//!   Linux  → systemd --user unit (~/.config/systemd/user), Restart=always.
//!
//! The exact `start` flags the operator picks are baked into the service so
//! `sgl service install --model-path ... --resource-percent 50` reproduces
//! their chosen config every launch.

const SERVICE_LABEL: &str = "cc.x402compute.sglnode";

/// Options captured from the CLI and embedded into the generated service.
pub struct ServiceStartOptions {
    pub model_path: Option<String>,
    pub model_name: Option<String>,
    pub orchestrator_url: String,
    pub resource_percent: u8,
    pub inference_port: u16,
    pub max_jobs: u32,
    pub heartbeat_interval: u64,
    pub enable_streaming: bool,
}

impl ServiceStartOptions {
    /// Build the `sgl start ...` argument vector (without the binary itself).
    fn start_args(&self) -> Vec<String> {
        let mut args = vec!["start".to_string()];
        if let Some(mp) = &self.model_path {
            args.push("--model-path".into());
            args.push(mp.clone());
        }
        if let Some(mn) = &self.model_name {
            args.push("--model-name".into());
            args.push(mn.clone());
        }
        args.push("--orchestrator-url".into());
        args.push(self.orchestrator_url.clone());
        args.push("--resource-percent".into());
        args.push(self.resource_percent.to_string());
        args.push("--inference-port".into());
        args.push(self.inference_port.to_string());
        args.push("--max-jobs".into());
        args.push(self.max_jobs.to_string());
        args.push("--heartbeat-interval".into());
        args.push(self.heartbeat_interval.to_string());
        if self.enable_streaming {
            args.push("--enable-streaming".into());
        }
        args
    }
}

fn current_exe() -> Result<String, String> {
    std::env::current_exe()
        .map_err(|e| format!("Cannot resolve current executable path: {e}"))?
        .to_str()
        .map(|s| s.to_string())
        .ok_or_else(|| "Executable path is not valid UTF-8".to_string())
}

fn log_path() -> Result<String, String> {
    let home = dirs::home_dir().ok_or("Cannot resolve home directory")?;
    Ok(home
        .join("Library/Logs/sgl-node.log")
        .to_str()
        .unwrap_or("/tmp/sgl-node.log")
        .to_string())
}

pub fn install(opts: &ServiceStartOptions) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        install_macos(opts)
    }
    #[cfg(target_os = "linux")]
    {
        install_linux(opts)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = opts;
        Err(
            "Service install is only supported on macOS and Linux. Run `sgl start ...` manually."
                .to_string(),
        )
    }
}

pub fn uninstall() -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        uninstall_macos()
    }
    #[cfg(target_os = "linux")]
    {
        uninstall_linux()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        Err("Service uninstall is only supported on macOS and Linux.".to_string())
    }
}

pub fn status() -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        status_macos()
    }
    #[cfg(target_os = "linux")]
    {
        status_linux()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        Err("Service status is only supported on macOS and Linux.".to_string())
    }
}

// ─── macOS (launchd) ────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
fn plist_path() -> Result<std::path::PathBuf, String> {
    let home = dirs::home_dir().ok_or("Cannot resolve home directory")?;
    Ok(home
        .join("Library/LaunchAgents")
        .join(format!("{SERVICE_LABEL}.plist")))
}

#[cfg(target_os = "macos")]
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(target_os = "macos")]
fn install_macos(opts: &ServiceStartOptions) -> Result<(), String> {
    let exe = current_exe()?;
    let log = log_path()?;
    let plist = plist_path()?;

    // ProgramArguments: caffeinate -i <exe> start <args...>
    // caffeinate -i blocks idle sleep so the node stays on the grid; if the
    // node exits, launchd (KeepAlive) restarts the whole thing.
    let mut program_args: Vec<String> = vec![
        "/usr/bin/caffeinate".to_string(),
        "-i".to_string(),
        exe.clone(),
    ];
    program_args.extend(opts.start_args());

    let args_xml: String = program_args
        .iter()
        .map(|a| format!("        <string>{}</string>", xml_escape(a)))
        .collect::<Vec<_>>()
        .join("\n");

    let working_dir = dirs::home_dir()
        .map(|h| h.to_str().unwrap_or("/").to_string())
        .unwrap_or_else(|| "/".to_string());

    let plist_body = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
{args}
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>15</integer>
    <key>ProcessType</key>
    <string>Interactive</string>
    <key>StandardOutPath</key>
    <string>{log}</string>
    <key>StandardErrorPath</key>
    <string>{log}</string>
    <key>WorkingDirectory</key>
    <string>{wd}</string>
</dict>
</plist>
"#,
        label = SERVICE_LABEL,
        args = args_xml,
        log = xml_escape(&log),
        wd = xml_escape(&working_dir),
    );

    if let Some(parent) = plist.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create LaunchAgents dir: {e}"))?;
    }
    std::fs::write(&plist, plist_body).map_err(|e| format!("Failed to write plist: {e}"))?;

    let uid = unsafe { libc::getuid() };
    let domain = format!("gui/{uid}");
    let target = format!("{domain}/{SERVICE_LABEL}");
    let plist_str = plist.to_str().ok_or("plist path not UTF-8")?;

    // Reload cleanly: bootout (ignore failure if not loaded) then bootstrap.
    let _ = run("launchctl", &["bootout", &target]);
    run("launchctl", &["bootstrap", &domain, plist_str])
        .map_err(|e| format!("launchctl bootstrap failed: {e}"))?;
    let _ = run("launchctl", &["enable", &target]);
    let _ = run("launchctl", &["kickstart", "-k", &target]);

    println!("✅ SGL node service installed (launchd: {SERVICE_LABEL})");
    println!("   Plist:   {}", plist.display());
    println!("   Logs:    {log}");
    println!("   It starts at login, restarts on crash, and blocks idle sleep.");
    println!("   Manage:  sgl service status | sgl service uninstall");
    println!();
    println!("   Note: closing a MacBook lid still sleeps the machine. To serve");
    println!("   with the lid closed, keep it plugged in to an external display");
    println!("   or enable clamshell/keep-awake in system settings.");
    Ok(())
}

#[cfg(target_os = "macos")]
fn uninstall_macos() -> Result<(), String> {
    let plist = plist_path()?;
    let uid = unsafe { libc::getuid() };
    let target = format!("gui/{uid}/{SERVICE_LABEL}");
    let _ = run("launchctl", &["bootout", &target]);
    if plist.exists() {
        std::fs::remove_file(&plist).map_err(|e| format!("Failed to remove plist: {e}"))?;
    }
    println!("✅ SGL node service uninstalled.");
    Ok(())
}

#[cfg(target_os = "macos")]
fn status_macos() -> Result<(), String> {
    let plist = plist_path()?;
    if !plist.exists() {
        println!("SGL node service: NOT installed.");
        println!("Install with: sgl service install --model-path <gguf> --model-name <name>");
        return Ok(());
    }
    let uid = unsafe { libc::getuid() };
    let target = format!("gui/{uid}/{SERVICE_LABEL}");
    println!("SGL node service: installed ({})", plist.display());
    println!();
    match run("launchctl", &["print", &target]) {
        Ok(out) => {
            for line in out.lines() {
                let t = line.trim();
                if t.starts_with("state =")
                    || t.starts_with("pid =")
                    || t.starts_with("last exit code =")
                    || t.starts_with("runs =")
                {
                    println!("  {t}");
                }
            }
        }
        Err(_) => println!("  (service registered but not currently loaded)"),
    }
    Ok(())
}

// ─── Linux (systemd --user) ─────────────────────────────────────────────────

#[cfg(target_os = "linux")]
fn unit_path() -> Result<std::path::PathBuf, String> {
    let home = dirs::home_dir().ok_or("Cannot resolve home directory")?;
    Ok(home
        .join(".config/systemd/user")
        .join(format!("{SERVICE_LABEL}.service")))
}

#[cfg(target_os = "linux")]
fn install_linux(opts: &ServiceStartOptions) -> Result<(), String> {
    let exe = current_exe()?;
    let unit = unit_path()?;

    let exec_start = std::iter::once(exe.clone())
        .chain(opts.start_args())
        .map(|a| {
            if a.contains(' ') {
                format!("\"{a}\"")
            } else {
                a
            }
        })
        .collect::<Vec<_>>()
        .join(" ");

    let unit_body = format!(
        r#"[Unit]
Description=SGL Network compute node
After=network-online.target
Wants=network-online.target

[Service]
ExecStart={exec_start}
Restart=always
RestartSec=15
StandardOutput=append:%h/.local/share/sgl-node/sgl-node.log
StandardError=append:%h/.local/share/sgl-node/sgl-node.log

[Install]
WantedBy=default.target
"#,
        exec_start = exec_start,
    );

    if let Some(parent) = unit.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create systemd user dir: {e}"))?;
    }
    // Ensure log dir exists.
    if let Some(home) = dirs::home_dir() {
        let _ = std::fs::create_dir_all(home.join(".local/share/sgl-node"));
    }
    std::fs::write(&unit, unit_body).map_err(|e| format!("Failed to write unit: {e}"))?;

    run("systemctl", &["--user", "daemon-reload"])
        .map_err(|e| format!("systemctl daemon-reload failed: {e}"))?;
    run(
        "systemctl",
        &[
            "--user",
            "enable",
            "--now",
            &format!("{SERVICE_LABEL}.service"),
        ],
    )
    .map_err(|e| format!("systemctl enable --now failed: {e}"))?;

    println!("✅ SGL node service installed (systemd --user: {SERVICE_LABEL})");
    println!("   Unit:  {}", unit.display());
    println!("   Logs:  ~/.local/share/sgl-node/sgl-node.log");
    println!(
        "   Tip: run `loginctl enable-linger $USER` so it runs without an active login session."
    );
    println!("   Manage: sgl service status | sgl service uninstall");
    Ok(())
}

#[cfg(target_os = "linux")]
fn uninstall_linux() -> Result<(), String> {
    let unit = unit_path()?;
    let _ = run(
        "systemctl",
        &[
            "--user",
            "disable",
            "--now",
            &format!("{SERVICE_LABEL}.service"),
        ],
    );
    if unit.exists() {
        std::fs::remove_file(&unit).map_err(|e| format!("Failed to remove unit: {e}"))?;
    }
    let _ = run("systemctl", &["--user", "daemon-reload"]);
    println!("✅ SGL node service uninstalled.");
    Ok(())
}

#[cfg(target_os = "linux")]
fn status_linux() -> Result<(), String> {
    let unit = unit_path()?;
    if !unit.exists() {
        println!("SGL node service: NOT installed.");
        println!("Install with: sgl service install --model-path <gguf> --model-name <name>");
        return Ok(());
    }
    println!("SGL node service: installed ({})", unit.display());
    match run(
        "systemctl",
        &["--user", "is-active", &format!("{SERVICE_LABEL}.service")],
    ) {
        Ok(out) => println!("  state = {}", out.trim()),
        Err(e) => println!("  state = unknown ({e})"),
    }
    Ok(())
}

// ─── helper ─────────────────────────────────────────────────────────────────

#[allow(dead_code)]
fn run(cmd: &str, args: &[&str]) -> Result<String, String> {
    let out = std::process::Command::new(cmd)
        .args(args)
        .output()
        .map_err(|e| format!("failed to run {cmd}: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}
