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

#[cfg_attr(not(any(target_os = "macos", target_os = "linux")), allow(dead_code))]
const SERVICE_LABEL: &str = "cc.x402compute.sglnode";

/// Options captured from the CLI and embedded into the generated service.
// On platforms without a service installer (Windows), the fields are constructed
// but never read — that's the documented stub path, not a bug.
#[cfg_attr(not(any(target_os = "macos", target_os = "linux")), allow(dead_code))]
pub struct ServiceStartOptions {
    pub model_path: Option<String>,
    pub model_name: Option<String>,
    pub orchestrator_url: String,
    pub resource_percent: u8,
    pub inference_port: u16,
    pub max_jobs: u32,
    pub context_size: u32,
    pub heartbeat_interval: u64,
    pub enable_streaming: bool,
    /// macOS: wrap the node in a Seatbelt sandbox (opt-in). Ignored on Linux,
    /// where equivalent systemd hardening is always applied.
    pub sandbox: bool,
}

impl ServiceStartOptions {
    /// Build the `sgl start ...` argument vector (without the binary itself).
    // Consumed only by the macOS/Linux installers; on other platforms (Windows)
    // service install is a stub, so this is dead code there.
    #[cfg_attr(not(any(target_os = "macos", target_os = "linux")), allow(dead_code))]
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
        args.push("--context-size".into());
        args.push(self.context_size.to_string());
        args.push("--heartbeat-interval".into());
        args.push(self.heartbeat_interval.to_string());
        if self.enable_streaming {
            args.push("--enable-streaming".into());
        }
        args
    }
}

#[cfg_attr(not(any(target_os = "macos", target_os = "linux")), allow(dead_code))]
fn current_exe() -> Result<String, String> {
    std::env::current_exe()
        .map_err(|e| format!("Cannot resolve current executable path: {e}"))?
        .to_str()
        .map(|s| s.to_string())
        .ok_or_else(|| "Executable path is not valid UTF-8".to_string())
}

#[cfg_attr(not(any(target_os = "macos", target_os = "linux")), allow(dead_code))]
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
    #[cfg(target_os = "windows")]
    {
        install_windows(opts)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        let _ = opts;
        Err("No service installer for this platform. Run `sgl start ...` in the foreground.".to_string())
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
    #[cfg(target_os = "windows")]
    {
        uninstall_windows()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        Err("No service installer for this platform (nothing to uninstall).".to_string())
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
    #[cfg(target_os = "windows")]
    {
        status_windows()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        Err("No service installer for this platform.".to_string())
    }
}

// ─── Windows (Task Scheduler) ───────────────────────────────────────────────
// The launchd-parity story on Windows: a per-user Scheduled Task (no admin) that
//   * starts the node at logon                      (≈ LaunchAgent RunAtLoad)
//   * restarts it if it crashes                     (≈ launchd KeepAlive)
//   * keeps running after the desktop app closes    (independent process)
//   * runs hidden via the S4U logon type            (no console window)
// S4U ("service for user") runs without a stored password and non-interactively.
// If task registration under S4U is denied (some locked-down machines), we fall
// back to an Interactive-logon task — same lifecycle, may briefly show a window
// at logon. All PowerShell is passed as ONE argv element (no cmd shell parsing).

/// PowerShell single-quoted literal: escape embedded quotes by doubling them.
#[cfg(target_os = "windows")]
fn ps_squote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

#[cfg(target_os = "windows")]
fn run_powershell(script: &str) -> Result<String, String> {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let out = std::process::Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", script])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .map_err(|e| format!("Failed to run PowerShell: {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
    if out.status.success() {
        Ok(stdout)
    } else {
        Err(if stderr.is_empty() { stdout } else { stderr })
    }
}

/// Quote ONE token for a Windows command line (CreateProcess rules): always wrap
/// in double quotes, escape embedded `"` as `\"`. Our tokens are file paths and
/// flag values — no trailing-backslash-before-quote cases (paths end in `.gguf`).
#[cfg(target_os = "windows")]
fn win_quote(token: &str) -> String {
    format!("\"{}\"", token.replace('"', "\\\""))
}

#[cfg(target_os = "windows")]
fn install_windows(opts: &ServiceStartOptions) -> Result<(), String> {
    let exe = current_exe()?;
    // Quote EVERY token (not just spaced ones) with CreateProcess escaping.
    let arg_string = opts
        .start_args()
        .iter()
        .map(|a| win_quote(a))
        .collect::<Vec<_>>()
        .join(" ");

    // Interactive-fallback action: S4U runs windowless by design, but when Windows
    // refuses S4U (seen live on a tester's Win10) the Interactive task showed a
    // PERSISTENT sgl console. Wrap that path in a hidden PowerShell launcher that
    // waits for the node and propagates its exit code — restart-on-failure still
    // works, no window (PS itself is spawned hidden by -WindowStyle Hidden).
    // Tokens are PS-single-quoted: PowerShell rejoins its post--Command argv with
    // spaces and re-parses, so single-quote grouping survives CreateProcess splitting.
    let ps_tokens = opts
        .start_args()
        .iter()
        .map(|a| format!("'{}'", a.replace('\'', "''")))
        .collect::<Vec<_>>()
        .join(" ");
    let hidden_fallback_arg = format!(
        "-NoProfile -WindowStyle Hidden -Command & '{}' {}; exit $LASTEXITCODE",
        exe.replace('\'', "''"),
        ps_tokens
    );

    // Reinstall-safety: stop the old task instance and tree-kill any running node
    // BEFORE registering, so `MultipleInstances IgnoreNew` can't leave a stale node
    // serving the OLD model/args after Start-ScheduledTask.
    let _ = run_powershell(&format!(
        "Stop-ScheduledTask -TaskName {label} -ErrorAction SilentlyContinue",
        label = ps_squote(SERVICE_LABEL),
    ));
    kill_node_trees();

    // $ErrorActionPreference='Stop' makes non-terminating cmdlet errors fatal so a
    // failed Register/Start can't exit 0 and report a phantom success.
    let script = format!(
        "$ErrorActionPreference = 'Stop'; \
         $u = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name; \
         $a = New-ScheduledTaskAction -Execute {exe} -Argument {args}; \
         $t = New-ScheduledTaskTrigger -AtLogOn -User $u; \
         $s = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries \
              -Hidden -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 10 \
              -RestartInterval (New-TimeSpan -Minutes 1) -MultipleInstances IgnoreNew -StartWhenAvailable; \
         try {{ \
           $p = New-ScheduledTaskPrincipal -UserId $u -LogonType S4U -RunLevel Limited; \
           Register-ScheduledTask -TaskName {label} -Action $a -Trigger $t -Settings $s -Principal $p -Force -ErrorAction Stop | Out-Null; \
           Start-ScheduledTask -TaskName {label} -ErrorAction Stop \
         }} catch {{ \
           $fa = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument {fbarg}; \
           $p = New-ScheduledTaskPrincipal -UserId $u -LogonType Interactive -RunLevel Limited; \
           Register-ScheduledTask -TaskName {label} -Action $fa -Trigger $t -Settings $s -Principal $p -Force -ErrorAction Stop | Out-Null; \
           Start-ScheduledTask -TaskName {label} -ErrorAction Stop \
         }}",
        exe = ps_squote(&exe),
        args = ps_squote(&arg_string),
        fbarg = ps_squote(&hidden_fallback_arg),
        label = ps_squote(SERVICE_LABEL),
    );
    run_powershell(&script).map_err(|e| format!("Couldn't install the background task: {e}"))?;
    println!("Background task installed and started (Task Scheduler: {SERVICE_LABEL}).");
    println!("The node keeps serving after the app closes and restarts at logon.");
    Ok(())
}

/// Tree-kill every running `sgl start` node (command-line matched, so login/setup/
/// update invocations are never touched). taskkill /T takes the llama-server.exe
/// child down with the node — Stop-ScheduledTask alone can orphan it.
#[cfg(target_os = "windows")]
fn kill_node_trees() {
    let _ = run_powershell(
        "Get-CimInstance Win32_Process -Filter \"Name='sgl.exe'\" | \
         Where-Object { $_.CommandLine -match '(\\s|\")start(\\s|\"|$)' } | \
         ForEach-Object { & taskkill /T /F /PID $_.ProcessId 2>$null } | Out-Null",
    );
}

#[cfg(target_os = "windows")]
fn uninstall_windows() -> Result<(), String> {
    let script = format!(
        "Stop-ScheduledTask -TaskName {label} -ErrorAction SilentlyContinue; \
         Unregister-ScheduledTask -TaskName {label} -Confirm:$false -ErrorAction SilentlyContinue",
        label = ps_squote(SERVICE_LABEL),
    );
    let _ = run_powershell(&script);
    kill_node_trees();
    println!("Background task removed (and any running node stopped).");
    Ok(())
}

#[cfg(target_os = "windows")]
fn status_windows() -> Result<(), String> {
    let script = format!(
        "(Get-ScheduledTask -TaskName {label} -ErrorAction Stop).State",
        label = ps_squote(SERVICE_LABEL),
    );
    match run_powershell(&script) {
        Ok(state) => {
            println!("Background task: {state}");
            Ok(())
        }
        Err(_) => {
            println!("Background task: not installed");
            Ok(())
        }
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

// Targeted-deny Seatbelt profile. We allow everything the inference engine
// legitimately needs (Metal/GPU, Secure Enclave attestation, model file reads,
// outbound network) and only deny reads/writes of the operator's most sensitive
// data — so a compromised llama.cpp (the one place attacker-controlled prompt
// bytes hit native code) cannot exfiltrate SSH keys, wallets, or browser data.
// "Allow default, deny secrets" (rather than "deny default, allow list") is the
// safe choice for an unattended GPU process we can't pre-test on every machine.
#[cfg(target_os = "macos")]
fn write_sandbox_profile() -> Result<String, String> {
    let home = dirs::home_dir().ok_or("Cannot resolve home directory")?;
    let home_str = home.to_str().ok_or("home directory path not UTF-8")?;
    let dir = home.join("Library/Application Support/sgl-node");
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("Failed to create sandbox profile dir: {e}"))?;
    let profile = dir.join("sandbox.sb");

    let body = format!(
        r#"(version 1)
(allow default)
;; Wall off the operator's secrets from the inference process.
(deny file-read* file-write*
    (subpath "{home}/.ssh")
    (subpath "{home}/.gnupg")
    (subpath "{home}/.aws")
    (subpath "{home}/.config/solana")
    (subpath "{home}/.config/gcloud")
    (subpath "{home}/Library/Keychains")
    (subpath "{home}/Library/Cookies")
    (subpath "{home}/Library/Application Support/Google/Chrome")
    (subpath "{home}/Library/Application Support/Firefox")
    (subpath "{home}/Library/Application Support/BraveSoftware")
    (subpath "{home}/Library/Application Support/Exodus")
    (subpath "{home}/Library/Application Support/Electrum")
    (literal "{home}/.zsh_history")
    (literal "{home}/.bash_history"))
"#,
        home = home_str,
    );
    std::fs::write(&profile, body)
        .map_err(|e| format!("Failed to write sandbox profile: {e}"))?;
    profile
        .to_str()
        .map(|s| s.to_string())
        .ok_or_else(|| "sandbox profile path not UTF-8".to_string())
}

/// `launchctl bootout` is ASYNCHRONOUS: it returns before the agent — and its
/// heavyweight `llama-server` child holding several GB of model in RAM — has
/// actually exited and been released from the domain. Bootstrapping the same
/// `Label` while the old instance is still draining makes launchd fail with
/// "Bootstrap failed: 5: Input/output error" (seen live when lowering the
/// context window forces a service reinstall). So: bootout, then poll the domain
/// until the label is gone (or a timeout), before the caller bootstraps.
#[cfg(target_os = "macos")]
fn bootout_and_wait(target: &str) {
    let _ = run("launchctl", &["bootout", target]);
    // `launchctl print <target>` succeeds while the service is still loaded and
    // fails once launchd has released it. Poll ~12s; the node normally drains in
    // 1–3s, so this returns early in the common case.
    for _ in 0..60 {
        if run("launchctl", &["print", target]).is_err() {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}

#[cfg(target_os = "macos")]
fn install_macos(opts: &ServiceStartOptions) -> Result<(), String> {
    let exe = current_exe()?;
    let log = log_path()?;
    let plist = plist_path()?;

    // ProgramArguments: caffeinate -i [sandbox-exec -f <profile>] <exe> start <args...>
    // caffeinate -i blocks idle sleep so the node stays on the grid; if the
    // node exits, launchd (KeepAlive) restarts the whole thing. When --sandbox
    // is set, the node (and its llama-server child) run under a Seatbelt profile.
    let mut program_args: Vec<String> = vec![
        "/usr/bin/caffeinate".to_string(),
        "-i".to_string(),
    ];
    if opts.sandbox {
        let profile = write_sandbox_profile()?;
        program_args.push("/usr/bin/sandbox-exec".to_string());
        program_args.push("-f".to_string());
        program_args.push(profile);
    }
    program_args.push(exe.clone());
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

    // Reload cleanly. bootout is async and the old node's llama-server child can
    // take a few seconds to die, so wait for the label to drain, then bootstrap —
    // and retry the whole dance if launchd still reports the label busy
    // ("Bootstrap failed: 5: Input/output error").
    bootout_and_wait(&target);
    let mut bootstrap_err = String::new();
    let mut bootstrapped = false;
    for attempt in 0..5u64 {
        match run("launchctl", &["bootstrap", &domain, plist_str]) {
            Ok(_) => {
                bootstrapped = true;
                break;
            }
            Err(e) => {
                bootstrap_err = e;
                // The previous instance is still draining — force another teardown,
                // back off, and retry.
                bootout_and_wait(&target);
                std::thread::sleep(std::time::Duration::from_millis(300 * (attempt + 1)));
            }
        }
    }
    if !bootstrapped {
        return Err(format!(
            "launchctl bootstrap failed after retries: {bootstrap_err} \
             (the previous node may still be shutting down — wait a few seconds and try again)"
        ));
    }
    let _ = run("launchctl", &["enable", &target]);
    let _ = run("launchctl", &["kickstart", "-k", &target]);

    println!("✅ SGL node service installed (launchd: {SERVICE_LABEL})");
    println!("   Plist:   {}", plist.display());
    println!("   Logs:    {log}");
    if opts.sandbox {
        println!("   Sandbox: ON (Seatbelt) — SSH keys, wallets, keychains, and");
        println!("            browser data are walled off from the inference process.");
    } else {
        println!("   Sandbox: off — pass `--sandbox` to wall off keys/wallets from");
        println!("            the inference process (recommended; test on your setup).");
    }
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

    // Re-expose the operator's chosen model read-only (ProtectHome=true would
    // otherwise hide it if it lives under $HOME). "-" tolerates a missing path.
    let model_ro = opts
        .model_path
        .as_ref()
        .map(|m| format!("ReadOnlyPaths=-{m}\n"))
        .unwrap_or_default();

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

# ── sandbox hardening ──────────────────────────────────────────────────────
# Contains the blast radius if the native inference engine (llama.cpp) is ever
# exploited via a crafted prompt: the process can still read its model and reach
# the network, but cannot touch the operator's home (SSH keys, wallets, etc.),
# gain privileges, or write outside the node's own state dirs.
# GPU-safe by design: devices stay accessible (no PrivateDevices), no W^X
# (no MemoryDenyWriteExecute) that would break CUDA/ROCm, and denied syscalls
# return EPERM instead of killing the process (no SIGSYS surprises).
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=-%h/.config/sgl-node -%h/.local/share/sgl-node
{model_ro}ProtectKernelTunables=true
ProtectKernelModules=true
ProtectKernelLogs=true
ProtectControlGroups=true
ProtectClock=true
ProtectHostname=true
RestrictRealtime=true
RestrictSUIDSGID=true
RestrictNamespaces=true
LockPersonality=true
RemoveIPC=true
CapabilityBoundingSet=
AmbientCapabilities=
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6 AF_NETLINK
SystemCallArchitectures=native
SystemCallErrorNumber=EPERM
SystemCallFilter=@system-service
SystemCallFilter=~@privileged @obsolete @mount @reboot @swap @raw-io @cpu-emulation

[Install]
WantedBy=default.target
"#,
        exec_start = exec_start,
        model_ro = model_ro,
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
    println!("   Sandbox: ON — systemd hardening confines the inference process");
    println!("            (home/keys/wallets protected; GPU + network preserved).");
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
