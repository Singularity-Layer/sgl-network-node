/// Prevent debuggers from attaching to this process.
/// Uses PT_DENY_ATTACH on macOS, prctl on Linux.
pub fn deny_debugger_attach() {
    #[cfg(target_os = "macos")]
    {
        const PT_DENY_ATTACH: libc::c_int = 31;
        let result =
            unsafe { libc::ptrace(PT_DENY_ATTACH, 0, std::ptr::null_mut::<libc::c_char>(), 0) };
        if result == 0 {
            tracing::debug!("Runtime hardening: PT_DENY_ATTACH enabled");
        } else {
            tracing::warn!(
                "Runtime hardening: PT_DENY_ATTACH failed (errno: {})",
                std::io::Error::last_os_error()
            );
        }
    }

    #[cfg(target_os = "linux")]
    {
        let result = unsafe { libc::prctl(libc::PR_SET_DUMPABLE, 0) };
        if result == 0 {
            tracing::debug!("Runtime hardening: PR_SET_DUMPABLE(0) enabled");
        } else {
            tracing::warn!("Runtime hardening: PR_SET_DUMPABLE failed");
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        tracing::warn!("Runtime hardening: no debugger protection available on this platform");
    }
}
