//! Conservative host/cgroup memory observation for the supported F32 Qwen worker.
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Memory {
    pub available_before_load_bytes: u64,
    pub estimated_required_bytes: u64,
    pub budget_bytes: u64,
}
pub fn available() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let mut available = text
            .lines()
            .find(|l| l.starts_with("MemAvailable:"))?
            .split_whitespace()
            .nth(1)?
            .parse::<u64>()
            .ok()?
            .saturating_mul(1024);
        if let (Ok(limit), Ok(used)) = (
            std::fs::read_to_string("/sys/fs/cgroup/memory.max"),
            std::fs::read_to_string("/sys/fs/cgroup/memory.current"),
        ) {
            if let (Ok(limit), Ok(used)) = (limit.trim().parse::<u64>(), used.trim().parse::<u64>())
            {
                available = available.min(limit.saturating_sub(used));
            }
        }
        Some(available)
    }
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("/usr/bin/vm_stat")
            .output()
            .ok()?;
        let text = String::from_utf8(output.stdout).ok()?;
        let page = text
            .lines()
            .next()?
            .split("page size of ")
            .nth(1)?
            .split_whitespace()
            .next()?
            .parse::<u64>()
            .ok()?;
        let pages: u64 = text
            .lines()
            .filter(|l| {
                l.starts_with("Pages free:")
                    || l.starts_with("Pages inactive:")
                    || l.starts_with("Pages speculative:")
            })
            .filter_map(|l| {
                l.split(':')
                    .nth(1)?
                    .trim()
                    .trim_end_matches('.')
                    .parse::<u64>()
                    .ok()
            })
            .sum();
        // Unified memory is counted once. Compressed memory is not treated as free.
        Some(pages.saturating_mul(page))
    }
    #[cfg(target_os = "windows")]
    {
        #[repr(C)]
        struct MemoryStatus {
            length: u32,
            load: u32,
            total_phys: u64,
            avail_phys: u64,
            total_page: u64,
            avail_page: u64,
            total_virtual: u64,
            avail_virtual: u64,
            avail_extended: u64,
        }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GlobalMemoryStatusEx(status: *mut MemoryStatus) -> i32;
        }
        let mut status = MemoryStatus {
            length: std::mem::size_of::<MemoryStatus>() as u32,
            load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page: 0,
            avail_page: 0,
            total_virtual: 0,
            avail_virtual: 0,
            avail_extended: 0,
        };
        // SAFETY: valid writable MEMORYSTATUSEX layout with its required length initialized.
        if unsafe { GlobalMemoryStatusEx(&mut status) } != 0 {
            Some(status.avail_phys)
        } else {
            None
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        None
    }
}
pub fn estimate(file_bytes: u64, layers: usize) -> u64 {
    // Pinned BF16 checkpoint is expanded to F32. KV: K+V, 4096 positions,
    // two KV heads, head dimension 64, four bytes; allow 384 MiB workspace.
    file_bytes.saturating_mul(2) + (layers as u64) * 2 * 4096 * 2 * 64 * 4 + 384 * 1024 * 1024
}
pub fn check(file_bytes: u64, layers: usize, budget_mib: Option<u64>) -> anyhow::Result<Memory> {
    let available = available()
        .ok_or_else(|| anyhow::anyhow!("cannot measure available memory on this host"))?;
    let budget = budget_mib
        .map(|m| m.saturating_mul(1024 * 1024))
        .unwrap_or(available / 5 * 4)
        .min(available);
    let required = estimate(file_bytes, layers);
    anyhow::ensure!(
        required <= budget,
        "shard needs estimated {} MiB including KV/workspace, budget {} MiB",
        required / (1024 * 1024),
        budget / (1024 * 1024)
    );
    Ok(Memory {
        available_before_load_bytes: available,
        estimated_required_bytes: required,
        budget_bytes: budget,
    })
}

#[cfg(all(
    test,
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
mod tests {
    #[test]
    fn measures_native_available_memory() {
        assert!(super::available().is_some_and(|bytes| bytes > 0));
    }
}
