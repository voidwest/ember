//! Low-frequency Linux process-residency snapshots for lifecycle experiments.
//!
//! This module intentionally reads only procfs accounting files at explicit
//! phase boundaries. It does not sample in the inference hot path.

use serde::Serialize;
use std::fs;
use std::time::Instant;

#[derive(Debug, Clone, Serialize)]
pub struct ResidencySnapshot {
    pub phase: String,
    pub elapsed_ns: u64,
    pub measurement_ns: u64,
    pub rss_kib: u64,
    pub peak_rss_kib: u64,
    pub anonymous_pss_kib: u64,
    pub file_pss_kib: u64,
    pub minor_faults: u64,
    pub major_faults: u64,
}

#[derive(Debug)]
pub struct ResidencyRecorder {
    process_start: Instant,
    enabled: bool,
    measurement_ns: u64,
    snapshots: Vec<ResidencySnapshot>,
}

impl Default for ResidencyRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl ResidencyRecorder {
    pub fn new() -> Self {
        Self {
            process_start: Instant::now(),
            enabled: true,
            measurement_ns: 0,
            snapshots: Vec::new(),
        }
    }

    /// Construct a phase recorder that preserves markers but skips procfs.
    /// Used only to quantify whether residency measurement perturbs timings.
    pub fn timing_only() -> Self {
        Self {
            process_start: Instant::now(),
            enabled: false,
            measurement_ns: 0,
            snapshots: Vec::new(),
        }
    }

    pub fn capture(&mut self, phase: impl Into<String>) -> anyhow::Result<()> {
        let phase = phase.into();
        if !self.enabled {
            self.snapshots.push(ResidencySnapshot {
                phase,
                elapsed_ns: self.process_start.elapsed().as_nanos() as u64,
                measurement_ns: 0,
                rss_kib: 0,
                peak_rss_kib: 0,
                anonymous_pss_kib: 0,
                file_pss_kib: 0,
                minor_faults: 0,
                major_faults: 0,
            });
            return Ok(());
        }

        let measurement_start = Instant::now();
        let smaps = fs::read_to_string("/proc/self/smaps_rollup")?;
        let status = fs::read_to_string("/proc/self/status")?;
        let stat = fs::read_to_string("/proc/self/stat")?;

        let measurement_ns = measurement_start.elapsed().as_nanos() as u64;
        self.measurement_ns = self.measurement_ns.saturating_add(measurement_ns);
        let (minor_faults, major_faults) = parse_process_faults(&stat)?;
        self.snapshots.push(ResidencySnapshot {
            phase,
            elapsed_ns: self.process_start.elapsed().as_nanos() as u64,
            measurement_ns,
            rss_kib: proc_kib(&smaps, "Rss:")?,
            peak_rss_kib: proc_kib(&status, "VmHWM:")?,
            anonymous_pss_kib: proc_kib(&smaps, "Pss_Anon:")?,
            file_pss_kib: proc_kib(&smaps, "Pss_File:")?,
            minor_faults,
            major_faults,
        });
        Ok(())
    }

    pub fn measurement_ns(&self) -> u64 {
        self.measurement_ns
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn snapshots(&self) -> &[ResidencySnapshot] {
        &self.snapshots
    }
}

fn proc_kib(contents: &str, field: &str) -> anyhow::Result<u64> {
    let line = contents
        .lines()
        .find(|line| line.starts_with(field))
        .ok_or_else(|| anyhow::anyhow!("missing procfs field {field}"))?;
    line.split_whitespace()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("missing value for procfs field {field}"))?
        .parse()
        .map_err(Into::into)
}

fn parse_process_faults(stat: &str) -> anyhow::Result<(u64, u64)> {
    let after_comm = stat
        .rsplit_once(')')
        .map(|(_, fields)| fields)
        .ok_or_else(|| anyhow::anyhow!("malformed /proc/self/stat"))?;
    let fields = after_comm.split_whitespace().collect::<Vec<_>>();
    let minor_faults = fields
        .get(7)
        .ok_or_else(|| anyhow::anyhow!("missing minflt in /proc/self/stat"))?
        .parse()?;
    let major_faults = fields
        .get(9)
        .ok_or_else(|| anyhow::anyhow!("missing majflt in /proc/self/stat"))?
        .parse()?;
    Ok((minor_faults, major_faults))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_proc_kib_field() {
        let contents = "Rss:               12345 kB\nPss_Anon:             9 kB\n";
        assert_eq!(proc_kib(contents, "Rss:").unwrap(), 12_345);
        assert_eq!(proc_kib(contents, "Pss_Anon:").unwrap(), 9);
    }

    #[test]
    fn parses_fault_fields_after_parenthesized_command() {
        let stat = "42 (ember test) R 1 2 3 4 5 6 700 8 900 10";
        assert_eq!(parse_process_faults(stat).unwrap(), (700, 900));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn captures_live_process_snapshot() {
        let mut recorder = ResidencyRecorder::new();
        recorder.capture("test").unwrap();
        let snapshot = &recorder.snapshots()[0];
        assert_eq!(snapshot.phase, "test");
        assert!(snapshot.rss_kib > 0);
        assert!(snapshot.peak_rss_kib > 0);
    }

    #[test]
    fn timing_only_preserves_phase_markers_without_procfs_work() {
        let mut recorder = ResidencyRecorder::timing_only();
        recorder.capture("test").unwrap();
        let snapshot = &recorder.snapshots()[0];
        assert_eq!(snapshot.phase, "test");
        assert_eq!(snapshot.measurement_ns, 0);
        assert_eq!(snapshot.rss_kib, 0);
    }
}
