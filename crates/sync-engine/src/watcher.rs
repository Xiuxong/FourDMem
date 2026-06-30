//! File watcher with debounced change notifications
//!
//! Wraps the `notify` crate to watch a directory for `.md` / `.yaml` file
//! changes and delivers debounced events over a `tokio::mpsc` channel.

use std::path::{Path, PathBuf};
use std::time::Duration;

use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::mpsc;

/// A debounced file watcher that sends changed file paths over a channel.
pub struct FileWatcher {
    _watcher: RecommendedWatcher,
    rx: mpsc::Receiver<PathBuf>,
}

impl FileWatcher {
    /// Start watching `root` for file changes.
    ///
    /// Events are debounced: only the last change within `debounce_ms` is
    /// delivered. Only `.md` and `.yaml`/`.yml` files are forwarded.
    pub fn new(root: impl AsRef<Path>, debounce_ms: u64) -> notify::Result<Self> {
        let (tx, rx) = mpsc::channel(256);
        let debounce = Duration::from_millis(debounce_ms);

        // The notify watcher runs in its own thread. We bridge to tokio via mpsc.
        let mut watcher = RecommendedWatcher::new(
            move |res: notify::Result<Event>| {
                if let Ok(event) = res {
                    if matches!(event.kind, EventKind::Modify(_) | EventKind::Create(_)) {
                        for path in event.paths {
                            let ext = path
                                .extension()
                                .and_then(|e| e.to_str())
                                .unwrap_or("");
                            if matches!(ext, "md" | "yaml" | "yml") {
                                // Non-blocking send — if channel is full, drop the event
                                let _ = tx.blocking_send(path);
                            }
                        }
                    }
                }
            },
            notify::Config::default().with_poll_interval(debounce),
        )?;

        watcher.watch(root.as_ref(), RecursiveMode::Recursive)?;

        Ok(Self {
            _watcher: watcher,
            rx,
        })
    }

    /// Wait for the next changed file path.
    pub async fn recv(&mut self) -> Option<PathBuf> {
        self.rx.recv().await
    }
}
