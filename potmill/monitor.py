import csv
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

TASK_STAGES = [
    "entropy",
    "labeling",
    "b_collecting",
    "featurization",
    "fitting",
    "cost",
    "pareto",
    "pops",
]


class ResourceMonitor:
    """Background GPU, CPU, and task progress monitor.

    Runs as a daemon thread, logging aggregated resource stats and
    pipeline task counts to CSV. Designed as a context manager.
    """

    GPU_QUERY_FIELDS = (
        "index,name,utilization.gpu,utilization.memory,"
        "memory.used,memory.total,temperature.gpu,power.draw"
    )
    CSV_HEADER = [
        "timestamp",
        "elapsed_s",
        "mean_gpu_util_pct",
        "mean_gpu_mem_util_pct",
        "mean_gpu_mem_used_gib",
        "mean_gpu_power_w",
        "mean_cpu_util_pct",
    ] + [col for s in TASK_STAGES for col in (f"n_{s}", f"n_{s}_running")]

    def __init__(self, log_dir, interval=30.0, console_interval=60.0, nodelist=None, n_nodes=1):
        self.log_dir = log_dir
        self.interval = interval
        self.console_interval = console_interval
        self.n_nodes = n_nodes
        self.nodelist = nodelist
        self._stop_event = threading.Event()
        self._thread = None
        self._csv_file = None
        self._csv_writer = None
        self._lock = threading.Lock()
        self._latest = None
        self._last_console_time = 0.0
        self._start_time = None
        self._prev_cpu_stats = {}  # node_rank -> (total, active)
        self._task_counts = {}  # stage_name -> remaining count

    def __enter__(self):
        csv_path = os.path.join(self.log_dir, "pipeline_monitor.csv")
        self._csv_file = open(csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self.CSV_HEADER)
        self._csv_file.flush()
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print(f"[Monitor] Started — logging to {csv_path}", flush=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
        print("[Monitor] Stopped", flush=True)
        return False

    def update_task_counts(self, counts):
        """Update pipeline task remaining counts (called from main thread).

        Args:
            counts: dict mapping stage name to remaining future count, e.g.
                     {"entropy": 30, "labeling": 8, ...}
        """
        with self._lock:
            self._task_counts = counts.copy()

    def get_latest(self):
        with self._lock:
            return self._latest

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            try:
                elapsed_s = time.monotonic() - self._start_time
                gpu_data = self._collect_gpu_data()
                cpu_util = self._collect_cpu_data()
                with self._lock:
                    task_counts = self._task_counts.copy()
                    self._latest = gpu_data

                self._write_csv(elapsed_s, gpu_data, cpu_util, task_counts)

                now = time.monotonic()
                if now - self._last_console_time >= self.console_interval:
                    self._print_console_summary(gpu_data, cpu_util, task_counts)
                    self._last_console_time = now
            except Exception as e:
                import traceback

                print(f"[Monitor] Warning: monitoring error: {e}", file=sys.stderr, flush=True)
                traceback.print_exc()
            self._stop_event.wait(self.interval)

    # ---- GPU collection (nvidia-smi) ----

    def _collect_gpu_data(self):
        query_args = [
            "nvidia-smi",
            f"--query-gpu={self.GPU_QUERY_FIELDS}",
            "--format=csv,noheader,nounits",
        ]
        if self.n_nodes > 1:
            cmd = ["flux", "exec", "-r", "all", "-l"] + query_args
            timeout = 30
        else:
            cmd = query_args
            timeout = 10

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return None
        if result.returncode != 0:
            return None
        return self._parse_gpu_output(result.stdout)

    def _parse_gpu_output(self, output):
        rows = []
        for raw_line in output.strip().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if self.n_nodes > 1 and ": " in line:
                rank_str, rest = line.split(": ", 1)
                try:
                    rank = int(rank_str)
                except ValueError:
                    rank = rank_str
            else:
                rank = 0
                rest = line

            parts = [p.strip() for p in rest.split(",")]
            if len(parts) < 8:
                continue

            rows.append(
                {
                    "node_rank": rank,
                    "gpu_index": parts[0],
                    "gpu_name": parts[1],
                    "gpu_util_pct": parts[2],
                    "mem_util_pct": parts[3],
                    "mem_used_mib": parts[4],
                    "mem_total_mib": parts[5],
                    "temperature_c": parts[6],
                    "power_draw_w": parts[7],
                }
            )
        return rows

    # ---- CPU collection (/proc/stat) ----

    def _collect_cpu_data(self):
        """Per-PHYSICAL-core CPU utilization (delta), averaged over the allocation.

        Reads the PER-CPU lines of /proc/stat (logical/SMT threads) and aggregates each physical
        core's SMT siblings (summed active time, capped at 100%). PotMill binds 1 rank per physical
        core, so a plain logical average would read ~half (the idle SMT siblings drag it down --
        Perlmutter is SMT-2). Per-physical gives true core occupancy and stays correct for the
        multithreaded GPU/UMA stages too (which DO use both siblings).
        """
        if self.n_nodes > 1:
            cmd = ["flux", "exec", "-r", "all", "-l", "cat", "/proc/stat"]
            timeout = 30
        else:
            cmd = ["cat", "/proc/stat"]
            timeout = 10

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return None
        if result.returncode != 0:
            return None

        sib = self._phys_map()  # logical cpu index -> physical core id (this node's topology)
        phys_active = {}  # (rank, physical core) -> summed active jiffy delta over its SMT siblings
        phys_total = {}  # (rank, physical core) -> per-thread total jiffy delta (~wall*HZ)
        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if self.n_nodes > 1 and ": " in line:
                rank_str, rest = line.split(": ", 1)
                try:
                    rank = int(rank_str)
                except ValueError:
                    continue
            else:
                rank = 0
                rest = line

            parts = rest.split()
            label = parts[0] if parts else ""
            if (
                len(parts) < 6
                or len(label) <= 3
                or not label.startswith("cpu")
                or not label[3:].isdigit()
            ):
                continue  # skip the "cpu" aggregate and non-cpu lines (intr/ctxt/...)
            idx = int(label[3:])
            values = [int(x) for x in parts[1:]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            total = sum(values)
            active = total - idle

            key = (rank, idx)
            prev = self._prev_cpu_stats.get(key)
            self._prev_cpu_stats[key] = (total, active)
            if prev is None:
                continue
            d_total = total - prev[0]
            d_active = active - prev[1]
            if d_total <= 0:
                continue
            pkey = (rank, sib.get(idx, idx))
            phys_active[pkey] = phys_active.get(pkey, 0) + d_active
            phys_total[pkey] = max(phys_total.get(pkey, 0), d_total)

        utils = [
            min(100.0, phys_active[k] / phys_total[k] * 100.0)
            for k in phys_active
            if phys_total.get(k, 0) > 0
        ]
        if utils:
            return sum(utils) / len(utils)
        return None

    def _phys_map(self):
        """logical-cpu-index -> physical-core-id from this node's topology (cached; nodes uniform)."""
        m = getattr(self, "_physmap", None)
        if m is not None:
            return m
        m = {}
        try:
            import glob

            for d in glob.glob("/sys/devices/system/cpu/cpu[0-9]*"):
                try:
                    idx = int(d.rsplit("cpu", 1)[-1])
                    with open(d + "/topology/thread_siblings_list") as fh:
                        sibs = fh.read().strip()
                    m[idx] = int(sibs.replace("-", ",").split(",")[0])
                except (OSError, ValueError):
                    pass
        except OSError:
            pass
        self._physmap = m
        return m

    # ---- CSV output ----

    def _write_csv(self, elapsed_s, gpu_data, cpu_util, task_counts):
        gpu_utils, mem_utils, mem_used, powers = [], [], [], []
        for row in gpu_data or []:
            try:
                gpu_utils.append(float(row["gpu_util_pct"]))
                mem_utils.append(float(row["mem_util_pct"]))
                mem_used.append(float(row["mem_used_mib"]))
                powers.append(float(row["power_draw_w"]))
            except (ValueError, TypeError):
                continue
        n = len(gpu_utils) or 1

        csv_row = [
            datetime.now().isoformat(timespec="seconds"),
            f"{elapsed_s:.1f}",
            f"{sum(gpu_utils) / n:.1f}" if gpu_utils else "",
            f"{sum(mem_utils) / n:.1f}" if mem_utils else "",
            f"{sum(mem_used) / n / 1024:.2f}" if mem_used else "",
            f"{sum(powers) / n:.1f}" if powers else "",
            f"{cpu_util:.1f}" if cpu_util is not None else "",
        ]
        for stage in TASK_STAGES:
            val = task_counts.get(stage)
            csv_row.append(str(val) if val is not None else "")
            val_running = task_counts.get(f"{stage}_running")
            csv_row.append(str(val_running) if val_running is not None else "")

        self._csv_writer.writerow(csv_row)
        self._csv_file.flush()

    # ---- Console summary ----

    def _print_console_summary(self, gpu_data, cpu_util, task_counts):
        lines = []

        # GPU summary
        if gpu_data:
            utils = []
            mem_used = []
            mem_total = []
            for row in gpu_data:
                try:
                    utils.append(float(row["gpu_util_pct"]))
                    mem_used.append(float(row["mem_used_mib"]))
                    mem_total.append(float(row["mem_total_mib"]))
                except (ValueError, TypeError):
                    continue
            if utils:
                n = len(utils)
                avg_util = sum(utils) / n
                min_util = min(utils)
                max_util = max(utils)
                avg_mem = sum(mem_used) / n / 1024
                avg_mem_total = sum(mem_total) / n / 1024 if mem_total else 0
                now = datetime.now().strftime("%H:%M:%S")
                gpu_line = (
                    f"[Monitor] {now} | {n} GPUs | "
                    f"Util: avg={avg_util:.0f}% min={min_util:.0f}% max={max_util:.0f}% | "
                    f"Mem: avg={avg_mem:.1f}/{avg_mem_total:.1f} GiB"
                )

                if self.n_nodes > 1:
                    by_node = {}
                    for row in gpu_data:
                        rank = row["node_rank"]
                        by_node.setdefault(rank, []).append(row)
                    parts = []
                    for rank in sorted(by_node.keys()):
                        node_utils = []
                        for r in by_node[rank]:
                            try:
                                node_utils.append(float(r["gpu_util_pct"]))
                            except (ValueError, TypeError):
                                continue
                            if node_utils:
                                node_avg = sum(node_utils) / len(node_utils)
                                parts.append(f"N{rank}={node_avg:.0f}%")
                    if parts:
                        gpu_line += " | " + " ".join(parts)
                lines.append(gpu_line)

        # CPU summary
        if cpu_util is not None:
            lines.append(f"[Monitor] CPU Util (physical): avg={cpu_util:.0f}%")

        # Active stages
        active = [f"{s}({c})" for s, c in task_counts.items() if c and c > 0]
        if active:
            lines.append(f"[Monitor] Active: {' '.join(active)}")

        for line in lines:
            print(line, flush=True)
