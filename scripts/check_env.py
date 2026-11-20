"""MarketCast environment check. Standard library only, so it runs on any Python.
Usage:  python scripts/check_env.py   (run from the project root, or double-click CHECK_ENV.bat)
Writes reports/env_report.txt and prints the same text. Safe: reads only, installs nothing."""
from __future__ import annotations
import ctypes, datetime as dt, importlib, importlib.metadata as md, os, platform, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "env_report.txt"
lines: list[str] = []
def log(s: str = "") -> None:
    print(s); lines.append(s)

# (import name, pip name, role, required on)  role: data = needed on data PC, train = GPU machine, opt = optional
PKGS = [
    ("numpy", "numpy", "data+train"), ("pandas", "pandas", "data+train"),
    ("pyarrow", "pyarrow", "data+train"), ("duckdb", "duckdb", "data+train"),
    ("scipy", "scipy", "data+train"), ("statsmodels", "statsmodels", "data+train"),
    ("arch", "arch", "data+train"), ("lightgbm", "lightgbm", "data+train"),
    ("sklearn", "scikit-learn", "data+train"), ("yaml", "pyyaml", "data+train"),
    ("tqdm", "tqdm", "data+train"), ("matplotlib", "matplotlib", "data+train"),
    ("pytest", "pytest", "dev"), ("hypothesis", "hypothesis", "dev"),
    ("scoringrules", "scoringrules", "dev (metric tests)"), ("properscoring", "properscoring", "dev (metric tests)"),
    ("mlflow", "mlflow", "train"), ("rich", "rich", "dev"),
    ("torch", "torch", "train"), ("torch_geometric", "torch-geometric", "train"),
    ("imdlib", "imdlib", "optional (IMD rainfall)"), ("chronos", "chronos-forecasting", "optional (baseline B6)"),
    ("timesfm", "timesfm", "optional (baseline B6)"),
]

def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa
        return f"(not available: {type(e).__name__})"

log(f"MarketCast environment report  {dt.datetime.now().isoformat(timespec='seconds')}")
log(f"Machine: {platform.node()}  OS: {platform.platform()}")
log(f"Python {sys.version.split()[0]}  at {sys.executable}")
log(f"Project root: {ROOT}")
if "onedrive" in str(ROOT).lower():
    log("WARNING: project is inside OneDrive. Fine for code; keep large data outside it (decision Q-03).")

log("\n== Packages ==")
missing = []
for imp, pip, role in PKGS:
    try:
        importlib.import_module(imp)
        try: v = md.version(pip)
        except Exception: v = "?"
        log(f"  OK       {pip:22s} {v:12s} [{role}]")
    except Exception as e:
        tag = "optional" if role.startswith("optional") else ("later" if role == "train" and not shutil.which("nvidia-smi") else "MISSING")
        log(f"  {tag:8s} {pip:22s} {'-':12s} [{role}]  ({type(e).__name__})")
        if tag == "MISSING": missing.append(pip)

log("\n== GPU / PyTorch ==")
try:
    import torch
    log(f"  torch {torch.__version__}  built for CUDA {torch.version.cuda}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        cc = f"sm_{p.major}{p.minor}"
        log(f"  GPU: {p.name}  VRAM {p.total_memory/2**30:.1f} GB  compute capability {cc}")
        arch = torch.cuda.get_arch_list()
        log(f"  torch arch list: {' '.join(arch)}")
        if cc not in arch and not any(a.startswith("compute_") for a in arch):
            log(f"  PROBLEM: this torch build has no kernels for {cc}. RTX 50-series needs a CUDA 12.8+ build (cu128).")
        try:
            x = torch.randn(256, 256, device="cuda"); (x @ x).sum().item(); log("  matmul on GPU: OK")
        except Exception as e: log(f"  matmul on GPU FAILED: {e}")
    else:
        log("  CUDA not available to torch (CPU-only build, no NVIDIA driver, or this is the data PC).")
except ImportError:
    log("  torch not installed")
smi = shutil.which("nvidia-smi")
log("  nvidia-smi: " + (sh([smi, "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"]) if smi else "not found on PATH"))

log("\n== Tools ==")
for t in ("git", "conda", "nvidia-smi"):
    log(f"  {t:11s} {shutil.which(t) or 'not found on PATH'}")
if shutil.which("git"): log("  " + sh(["git", "--version"]))

log("\n== Resources ==")
du = shutil.disk_usage(ROOT); log(f"  Disk free at project drive: {du.free/2**30:.0f} GB of {du.total/2**30:.0f} GB")
try:
    class MS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("ml", ctypes.c_ulong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                    ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
    m = MS(); m.l = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    log(f"  RAM: {m.tp/2**30:.1f} GB total, {m.ap/2**30:.1f} GB free")
except Exception: log("  RAM: (could not read)")
log(f"  CPU threads: {os.cpu_count()}")

log("\n== Project data ==")
raw = ROOT / "data" / "raw"
if raw.exists():
    for sub in ("nse/cm_bhavcopy", "nse/delivery_mto", "nse/fo_bhavcopy", "nse/indices_close_all", "nse/reference", "global", "factors", "flows", "macro"):
        d = raw / sub
        n = sum(1 for _ in d.rglob("*") if _.is_file()) if d.exists() else 0
        log(f"  raw/{sub:16s} {n:6d} files")
else:
    log("  data/raw not found")
proc = ROOT / "data" / "processed"
log("  processed: " + (", ".join(sorted(p.name for p in proc.glob('*.parquet'))) if proc.exists() else "NOT BUILT YET (run scripts\\marketcast_build_panel.py)"))

log("\n== Verdict ==")
log("  'later' = training-only package, expected to be absent on a machine without an NVIDIA GPU.")
log("  Missing required packages: " + (", ".join(missing) if missing else "none"))
if missing:
    log("  Fix: see dev_plan\\09_ENVIRONMENT_SETUP.md (conda env create -f environment-data.yml  or  environment-gpu.yml)")
OUT.parent.mkdir(exist_ok=True); OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"\nReport saved to {OUT}")
