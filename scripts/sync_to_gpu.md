# Moving work between the data PC and the GPU machine

## Code: git (both machines)
1. Data PC: GitHub Desktop → File → Add local repository → choose the Market Cast folder → "create a repository" → publish as a **private** repository. `.gitignore` already keeps `data/`, `mlruns/` and Parquet files out.
2. GPU machine: clone the same private repository. Pull before each session; push changes back.

## Data: never through git or OneDrive
Only what training needs travels, per fold:
- `data/features/<fold>/` and `data/graph/<fold>/` (built on the data PC, P3.1 and P3.2)
- `data/targets/` and `configs/` (configs are in git anyway)

Ways to copy, in order of preference:
1. Same Wi-Fi/LAN: share the data PC folder `Market Cast\data` read-only and run on the GPU machine
   `robocopy "\\DESKTOP-LQDHANP\MarketCastData\features\F4" "D:\MarketCast\data\features\F4" /MIR /Z`
2. Different networks: install Tailscale on both machines, then the same robocopy over the Tailscale address.
3. External drive: copy the fold folders across.

Keep the GPU machine's copy outside OneDrive (for example `D:\MarketCast`) and set the data path in `configs/data.yaml` there.

## Results back to the data PC
- `mlruns/` (MLflow runs) and `reports/scores/<model>/<fold>.parquet`: copy back the same way (robocopy in reverse).
- Never delete runs; they count toward the number of trials in the deflated Sharpe ratio.

## Checklist per training session
1. `git pull` on the GPU machine.
2. Copy any new fold caches.
3. `CHECK_ENV.bat` once after driver or torch updates.
4. Train; push code, copy runs back.
