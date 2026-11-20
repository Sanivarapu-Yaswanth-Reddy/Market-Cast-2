# 08 Decisions, risks and open questions

## 1. Decision log (append new decisions; never rewrite old ones; supersede instead)

| ID | Decision | Rationale | Revisit when |
|---|---|---|---|
| D-01 | PyTorch + PyTorch Geometric | doc's stack table; largest graph ecosystem; JAX/Jraph only if mirroring DeepMind code becomes necessary | never for v1 |
| D-02 | Parquet + DuckDB, no server | fast as-of joins on a laptop | data > 50 GB |
| D-03 | Self-built MC500 universe instead of official Nifty 500 history | official membership history not on disk (G1); rule mimics semi-annual reconstitution | G1 filled: compare and report |
| D-04 | NSE is the primary exchange for every stock | data on disk is NSE; BSE only to fill gaps | BSE-only listings matter |
| D-05 | Target z uses log returns: z = ℓ / σ̂ | additive over horizons; doc's r/σ̂ differs by < 0.5% for daily moves under 10% | — |
| D-06 | Realized variance = overnight² + Garman–Klass (daily OHLC) | no intraday data; GK is about 7× more efficient than squared close-to-close returns | intraday data obtained |
| D-07 | Significance w = share of 60-day median traded value until free-float market cap exists | float-mcap needs shareholding history (G4) | G4 filled |
| D-08 | Sector history before the first classification snapshot uses the snapshot, flagged `hindsight_class` | historical four-level classification unavailable (G3) | snapshots accumulate; run an ablation with/without flagged periods |
| D-09 | Decision dates from 2008-01-01; 2005–2007 is feature warm-up | 252-day features; includes the 2008 crisis in training | longer raw history downloaded |
| D-10 | Vault = calendar 2026 | most recent full regime, untouched | each year roll forward by one year after the vault is opened |
| D-11 | Cutoff = 15:30 IST, with same-day NSE EOD files included | matches the doc's announcement and time-zone rules | pre-open event tokens added (v2) |
| D-12 | FGN noise drawn per scenario per rollout step | matches FGN; coherence across stocks comes from sharing across nodes | if paths look too jumpy across steps, test a persistent component |
| D-13 | Non-price channels persist at last value during rollout | doc: slow variables fixed within the horizon | volume/OI heads added |
| D-14 | Held-out stocks: the named 20 (`heldout_named_20.csv`, chosen 2026-11-16: one per NSE industry, two in Financial Services, 5 large / 9 mid / 6 small, all listed before 2008, no top Nifty 50 weights) plus a stratified sha256-ordered top-up to 15% of companies | owner asked for 20 companies across sectors so the test shows capability rather than memorization; mega-caps excluded so the market aggregate is not distorted | never, except to replace a name that stops trading |
| D-15 | Conservative publication-lag table for monthly macro instead of vintages | FRED gives latest-revised values only | ALFRED vintages ingested (G9) |
| D-16 | A global value dated US day d is usable for India date d+1 onward | US close is after the Indian close | — |
| D-17 | Published archive files only; no NSE internal `/api/` endpoints | terms of use and access rules | licensed feed |
| D-18 | Circuit-band inference only for non-F&O stocks | F&O stocks have dynamic bands, not fixed circuits | band history obtained (G7) |
| D-20 | Modelling key = company_id = ISIN[:9] (issuer code + equity type) | Indian ISINs change serial digits on face-value splits, so a raw-ISIN key would cut a company's history in two | verify on all inferred splits in P1.3 |
| D-22 | Split and bonus adjustment factors are inferred from price gaps matching standard ratios, not read from NSE's PREVCLOSE | verified 2026-11-16 that PREVCLOSE is unadjusted in both bhavcopy formats (RELIANCE 2017-09-07: prev 1645.4, open 823; HDFCBANK 2019-09-19: prev 2187.75, open 1099.9; GARFIBRES 2025-01-03) | precision and recall in P1.3; official table (G15) replaces inference when obtained |
| D-21 | Training on the owner's second computer with an NVIDIA GPU; data work on the main PC | owner confirmed GPU access 2026-11-16 | GPU too small for the default config |
| D-19 | Score in memory; persist full paths only for every 5th date and tagged dates | disk budget | — |
| D-23 | (2026-11-16, coding session) The graph is plain index arrays with an own scatter-softmax attention layer; **no PyTorch Geometric** (supersedes the PyG half of D-01). The EWMA variance update caps \|ℓ\| at ln 1.4 (targets stay unclipped) | one dependency fewer on a new GPU architecture; one unadjusted demerger row would otherwise inflate σ̂ for ~100 sessions | PyG needed for speed; official corporate-action table (G15) |
| D-24 | Feature store holds RAW channels once plus per-fold normaliser statistics (not one normalised tensor per fold). 52-week position and drawdown are scaled by σ̂·√window; "drawdown n" = distance below the n-day high; skew/kurtosis are population moments. Inside a rollout: gap ← 0, range ← Parkinson value from v̂, close location ← 0.5, market-relative and liquidity channels held, σ floor frozen, circuit band assumed 20% | 1.8 GB instead of ~25 GB; the rollout must recompute price channels from raw values, and each formula must be O(1) per step and identical in numpy and torch (tested) | rollout skill lags the direct heads |
| D-25 | `is_rbi_policy_day` is **not** a forcing. Sector tree has 3 levels (macro sector → sector → industry) built from constituent lists + `configs/sector_manual.yaml` + keyword rules | the RBI file lists only days the rate CHANGED, so using it for future days leaks that a change happens; the NSE four-level file is not on disk (G3) | MPC meeting calendar compiled; G3 filled |
| D-27 | (owner, 2026-11-16) Infosys (`INE009A01`) is excluded from the stratified held-out top-up via `configs/heldout_exclusions.yaml`; the next company in its bucket's sha256 order (`INE018I01`) takes the slot. Held-out set stays 231 companies (15.2%) | the sha rule had no mega-cap guard, unlike the named 20; a top Nifty 50 weight missing from the training market aggregate distorts it | owner decides on the other Nifty 50 names the rule picked (Adani Ports, Hindustan Unilever, Jio Financial, L&T, Tata Motors PV, Trent) |
| D-26 | FGN conditional norms use one Linear(128→d) shared across node types times a learned per-type gain vector (not a full Linear per node type). Default model = 17.0 M parameters | the full per-type maps alone cost 12 M parameters (27.8 M total against the 11–15 M budget) | ablation shows per-type maps matter |

## 2. Risk register

| Risk | Signal | Mitigation | Owner of check |
|---|---|---|---|
| Lookahead leakage | backtest much better than live; skill concentrated right after events | PIT store, as-of joins, purge/embargo, truncation-invariance test, frozen live forecasts | leakage suite in CI |
| Survivorship bias | every model looks skilful | delisted names in universe from day one; terminal returns | P1.5 check |
| Corporate-action errors | spikes of ±50% returns, broken volatility | factor inference + hand verification + 20 stock-day gate | P1.3 |
| Missing dividends (G2) | small negative bias on ex-dates | ex-date masking; fill G2 in P5.1 | P1.9 |
| Text-model contamination | news module "knows" outcomes | evaluate only after the model's knowledge cutoff; rule-based extraction first | P5.4 |
| Overfitting / multiple testing | best of many variants fails on the vault | log every run; DSR; held-out stocks and years; vault opened once | every gate |
| Small effective sample | wide CIs; unstable rankings | block bootstrap; report CIs; avoid tuning on test | every scorecard |
| Regime change | skill decays after structural shifts (T+1, SEBI F&O rules) | quarterly refits; decay monitoring; before/after slices | P6.2 |
| Reflexivity | a pattern weakens once known | focus on risk estimates, which stay useful when shared | product |
| Tail underestimation | record moves fall outside all paths | report tail coverage; blend with hand-built stress scenarios | G4 |
| Hub domination in the graph | large caps drown the rest | normalized messages, hub cap, ablation | P3.5 |
| Data-access breakage | NSE blocks or changes archive paths | polite downloader, cooldown, resumable; formats documented | P0.1, P6.1 |
| Disk and OneDrive sync | sync conflicts, locked files, slow builds | move `data/` outside OneDrive (ask owner); atomic writes | P0.2 |
| Compute limits | runs too slow on free GPUs | fast-mode folds; small model; rent GPU only with approval | P3.4 |
| User overtrust | ranges read as promises | ranges only, plain-language uncertainty, no single targets | P6.4 |
| Regulation (SEBI RA/IA, algo rules) | any public stock-specific view | internal until legal advice; portfolio-level education only | owner before P6 public use |

## 3. Open questions for the owner

| ID | Question | Why it matters | Default if unanswered |
|---|---|---|---|
| Q-01 | ~~GPU?~~ **Answered 2026-11-16:** an NVIDIA GPU is available on another computer (D-21). Still open: GPU model/VRAM (record in P0.2) and any budget for paid data (consensus, CMIE Prowess) | compute plan; expectations layer | no paid data |
| Q-02 | ~~Which 20 companies?~~ **Answered 2026-11-16:** owner left the choice to the agent: any 20 across sectors. Chosen list in `heldout_named_20.csv` (D-14) | — | — |
| Q-03 | May `data\` move outside OneDrive (e.g. `C:\MarketCastData` with a junction)? | 4–5 GB raw plus processed and features | leave in place; warn about sync |
| Q-04 | Review `scenario_tags_seed.csv` (add or remove episodes) | evaluation slices | use verified rows only |
| Q-05 | Accept MC500, or compile official Nifty 500 history from index-change notices? | universe realism | MC500 |
| Q-06 | Any external deadline (competition, demo, app milestone)? | phase ordering | doc's six-month plan |
| Q-07 | Confirm transaction-cost assumptions in `configs/costs.yaml` | economic-value test | values in 05 §5 |
| Q-08 | When will legal advice be sought for the app's risk views? | gates public use | nothing public |

## 4. Facts to verify before they are used in a gate

* The three 2025 RBI rate-change dates (meeting dates used as effective dates) and whether a 2026 change exists (RBI homepage showed 5.50% repo during collection, against 5.25% in the file after the December 2025 cut).
* Special-session and halt dates listed in 06 P1.1 and P1.9.
* The F&O ban archive URL pattern (06 P1.7).
* Every row of `scenario_tags_seed.csv` (marked `verified = false`).
