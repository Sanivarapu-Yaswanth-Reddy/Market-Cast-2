# 04 Model architecture

Ten layers in three parts (doc: Sense, Simulate, Answer). Sizes are first-version defaults for MC500 on one GPU. Every number marked (cfg) lives in `configs/model/*.yaml` and is a tunable hyperparameter on validation years only.

Notation: N_s ≈ 500 stock nodes per date (variable; masks, never padding with fake companies), N_h hierarchy nodes (NSE tree ≈ 12 macro-economic sectors + 22 sectors + 59 industries + ≈ 197 basic industries, plus 1 market node; exact counts come from the classification snapshot in P1.6), N_g = 16 global nodes, T = 60 history days, F_s ≈ 80 stock channels, d_e = 128, d = 256.

## Part 1: Sense

### L0 PIT data foundation
Specified in `02_DATA_CONTRACT.md`. The model never reads raw files; it reads fold-specific tensors produced by `features/build.py` from the PIT store.

### L1 Market state
Per decision date t: `X_s ∈ R^{N_s×T×F_s}` (stock history), `X_g ∈ R^{N_g×T×F_g}` (global history), `F ∈ R^{(T+H)×F_f}` (forcings for history and future days), masks `M_s ∈ {0,1}^{N_s×T}`. Feature definitions: `03_FEATURES_AND_TARGETS.md`.

### L2 Temporal encoder (shared by all stocks)
```
x_{i,τ} → Linear(F_s+F_f → d_e) + learned position emb(τ) → 4 × [pre-LN Transformer block, 4 heads, d_ff = 512, dropout 0.1]
       → attention pooling with one learned query (masked) → h0_i ∈ R^{d_e} → Linear(d_e → d)
```
Global nodes use a second encoder with the same shape (separate weights). Hierarchy nodes start from the significance-weighted mean of their children's h0 plus a learned node-type embedding; where an official NSE index exists for the node (e.g. Nifty Bank, Nifty IT), its index history is encoded by the global encoder and added. New listings: masked positions; if fewer than 20 valid days, h0_i is replaced by its basic industry's h0 plus a learned "new listing" embedding.

### L3 Event assimilation (Phase 5)
Each event becomes a token `e = [type emb (≈40 types), direction ∈ {−1,0,+1}, size vs expectation (z-scored), confidence, ln(1+age_days), text emb → Linear(·→d)]`. Node i attends over its tokens: `h_i += CrossAttn(q = h_i, K/V = tokens_i)` with additive bias `−κ_type·age_days` (κ learned, ≥ 0). Shock-mode inputs (named shocks) enter through the same path as synthetic tokens on the chosen node.

## Part 2: Simulate

### L4 Sector fabric (graph)

Node types: `stock, basic_industry, industry, sector, macro_sector, market, global`.

| Edge type | Direction | Construction (all PIT as of t) | Weight prior w_j |
|---|---|---|---|
| `up` | child → parent | sector_map valid on t; conglomerates get one `segment` membership per business (revenue-share weights, refreshed at each annual report filing date) | child significance share within parent |
| `down` | parent → child | reverse of `up` | parent → child uses child share (learned per-child gate acts as beta) |
| `corr` | stock → stock | each stock's top-10 peers by 120-day correlation of z, refreshed on the last trading day of each month from past data only; shrink ρ̃ = 0.5·ρ̂ + 0.5·ρ̄_industry; at most 3 of 10 neighbours replaced per refresh | ρ̃ (as edge feature) and peer significance |
| `global` | global → industry / market | curated exposure table (crude → paints, aviation, OMCs; USDINR → IT, pharma exporters; metals → steel, aluminium; agri → FMCG, fertilisers, tractors); every global → market | 1 |
| `supply` | stock → stock | curated supplier–customer links with `available_at` of the source document (Phase 4+) | confidence |
| `group` | stock ↔ stock | business-group links (curated for the top 100 first; G8) | confidence |

Significance (`ref.significance`): w_j = float-mcap share of child j inside its parent when G4 is filled; until then, share of 60-day median traded value (decision D-07). Computed as of t, never with later prices.

Attention with the significance prior (the doc's formula, per edge type e and head h):

```
α_ij = softmax_j( q_i·k_j / sqrt(d_head) + λ_{e,h}·log w_j + b_e )        λ init: 1.0 for up, 0.5 otherwise; learned
Hub cap: for receivers with ≥ 3 senders, α_ij ← min(α_ij, 0.5), then renormalize once.
Messages are LayerNorm'd before aggregation.
```

Separate edge types for curated and learned edges, so ablations can remove each.

### L5 Processor (graph transformer)
10 layers (cfg 8–12), d = 256, 8 heads, d_ff = 1024, pre-norm, residual. Each layer runs three passes and an MLP:

```
for layer ℓ in 1..L:
    # Up: level-synchronous bottom-up so a stock signal reaches the market node within one layer
    for level in [basic_industry, industry, sector, macro_sector, market]:
        h[level] += Attn_up(h[level] ← h[children])            # market also attends to all global nodes
    # Down: top-down, with a learned per-child sensitivity gate g_i = sigmoid(Linear(h_i))
    for level in [macro_sector, sector, industry, basic_industry, stock]:
        h[level] += g ⊙ Attn_down(h[level] ← h[parent(s)])
    # Across: cross-links
    h[stock]    += Attn_across(h[stock] ← h[corr ∪ supply ∪ group neighbours])
    h[industry] += Attn_global(h[industry] ← h[global exposures])
    # Node-wise MLP with FGN-conditioned norm (L6)
    h += MLP(CondLN_ℓ(h, ε))
```

Thin basic industries (≤ 3 listed members) have their state mixed with the parent industry: h ← 0.5·h + 0.5·h_parent before the down pass.

### L6 Stochastic core (FGN)
```
ε ~ N(0, I_16)   one draw per scenario per rollout step, shared by every node and layer
c = MLP(16 → 128 → 128, SiLU)(ε)
for every norm in processor and decoder:  CondLN(x) = LN(x)·(1 + γ_{ℓ,type}(c)) + β_{ℓ,type}(c)
γ, β: Linear(128 → d) per layer and per node type, zero-initialised (so training starts deterministic)
```
Because one 16-dimensional vector moves the whole market, the cheapest way to cut CRPS is to learn coherent market moods (risk-off, rotation), not independent noise per stock. Epistemic uncertainty: train 4 seeds independently; a 64-path forecast is 4 seeds × 16 noise draws.

### L7 Output heads
| Head | Input | Output | Loss |
|---|---|---|---|
| return | stock h | ẑ_{i,t+1} (one sample per scenario) | fair CRPS |
| volatility | stock h | v̂_{i,t+1} = ln σ̂_{i,t} + Δ̂ (residual on current log vol) | fair CRPS on v |
| market consistency | market node h | ẑ for official Nifty 50 (when available) | fair CRPS; plus penalty on gap with the member aggregate |
| trend cross-check | market node h | softmax over {up, sideways, down} for h = 5, 20 | log loss (labels fold-local) |
| event (Phase 5) | stock h + event tokens | results-day abnormal-return sample; P(DD ≥ 10% within 20d); P(circuit hit next day) | CRPS; log loss |

Sector outputs have no head: sector return in scenario m is the significance-weighted sum of member returns, A_s = R_s − R_market. This guarantees sector scenarios never contradict stock scenarios.

**Phase 3 deterministic variant:** same network, no ε, volatility head outputs ŝ = predicted log variance, loss = QLIKE-consistent `RV²/exp(ŝ) + ŝ`. Also train direct 5- and 20-day heads for comparison with the rollout later.

### L8 Autoregressive rollout
```
inputs: state at t (histories, masks), forcings F_{t+1..t+H}, noise draws ε^{m,k}
for m in 1..M:                       # vectorised over m on GPU
    S = state_t
    for k in 1..H:
        (ẑ, v̂) = f_θ(S, F_{t+k}; ε^{m,k})
        ℓ̂ = ẑ · σ̂(S)                                    # back to log return
        if stock has fixed circuit band b (non-F&O): ℓ̂ = clip(ℓ̂, ln(1−b), ln(1+b)); set circuit_carry
        update price-based channels from ℓ̂ and v̂:
            z history (shift window), EWMA s² (λ = 0.94), RV windows (RV² = exp(2v̂)),
            momentum, drawdown, 52-week position, gap and range proxies (gap ← 0, range ← from v̂)
        hold non-price channels (volume, delivery, OI, fundamentals, globals, macro) at last value
        set forcing rollout_step = k
        record ℓ̂^{m,k}, v̂^{m,k}
    monitor path statistics: share of paths with any |cumulative return| > 60% in 20 days must be < 0.1%
```
Training: single-step pretraining, then unrolled fine-tuning on K = 5 then K = 10 steps (backprop through samples; they are deterministic functions of ε), with gradient checkpointing. Global nodes are held flat in v1; a v2 option adds global-node heads.

## Part 3: Answer

### L9 Decision and explanation layer (deterministic)
Given paths ℓ̂ ∈ R^{M×H×N} and portfolio value weights w:
```
P^m_{i,k} = exp(Σ_{j≤k} ℓ̂^m_{i,j})                    # price relative
R^m_k     = Σ_i w_i·(P^m_{i,k} − 1)                    # buy-and-hold portfolio return
outcome range = quantiles of R^m_H (5/50/95)
VaR_q = −Quantile_{1−q}(R_H);   ES_q = −mean(R_H | R_H ≤ −VaR_q)
P(drawdown ≥ x) = mean_m[ min_k R^m_k ≤ −x ]
Euler ES contribution_i = mean(w_i·(P^m_{i,H} − 1) | tail) / mean(R_H | tail)
filtered views: recompute everything on the subset of paths meeting a filter (e.g. Nifty 20-day < −5%)
```
Mutual funds and ETFs: map to holdings from monthly portfolio disclosures (later); unsupported assets are flagged, never silently dropped. The language layer gets only the JSON (01 §7) and a validator rejects any text whose numbers do not match.

## Training recipe

| Stage | Data | Loss | Notes |
|---|---|---|---|
| S1 single-step pretrain | all train dates × training stocks | weighted fair CRPS on z and v (+ trend log loss weight 0.1) | 2 samples per input (FGN found 2 enough) |
| S2 event co-train (P5) | + event table | + event head losses | |
| S3 unrolled fine-tune | K = 5, then 10 | mean over steps of the S1 loss | lower lr ×0.1 |
| S4 seeds | repeat S1–S3 for 4 seeds | | |
| S5 rolling refit | quarterly, expanding window | | track skill decay between refits |

Optimizer: AdamW, peak lr 3e−4, 1,000 warm-up steps, cosine decay to 3e−6, weight decay 0.05, grad clip 1.0, bf16 autocast, EMA of weights 0.999. A batch is 2 decision dates × 2 noise samples (one sample is the whole market graph). Early stopping on validation CRPS with patience of 5 evaluations.

Parameter budget: encoder ≈ 0.8 M; processor ≈ 1.2 M per layer (3 attention passes + MLP) ≈ 10–13 M at 8–10 layers; heads and FGN maps < 1 M. Total ≈ 11–15 M, roughly a tenth of GraphCast.

## Fair CRPS (training form, per scalar target, M samples)
```
CRPS_fair(x^{1..M}, y) = (1/M)·Σ_m |x^m − y|  −  1/(2M(M−1))·Σ_{m≠m'} |x^m − x^{m'}|
L = Σ_{i,t} mask_{i,t}·w_{i,t}·[ CRPS_fair(ẑ_{i,t}, z_{i,t}) + α_v·CRPS_fair(v̂_{i,t}, v_{i,t}) ] / Σ mask·w
```
α_v (default 1.0) and the trend-loss weight are set on validation years.
