# Assignment 3: Optimization 2 — Experimental Report

Official handout writeup (PDF #1, ≤20 pages).  
Label discipline: **MEASURED** = numbers from `media/p1/*` or `media/p2/*`; **FITTED** = curves on that grid (never “measured”); **PREDICTION** = locked pre-run bands; **HIT/MISS** only after MEASURED.  
Configs / full cells → [experimental appendix](a3-experimental-appendix.md). Mechanism drill → intuition guide (not duplicated here).

## Setup

- d8 transformer, next-token prediction; val loss ↓ better; AdamW unless overridden (P3 full-batch GD; P4(c) SGDM).
- ~614M tokens unless a helper scales down.
- Starter: `experiments/a3_optimization/` (`a3_release` @ `57c60bb`); Modal `cs312-daxton`; W&B `daxgute-stanford-university` / `assignments`.
- Batch 1 (P1): prefix → branches → measure **DONE** (30/30 cells in `media/p1/a3-mode-connectivity.json`).
- Batch 2 (P2+P3): **9/9 DONE** (`media/p2/batch2_finals.csv` / `batch2_final_metrics.json`).

---

## Problem 1: Mode Connectivity

**Config:** `wsd0.0`, peak LR `0.003`, \(x\in\{0,50,100,250,500,1000\}\), \(y\in\{25,50,100,250,500\}\), `batch_order_seed∈\{1,2\}`. Details → appendix §P1.

**Mechanism.** Under constant post-warmup LR, late training keeps large bounce amplitude (river-valley; P2). Forking copies weights + Adam state at step \(x\); different batch orders then accumulate wall-orbit disagreement even when endpoint losses match. Longer shared prefix synchronizes trajectories before the fork, so midpoint weight-average stays in-basin; short \(x\) + long \(y\) is where the midpoint ridge appears.

**Prefix (**MEASURED**):** `model-d8-lr0.003-tok614M-wsd0.0-stop1000` / W&B `rq1k0bz1` — end train `3.43004`, val `3.48832` (1000 steps).  
**Grid source:** `media/p1/barrier-table.csv`, `barrier-summary.json`, `a3-mode-connectivity.json`.

### (a) Endpoints + ensemble

Endpoints \(L(\theta_1),L(\theta_2)\) and function-space ensemble \(L(p_{\mathrm{ens}})\) for all 30 cells → appendix / `barrier-table.csv`.

| Cell | \(L_1\) | \(L_2\) | \(L_{\mathrm{ens}}\) | \(L(\theta_{\mathrm{avg}})\) |
|------|---------|---------|----------------------|------------------------------|
| \((0,500)\) | 3.7511 | 3.7483 | 3.6454 | 4.3734 |
| \((1000,25)\) | 3.4786 | 3.4796 | 3.4221 | 3.4194 |

Where \(B>0.01\), ensemble sits well below weight-average — parameter midpoint is bad while averaged *predictions* remain good.

![Ensemble vs weight-average; \(G\) vs \(B\) (**MEASURED** points + **FITTED** Fit C)](../media/p1/plots/ensemble_vs_avg.png)

### (b) Barrier \(B\)

\(B=L(\theta_{\mathrm{avg}})-\tfrac{L(\theta_1)+L(\theta_2)}{2}\) at \(\alpha=0.5\).

**MEASURED** (`barrier-summary.json`): \(B_{\min}=-0.1161\) @ \((250,100)\); \(B_{\max}=0.6237\) @ \((0,500)\); mean \(B=-0.0334\); **26/30** cells with \(B\le 0.01\).  
Failures (\(B>0.01\)): \((0,250)=0.2173\), \((0,500)=0.6237\), \((50,500)=0.1299\), \((100,500)=0.0319\). Every \(x\ge 250\) cell has \(B\le 0.01\).

| \(x\) \\ \(y\) | 25 | 50 | 100 | 250 | 500 |
|---------------|------|------|------|------|------|
| 0 | −0.0288 ✓ | −0.0416 ✓ | −0.0275 ✓ | 0.2173 | 0.6237 |
| 50 | −0.1009 ✓ | −0.1053 ✓ | −0.0910 ✓ | 0.0055 ✓ | 0.1299 |
| 100 | −0.0932 ✓ | −0.1019 ✓ | −0.0972 ✓ | −0.0512 ✓ | 0.0319 |
| 250 | −0.0963 ✓ | −0.1106 ✓ | −0.1161 ✓ | −0.0779 ✓ | −0.0522 ✓ |
| 500 | −0.0747 ✓ | −0.0915 ✓ | −0.0953 ✓ | −0.0910 ✓ | −0.0763 ✓ |
| 1000 | −0.0597 ✓ | −0.0769 ✓ | −0.0855 ✓ | −0.0855 ✓ | −0.0825 ✓ |

![MEASURED \(B(x,y)\) heatmap](../media/p1/plots/B_heatmap.png)

### FITTED (Batch 1 grid only — not MEASURED)

| ID | Relation | Form | \(R^2\) | Domain |
|----|----------|------|---------|--------|
| Fit A growth | \(B\) vs \(y\) @ \(x=0\) | \(B=-0.190+0.001628\,y\) | \(\approx 1\) | \(x=0\), \(y\ge 100\) |
| Fit B | \(B\) vs \(x\) @ \(y=500\) | \(B=-0.066+0.687\,e^{-0.0232\,x}\) | 0.995 | \(y=500\), \(x\le 1000\) |
| Fit C | \(G\) vs \(B\) | \(G=0.090+1.033\,B\) | 0.989 | all 30 cells |

Identity: \(G=B+(\bar L-L_{\mathrm{ens}})\); MEASURED \(\bar L-L_{\mathrm{ens}}\) mean \(\approx 0.089\) ≈ constant → \(G\approx B+0.09\). Full coeffs → intuition-guide Appendix Q1.1 / `internal/p1-scaling-fit-status.md`.

![MEASURED \(B\) vs \(y\) + FITTED](../media/p1/plots/B_vs_y_curves.png)

![MEASURED \(B\) vs \(x\) @ \(y=500\) + FITTED](../media/p1/plots/B_vs_x_y500.png)

Out-of-grid bands (`P1-B-x0-y750`, `P1-B-x25-y500`, …) remain **PREDICTION** / Kind=`extrapolated` → [ledger](a3-prediction-ledger.md).

### (c) Interpolation curves

Every cell has \(\alpha\in\{0,0.25,0.5,0.75,1\}\) in `a3-mode-connectivity.json`.

| Cell | \(L(\theta(\alpha))\) | \(B\) | \(L_{\mathrm{ens}}\) | Shape |
|------|------------------------|-------|----------------------|-------|
| \((0,500)\) | [3.7511, 3.9156, 4.3734, 3.8961, 3.7483] | 0.6237 | 3.6454 | unimodal ridge @ 0.5 |
| \((50,500)\) | [3.7009, 3.7345, 3.8253, 3.7265, 3.6898] | 0.1299 | 3.5943 | milder ridge |
| \((1000,500)\) | [3.3812, 3.3170, 3.2986, 3.3172, 3.3810] | −0.0825 | 3.3017 | concave (no ridge) |

### (d) Same-basin controls

Ablations (LR / batch / schedule): **PENDING**.  
**Grid implication (**MEASURED**, `wsd0.0` only):** \(B>0.01\) only at small \(x\) + large \(y\); Fit B excess-barrier \(\sim e^{-0.023\,x}\) at \(y=500\) is the scorecard for future schedule ablations — not a Batch 2 result.

---

## Problem 2: River-Valley Dynamics

**Config / scorecard:** appendix §P2 · [scorecard](a3-batch2-prediction-scorecard.md).  
**MEASURED finals:** `media/p2/batch2_finals.csv` (jobs 0–6). Best raw val among schedule arms: linear \(L^\star=2.9282028675079346\). Regret \(R=L-L^\star\).

**Mechanism.** Decayed schedules shrink late bounce amplitude and let the iterate settle into the valley floor; constant `wsd0.0` keeps wall-bouncing through the full budget, so final val sits far above decayed peers. EMA / temporary dip / horizon-free arms probe whether averaging or budget-agnostic LR can recover that gap without a tuned decay.

| Job | Arm | W&B | Final val | Quantity (**MEASURED**) |
|-----|-----|-----|-----------|-------------------------|
| 1 | linear | `r0294p93` | 2.92820 | \(R=0\) (best) |
| 2 | cos | `sktw67q8` | 2.93412 | \(R=0.00592\) |
| 3 | `wsd0.2` | `soosjcxr` | 2.93774 | \(R=0.00954\) |
| 0 | `wsd0.0` | `widt420l` | 3.09751 | \(R=0.16930\) (worst) |
| 4 | `wsd0.2`+dip | `rms7iyyt` | 2.93899 | \(R_{\mathrm{dip}}=0.00125\) vs job 3 |
| 5 | invsqrt | `1jri34em` | 3.02046 | \(\Delta L=0.09225\) vs linear |
| 6 | SGDR \(T=2000\) | `51b3qq4n` | 3.03480 | \(\Delta L=0.10660\) vs linear |

### (a) Schedule comparison

**PREDICTION:** decayed ≻ constant; top-2 among `{linear, cos, wsd0.2}`; `wsd0.0` \(R\in[0.02,0.08)\).  
**MEASURED ranking:** linear \(<\) cos \(<\) `wsd0.2` \(≪\) `wsd0.0`.

| Ledger | PREDICTION | Observed | |
|--------|------------|----------|--|
| `P2-sched-rank` | \(\min L_{\mathrm{decayed}} < L_{\mathrm{wsd0.0}}\) | linear best; `wsd0.0` worst | **HIT** |
| `B2-P2-linear-R` | \(R\in[0,0.02)\) | \(0\) | **HIT** |
| `B2-P2-cos-R` | \(R\in[0,0.02)\); top-2 | \(0.00592\); #2 | **HIT** |
| `B2-P2-wsd0.2-R` | \(R\in[0,0.02)\); top-2 | \(0.00954\); #3 | band **HIT**; top-2 **MISS** |
| `B2-P2-wsd0.0-R` | \(R\in[0.02,0.08)\); worst | \(0.16930\); worst | band **MISS**; worst **HIT** |

Constant-LR regret was under-predicted by ~\(2\times\) the upper band edge — bounce amplitude under `wsd0.0` dominates. LR/train/val plots: **PENDING** (appendix empty).

### (b) Eval-only EMA

**PREDICTION:** gap-closed fraction \(f\in[0.5,0.75)\) (`P2-ema-gap-frac`).  
**Results:** **PENDING** — need EMA series export; do not invent.

### (c) Temporary LR dip

Dip `lr_dip=(0.4,0.45,0.5,0.1)` on `wsd0.2`; control = job 3.  
**PREDICTION:** final \(R\in[-0.02,0.02)\); mid-dip \(R_{\mathrm{mid}}<-0.02\).  
**MEASURED:** \(R_{\mathrm{dip}}=0.00125\in[-0.02,0.02)\) → `B2-P2-dip-R` **HIT** (temporary dip does not change the final). Mid-dip: **PENDING**.

### (d) Horizon-free schedules

**PREDICTION:** \(\Delta L=L_{\mathrm{hf}}-L_{\mathrm{linear}}\in[0.01,0.08)\); \(\lvert L_{\mathrm{sgdr}}-L_{\mathrm{invsqrt}}\rvert\in[0,0.03)\).  
**MEASURED:** invsqrt \(\Delta L=0.09225\) **MISS**; SGDR \(\Delta L=0.10660\) **MISS**; closeness \(0.01435\) **HIT**. Horizon-free arms track each other but both sit ~0.1 above the tuned linear floor.

---

## Problem 3: The Edge of Stability

Config → appendix §P3. Batch 2 jobs 7–8 **MEASURED** via `media/p2/p3_*_sharpness_summary.json`.

**Mechanism.** Full-batch GD with fixed \(\eta\) drives \(\lambda_{\max}\) toward \(\sim 2/\eta\) (progressive sharpening → edge). Minibatch AdamW instead tracks a pre-update product \(\eta\lambda^{\mathrm{pre}}\) that can sit far above 2 while still training — a different “edge” diagnostic than SGD’s \(2/\eta\).

### (a) Full-batch (Cohen-style)

Clean case \(N=32\), \(\eta=0.01\) (job 7 / W&B `0dewkykj`). η-halve/double @ 750: **PENDING** (not in Batch 2 nine).

| Quantity | PREDICTION | MEASURED | |
|----------|------------|----------|--|
| median \(\lambda_{\max}\) steps 1200–1500 | `[150,250)` | **245.16** (`p3_sgd_sharpness_summary.json`) | **HIT** (near \(2/\eta=200\)) |

Late train loss still decreases (final train `1.26862` @ 1499; val `8.05371` — small-\(N\) overfitting, not the edge metric). Crossing ~step 375; plateau \(\eta\lambda\approx 2.55\) (`summarize_raw`).

### (b) Minibatch AdamW

Job 8 / `foew12y0` (val `3.23971` @ 2999). Primary: post-warmup median \(\eta\lambda^{\mathrm{pre}}\).

| Quantity | PREDICTION | MEASURED | |
|----------|------------|----------|--|
| post-warmup median \(\eta\lambda^{\mathrm{pre}}\) | `[30,50)` | **30.99** (`p3_adamw_sharpness_summary.json`) | **HIT** (near 38) |

### (c) Open exploration (~18 runs)

**PENDING** — no Batch 2/3 exploratory arms scored here.

---

## Problem 4: Inside the Hessian

Config → appendix §P4. Batch 3 stage `a` **DONE 13/13** (`ap-DAUDZsFRech705uMDN7q3x`). Full cells → `media/p4/stage-a/`; plot `media/p4/p4_lambda_max_early.png`.

**Trajectory (**MEASURED** 13/13).** Collapse by 15 → dip \(\sim 2.195\) @1526 (earlyfit **HIT**) → rebound to \(\lambda@9375=5.967310256347902\) (**inside** handout `[4.5,6.0]`). Norm mid→final **HIT**. Block-energy **HIT** (excess_final=0.08727119221832025; fam_above=True; layer_above=True). Rebound: `internal/p4-lambda-rebound-13of13.md`. Scores: `internal/p4-9375-hit-miss-summary.md`.

**MISS mechanisms (predictive — locked PREDICT kept; not post-hoc rewrites).** Pre-registered stories vs file-exact fails (`internal/p4-9375-hit-miss-summary.md`):
- **`P4-a-family-sharpest`:** PREDICT late sharpest ∈ {`attention`,`mlp`} (residual-block sharpening; embed/head early-only). MEASURED late=`head` (\(5.641603135007765\)); early_top2=`embedding`,`attention`; ratio \(7.38436579244244\notin[1.5,5)\). Residual-block late locus **failed** — head stayed sharpest at 9375.
- **`P4-a-density-neg`:** PREDICT neg mass **shrinks** 0→9375 (saddle cleanup) with final ∈`[0,0.15)`. MEASURED \(0.004510114309467541\to\cdots\to0.06965100420936976\); final band OK but shrink_0→9375=**false**. Cleanup direction **failed** on this AdamW path.
- **`P4-lam-9375-earlyfit`:** PREDICT \(\lambda@9375\in[0.3,4.0)\) from early decay / preferred asymptote \(\sim 3.51\). MEASURED \(5.967310256347902\notin[0.3,4.0)\) but **inside** handout `[4.5,6.0]`. Earlyfit envelope undershot the rebound into the handout edge band.

Handout finals (given): \(\lambda_{\max}\in[4.5,6.0]\), val \(\in[2.925,2.931]\). PREDICT bands unchanged.

### (a) Measure provided AdamW ckpts

**Results:** Stage-a **MEASURED 13/13**. Copied from dry-run DRAFT_* (file-exact).

| Ledger ID | PREDICTION (locked) | Observed |
|-----------|---------------------|----------|
| `P4-a-family-sharpest` | Late: attention or mlp; sharpest/median final `[1.5,5)` | **MISS** — late=`head` (\(5.641603135007765\)); early_top2=`embedding`,`attention`; ratio=\(7.38436579244244\) |
| `P4-a-norm-sharpness` | Nondecreasing; final \(\lambda_{\max}\) in `[4.5,6.0]` | **HIT** — \(\lambda\) \(2.1954286466134665\to5.967310256347902\); \(\mathrm{tr}(H^2)/d\) \(8.856225326736765\mathrm{e}{-7}\to5.364973956804868\mathrm{e}{-6}\) |
| `P4-a-density-neg` | Neg mass shrinks; final neg frac `[0,0.15)` | **MISS** — neg_frac \(0.00451\to0.01491\to0.93446\to0.06965100420936976\); shrink_0→9375=false (final>init); final band OK |
| `P4-a-block-energy` | Family within-frac > random; excess `[0.05,0.30)` | **HIT** — family_within-random excess_final=0.08727119221832025; fam_above=True; layer_above=True |
| `P4-lam-1526-earlyfit` | \(\lambda@1526\in[1.4,4.0)\) | **HIT** — \(2.1954286466134665\) |
| `P4-lam-9375-earlyfit` | \(\lambda@9375\in[0.3,4.0)\) | **MISS** — \(5.967310256347902\) (inside handout `[4.5,6.0]`) |

### (b) Move \(\lambda_{\max}\) without moving the loss

**Results:** **MEASURED** (hand-rescale after-arm scored; see Observed).

**Baseline (**MEASURED** `media/p4/b-rescale-reference.json`):** \(\lambda_{\max}=5.967310256347902\), `probe_loss`\(=2.9474442899227142\), `scale={}` (unscaled AdamW@9375; matches stage-a final). Soft only — not an invar score.

**Hand-rescale:** after-arm scored from `internal/p4-b-score-dryrun.md` — `P4-invar-dL` **MISS** (|ΔL|=2.73853462934494; λ ratio=0.4658784038724411; λ 5.967310256347902→2.780040977639008; probe_loss 2.9474442899227142→5.685978919267654).

| Ledger ID | PREDICTION (locked) | Observed |
|-----------|---------------------|----------|
| `P4-invar-dL` | \(\lvert\Delta L_{\mathrm{val}}\rvert\in[0,0.002)\); \(\lambda\) ratio \(\ge 2\) or \(\le 0.5\) | **MISS** — `|ΔL|`=2.73853462934494; λ ratio=0.4658784038724411; λ 5.967310256347902→2.780040977639008 |
| `P4-b-train-2x` | \(\lambda\) factor \(\ge 2\times\); val in `[2.925,2.931]` | — |
| `P4-b-invar-vs-train` | Hand \(\lvert\Delta L\rvert\) < successful train-arm \(\lvert\Delta L\rvert\) | — |

### (c) SGDM top subspace

**Results:** **MEASURED**.

| Ledger ID | PREDICTION (locked) | Observed |
|-----------|---------------------|----------|
| `P4-top10-frac` | Mean \(\|P_{10}g\|/\|g\|\in[0.6,0.85)\) | **MISS** — mean top_frac=0.09296940652964016 |
| `P4-c-loss-rank` | full ≈ top < removed; gap `[0.02,0.20)` | **MISS** — L_full=3.582527816295624; L_top=5.502330541610718; L_removed=3.5821831226348877; gap=-0.000344693660736084 |
| `P4-c-k-ablate` | Optional \(k\in\{3,30\}\) bands | — |
| `P4-c-stale-evecs` | Optional stale refresh extra loss `[0,0.05)` | — |

---

## Problem 5 (optional): Own prediction problem

**Status:** **MEASURED** — `P5-B-cool-x0` **MISS**; `P5-dB-cool` **MISS**; `P5-B-const-ctrl` **HIT**; `P5-rank-arms` **MISS**; `P5-G-cool-fitC` **HIT**; `P5-emergent-basin` **MISS**; `P5-L-cool-dir` **INSUFFICIENT** (from `media/p5/` / `internal/p5-score-dryrun.md`). Practice quiz **Q25**.

**Phenomenon:** after early fork under `wsd0.0` already yields large \(B\), can a short matched cooldown collapse \(B\le 0.01\) (same valley walls) vs constant-LR control (true modes)?

**Baseline (**MEASURED** P1 only):** \((0,500)\) \(B=0.6237\); \((50,500)\) \(B=0.1299\); \((1000,500)\) \(B=-0.0825\).

| ID | Quantity | PREDICTION |
|----|----------|------------|
| `P5-B-cool-x0` | \(B_C\) after cooldown | \((-\infty,0.01]\) | **MISS** — B_C=0.7697049379348755 |
| `P5-dB-cool` | \(B_R-B_C\) (\(B_R=0.6237\)) | \([0.5,0.8)\) | **MISS** — dB=-0.14600493793487546 (B_R=0.6237 [JSON] − B_C=0.7697049379348755) |
| `P5-B-const-ctrl` | \(B_K\) after +100 constant LR | \([0.5,\infty)\) | **HIT** — B_K=0.7320129871368408 |
| `P5-rank-arms` | Ranking | \(B_C < B_R\) and \(B_C < B_K\) | **MISS** — B_C=0.7697049379348755 B_R=0.6237 B_K=0.7320129871368408 |
| `P5-emergent-basin` | Forced choice | Same-valley if \(B_C\le 0.01\) and \(B_K>0.01\) | **MISS** — B_C=0.7697049379348755 B_K=0.7320129871368408 → not-same-valley (JSON emergent='true-modes') |
Code: `experiments/a3_optimization/p5_cooldown_barrier.py`. Observed: **MEASURED** — `P5-B-cool-x0` **MISS**; `P5-dB-cool` **MISS**; `P5-B-const-ctrl` **HIT**; `P5-rank-arms` **MISS**; `P5-G-cool-fitC` **HIT**; `P5-emergent-basin` **MISS**; `P5-L-cool-dir` **INSUFFICIENT**. Media: `media/p5/`.

**MISS mechanisms (predictive — locked PREDICT kept; not post-hoc rewrites).** File-exact from `media/p5/` / `internal/p5-score-dryrun.md`:
- **Cooldown did not collapse the barrier:** `P5-B-cool-x0` PREDICT \(B_C\le 0.01\) → MEASURED B_C=0.7697049379348755 (**MISS**); `P5-dB-cool` PREDICT \(\Delta B\in[0.5,0.8)\) → dB=-0.14600493793487546 (**MISS**, negative — \(B_C>B_R\)); `P5-rank-arms` PREDICT \(B_C < B_R\) and \(B_C < B_K\) → **MISS**. Short matched cooldown failed to create connectivity on the worst P1 cell.
- **Constant-LR control stayed high-\(B\):** `P5-B-const-ctrl` PREDICT \(B_K\ge 0.5\) → B_K=0.7320129871368408 (**HIT**). Continuing constant LR did not invent a valley — control arm matches keep-bouncing / high-\(B\).
- **Fit C held:** `P5-G-cool-fitC` → G_C=0.8391239460323332 with |G−(B+0.09)|=0.020580991902542256 (**HIT**). Ensemble–avg gap still tracks \(G\approx B+0.09\) even while the barrier stays large.
- **Emergent basin:** PREDICT same-valley walls → JSON emergent='true-modes' / not-same-valley (**MISS** `P5-emergent-basin`). With \(B_C\) and \(B_K\) both ≫0.01, the forced choice is true disconnected modes, not opposite walls of one valley.
- **`P5-L-cool-dir`:** **INSUFFICIENT** (dL null / \(L_R,L_C\) missing) — no direction takeaway until endpoints land.

---

## Interpretation summary

**P1 (**MEASURED** + FITTED).** Weight averaging works on 26/30 cells. Failures = small-\(x\), large-\(y\) under constant LR; worst \(B=0.6237\) at \((0,500)\). There, ensemble still wins (\(G\) up to 0.7281). Fits A–C summarize growth in \(y\), exponential collapse in shared \(x\), and \(G\approx B+0.09\).

**P2 (**MEASURED** 0–6).** Decayed schedules beat constant (**HIT** rank); `wsd0.0` regret \(0.169\) under-predicted (**MISS** band). Dip final ≈0 (**HIT**). Horizon-free \(\Delta L\sim 0.09\)–\(0.11\) vs linear (**MISS** bands) but track each other (**HIT** closeness). EMA / mid-dip still **PENDING**.

**P3 (**MEASURED** jobs 7–8).** SGD late \(\lambda_{\max}=245\) → **HIT** `[150,250)`. AdamW post-warmup \(\eta\lambda^{\mathrm{pre}}=31.0\) → **HIT** `[30,50)`. η-change / ~18-run exploration **PENDING**.

**P4 (**MEASURED** 13/13 + b baseline).** Rebound to \(\lambda@9375=5.967310256347902\) (**inside** handout `[4.5,6.0]`). Predictive MISSes kept locked bands: late sharpest=`head` not attn\|mlp; dens-neg no shrink 0→9375; earlyfit@9375 rebound past `[0.3,4.0)` into handout. Norm **HIT**; block-energy **HIT** (excess_final=0.08727119221832025). P4(b): reference baseline MEASURED (`scale={}`); hand-rescale scored — `P4-invar-dL` **MISS** (|ΔL|=2.73853462934494; ratio=0.4658784038724411). b/c: `P4-top10-frac` **MISS** (mean=0.09296940652964016); `P4-c-loss-rank` **MISS** (gap=-0.000344693660736084). **P5:** MEASURED — cooldown did not collapse barrier (`P5-B-cool-x0` / `P5-dB-cool` / `P5-rank-arms` / `P5-emergent-basin` **MISS**; emergent='true-modes'); constant-LR control stayed high-\(B\) (`P5-B-const-ctrl` **HIT** B_K=0.7320129871368408); Fit C held (`P5-G-cool-fitC` **HIT**); `P5-L-cool-dir` **INSUFFICIENT**.

Calibration: [scorecard](a3-batch2-prediction-scorecard.md), [ledger](a3-prediction-ledger.md). Full tables: [appendix](a3-experimental-appendix.md).
