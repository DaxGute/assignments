# Assignment 3: Results, Intuition & Quantitative Properties

CS 312 · Optimization 2 · **Hand-in PDF #1** (of 2).  
Companion practice quiz → [`a3-practice.md`](a3-practice.md) / `media/pdfs/a3-practice.pdf`.

Label discipline: **MEASURED** = numbers from `media/p1/*`, `media/p2/*`, `media/p4/*`, `media/p5/*`; **FITTED** = curves on that grid (never “measured”); **PREDICTION** = locked pre-run bands; **HIT/MISS** only after MEASURED. Every float below is copied file-exact from store evidence — no invention.

Consolidated from: assignment report · experimental appendix highlights · predictive intuition · quantitative prediction reference. Older five-PDF split is **superseded** (see [`a3-deliverables-readme.md`](a3-deliverables-readme.md)).

## Setup

- d8 transformer, next-token prediction; val loss ↓ better; AdamW unless overridden (P3 full-batch GD; P4(c) SGDM).
- ~614M tokens unless a helper scales down.
- Starter: `experiments/a3_optimization/` (`a3_release` @ `57c60bb`); Modal `cs312-daxton`; W&B `daxgute-stanford-university` / `assignments`.
- Batch 1 (P1): **DONE** 30/30 cells in `media/p1/a3-mode-connectivity.json`.
- Batch 2 (P2+P3): **9/9 DONE** (`media/p2/batch2_finals.csv`).
- Batch 3 (P4): stage-a **13/13** + b/c scored; P5 cooldown **MEASURED**.

---

## Problem 1: Mode Connectivity

**Config:** `wsd0.0`, peak LR `0.003`, \(x\in\{0,50,100,250,500,1000\}\), \(y\in\{25,50,100,250,500\}\), `batch_order_seed∈\{1,2\}\).

**Mechanism / intuition.** Under constant post-warmup LR, late training keeps large bounce amplitude (river-valley; P2). Forking copies weights + Adam state at step \(x\); different batch orders accumulate wall-orbit disagreement even when endpoint losses match. Longer shared prefix synchronizes trajectories before the fork, so midpoint weight-average stays in-basin; short \(x\) + long \(y\) is where the midpoint ridge appears. Ensemble averaging is in **function space** — two nets can disagree in \(\theta\) yet agree on soft predictions, so \(L(p_{\mathrm{ens}})\) can stay low while \(L(\theta_{\mathrm{avg}})\) sits on a ridge.

**Prefix (**MEASURED**):** `model-d8-lr0.003-tok614M-wsd0.0-stop1000` / W&B `rq1k0bz1` — end train `3.43004`, val `3.48832` (1000 steps).  
**Grid source:** `media/p1/barrier-table.csv`, `barrier-summary.json`, `a3-mode-connectivity.json`.

### (a) Endpoints + ensemble

| Cell | \(L_1\) | \(L_2\) | \(L_{\mathrm{ens}}\) | \(L(\theta_{\mathrm{avg}})\) |
|------|---------|---------|----------------------|------------------------------|
| \((0,500)\) | 3.7511 | 3.7483 | 3.6454 | 4.3734 |
| \((1000,25)\) | 3.4786 | 3.4796 | 3.4221 | 3.4194 |

Where \(B>0.01\), ensemble sits well below weight-average — parameter midpoint is bad while averaged *predictions* remain good. MEASURED failure gaps \(G=L(\theta_{\mathrm{avg}})-L(p_{\mathrm{ens}})\): \((100,500)\,0.1311\); \((50,500)\,0.2309\); \((0,250)\,0.3273\); \((0,500)\,0.7281\); median \(G\approx 0.279\).

![Ensemble vs weight-average; \(G\) vs \(B\) (**MEASURED** points + **FITTED** Fit C)](../media/p1/plots/ensemble_vs_avg.png)

### (b) Barrier \(B\)

\[
B=L(\theta_{\mathrm{avg}})-\tfrac{L(\theta_1)+L(\theta_2)}{2},\qquad
\theta_{\mathrm{avg}}=\tfrac12(\theta_1+\theta_2)
\]
at \(\alpha=0.5\). Handout: averaging “works” if \(B\le 0.01\).

**MEASURED** (`barrier-summary.json`, file-exact): \(B_{\min}=-0.1160578727722168\) @ \((250,100)\); \(B_{\max}=0.6237311363220215\) @ \((0,500)\); mean \(B=-0.033404584725697836\); **26/30** cells with \(B\le 0.01\).  
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

### Quantitative properties (FITTED — Batch 1 grid only)

| ID | Relation | Form | \(R^2\) | Domain |
|----|----------|------|---------|--------|
| Fit A growth | \(B\) vs \(y\) @ \(x=0\) | \(B=-0.1901+0.001628\,y\) | \(\approx 1.000\) | \(x=0\), \(y\ge 100\) |
| Fit A quad | \(B\) vs \(y\) @ \(x=0\) | \(B=-0.0765+7.61\times10^{-4}\,y+1.30\times10^{-6}\,y^2\) | 0.991 | all five \(y\) |
| Fit B | \(B\) vs \(x\) @ \(y=500\) | \(B=-0.0663+0.6870\,e^{-0.02316\,x}\) | 0.995 | \(y=500\), six \(x\) |
| Fit C | \(G\) vs \(B\) | \(G=0.0897+1.0328\,B\) | 0.989 | all 30 cells |

Identity check: \(G=B+(\bar L-L_{\mathrm{ens}})\); MEASURED \(\bar L-L_{\mathrm{ens}}\) mean \(\approx 0.089\) → \(G\approx B+0.09\). Out-of-grid bands remain **PREDICTION** / Kind=`extrapolated`.

![MEASURED \(B\) vs \(y\) + FITTED](../media/p1/plots/B_vs_y_curves.png)

![MEASURED \(B\) vs \(x\) @ \(y=500\) + FITTED](../media/p1/plots/B_vs_x_y500.png)

### (c) Interpolation curves

Every cell has \(\alpha\in\{0,0.25,0.5,0.75,1\}\) in `a3-mode-connectivity.json`.

| Cell | \(L(\theta(\alpha))\) | \(B\) | \(L_{\mathrm{ens}}\) | Shape |
|------|------------------------|-------|----------------------|-------|
| \((0,500)\) | [3.7511, 3.9156, 4.3734, 3.8961, 3.7483] | 0.6237 | 3.6454 | unimodal ridge @ 0.5 |
| \((50,500)\) | [3.7009, 3.7345, 3.8253, 3.7265, 3.6898] | 0.1299 | 3.5943 | milder ridge |
| \((1000,500)\) | [3.3812, 3.3170, 3.2986, 3.3172, 3.3810] | −0.0825 | 3.3017 | concave (no ridge) |

### (d) Same-basin controls

Ablations (LR / batch / schedule): **PENDING**.  
**Grid implication (**MEASURED**, `wsd0.0` only):** \(B>0.01\) only at small \(x\) + large \(y\); Fit B excess-barrier \(\sim e^{-0.023\,x}\) at \(y=500\) is the scorecard for future schedule ablations.

---

## Problem 2: River-Valley Dynamics

**MEASURED finals:** `media/p2/batch2_finals.csv` (jobs 0–6). Best raw val: linear \(L^\star=2.9282028675079346\). Regret \(R=L-L^\star\).

**Mechanism / intuition.** Decayed schedules shrink late bounce amplitude and let the iterate settle into the valley floor; constant `wsd0.0` keeps wall-bouncing through the full budget, so final val sits far above decayed peers. EMA / temporary dip / horizon-free arms probe whether averaging or budget-agnostic LR can recover that gap without a tuned decay.

| Job | Arm | W&B | Final val (exact) | Quantity (**MEASURED**) |
|-----|-----|-----|-------------------|-------------------------|
| 1 | linear | `r0294p93` | 2.9282028675079346 | \(R=0\) (best) |
| 2 | cos | `sktw67q8` | 2.934119462966919 | \(R=0.005916595458984375\) |
| 3 | `wsd0.2` | `soosjcxr` | 2.93774151802063 | \(R=0.009538650512695312\) |
| 0 | `wsd0.0` | `widt420l` | 3.0975050926208496 | \(R=0.16930222511291504\) (worst) |
| 4 | `wsd0.2`+dip | `rms7iyyt` | 2.938987970352173 | \(R_{\mathrm{dip}}=0.0012464523315429688\) vs job 3 |
| 5 | invsqrt | `1jri34em` | 3.020456075668335 | \(\Delta L=0.09225320816040039\) vs linear |
| 6 | SGDR \(T=2000\) | `51b3qq4n` | 3.0348048210144043 | \(\Delta L=0.10660195350646973\) vs linear |

### (a) Schedule comparison

**PREDICTION:** decayed ≻ constant; top-2 among `{linear, cos, wsd0.2}`; `wsd0.0` \(R\in[0.02,0.08)\).  
**MEASURED ranking:** linear \(<\) cos \(<\) `wsd0.2` \(≪\) `wsd0.0`.

| Ledger | PREDICTION | Observed | |
|--------|------------|----------|--|
| `P2-sched-rank` | \(\min L_{\mathrm{decayed}} < L_{\mathrm{wsd0.0}}\) | linear best; `wsd0.0` worst | **HIT** |
| `B2-P2-linear-R` | \(R\in[0,0.02)\) | \(0\) | **HIT** |
| `B2-P2-cos-R` | \(R\in[0,0.02)\); top-2 | \(0.005916595458984375\); #2 | **HIT** |
| `B2-P2-wsd0.2-R` | \(R\in[0,0.02)\); top-2 | \(0.009538650512695312\); #3 | band **HIT**; top-2 **MISS** |
| `B2-P2-wsd0.0-R` | \(R\in[0.02,0.08)\); worst | \(0.16930222511291504\); worst | band **MISS**; worst **HIT** |

Constant-LR regret was under-predicted by ~\(2\times\) the upper band edge — bounce amplitude under `wsd0.0` dominates.

### (b) Eval-only EMA

**PREDICTION:** gap-closed fraction \(f\in[0.5,0.75)\) (`P2-ema-gap-frac`).  
**Results:** **PENDING** — need EMA series export; do not invent.

### (c) Temporary LR dip

Dip `lr_dip=(0.4,0.45,0.5,0.1)` on `wsd0.2`; control = job 3.  
**PREDICTION:** final \(R\in[-0.02,0.02)\); mid-dip \(R_{\mathrm{mid}}<-0.02\).  
**MEASURED:** \(R_{\mathrm{dip}}=0.0012464523315429688\in[-0.02,0.02)\) → `B2-P2-dip-R` **HIT**. Mid-dip: **PENDING**.

### (d) Horizon-free schedules

**PREDICTION:** \(\Delta L=L_{\mathrm{hf}}-L_{\mathrm{linear}}\in[0.01,0.08)\); \(\lvert L_{\mathrm{sgdr}}-L_{\mathrm{invsqrt}}\rvert\in[0,0.03)\).  
**MEASURED:** invsqrt \(\Delta L=0.09225320816040039\) **MISS**; SGDR \(\Delta L=0.10660195350646973\) **MISS**; closeness \(0.014348745346069336\) **HIT**. Horizon-free arms track each other but both sit ~0.1 above the tuned linear floor.

---

## Problem 3: The Edge of Stability

**Mechanism / intuition.** Full-batch GD with fixed \(\eta\) drives \(\lambda_{\max}\) toward \(\sim 2/\eta\) (progressive sharpening → edge). Minibatch AdamW instead tracks a pre-update product \(\eta\lambda^{\mathrm{pre}}\) that can sit far above 2 while still training — a different “edge” diagnostic than SGD’s \(2/\eta\). Handout thresholds (mechanism, not Observed): GD edge \(\eta\lambda=2\) ⇒ \(\lambda=200\) at \(\eta=0.01\); Adam \(\beta_1=0.9\) precond edge \(38\).

### (a) Full-batch (Cohen-style)

Clean case \(N=32\), \(\eta=0.01\) (job 7 / W&B `0dewkykj`). η-halve/double @ 750: **PENDING**.

| Quantity | PREDICTION | MEASURED | |
|----------|------------|----------|--|
| median \(\lambda_{\max}\) steps 1200–1500 | `[150,250)` | **245.16314737956807** | **HIT** (near \(2/\eta=200\)) |

Late train loss still decreases (final train `1.268615484237671` @ 1499; val `8.053705215454102` — small-\(N\) overfitting, not the edge metric). Crossing ~step 375; plateau \(\eta\lambda\approx 2.55\).

### (b) Minibatch AdamW

Job 8 / `foew12y0` (val `3.2397050857543945` @ 2999). Primary: post-warmup median \(\eta\lambda^{\mathrm{pre}}\).

| Quantity | PREDICTION | MEASURED | |
|----------|------------|----------|--|
| post-warmup median \(\eta\lambda^{\mathrm{pre}}\) | `[30,50)` | **30.987715914514** | **HIT** (near 38) |

### (c) Open exploration (~18 runs)

**PENDING** — no Batch 2/3 exploratory arms scored here.

---

## Problem 4: Inside the Hessian

**Mechanism / intuition.** Over AdamW training, \(\lambda_{\max}\) collapses from init, dips mid-training, then rebounds into the handout edge band `[4.5,6.0]`. Family-local sharpness can migrate (embed early → head late). Negative-eigenvalue mass need not monotonically shrink. Block energy tests whether Hessian mass concentrates in architectural families vs random subspaces. Hand-rescale can move \(\lambda\) but typically also moves loss (invariance hard). Top-\(k\) SGDM subspace may carry little of the update energy.

**Trajectory (**MEASURED** 13/13).** Collapse by 15 → dip \(2.1954286466134665\) @1526 (earlyfit **HIT**) → rebound to \(\lambda@9375=5.967310256347902\) (**inside** handout `[4.5,6.0]`). Block-energy **HIT** (excess_final=0.08727119221832025; fam_above=True; layer_above=True). Plot: `media/p4/p4_lambda_max_early.png`.

![P4 \(\lambda_{\max}\) over training (**MEASURED** 13/13)](../media/p4/p4_lambda_max_early.png)

### MEASURED \(\lambda_{\max}\) card

| Step | \(\lambda_{\max}\) (exact) |
|------|----------------------------|
| 0 | 1431.5030691763686 |
| 3 | 1571.7420262270407 |
| 15 | 53.36426571742344 |
| 61 | 18.054717868295125 |
| 153 | 6.502395300121604 |
| 305 | 4.32926630998023 |
| 763 | 3.02398152405625 |
| 1526 | 2.1954286466134665 |
| 3052 | 3.1637529312879873 |
| 4578 | 3.7423705126158477 |
| 6104 | 4.155164815834522 |
| 7630 | 5.281516516770698 |
| 9375 | 5.967310256347902 |

**Density (file-exact):** neg_frac \(0.004510114309467541\to 0.01491374251600709\to 0.9344596369432798\to 0.06965100420936976\) (steps 0→153→1526→9375). Final in `[0,0.15)` but **did not shrink** vs init → `P4-a-density-neg` **MISS**.

### (a) Scorecard

| Ledger ID | PREDICTION (locked) | Observed |
|-----------|---------------------|----------|
| `P4-a-family-sharpest` | Late: attention or mlp; ratio `[1.5,5)` | **MISS** — late=`head` (\(5.641603135007765\)); early_top2=`embedding`,`attention`; ratio=\(7.38436579244244\) |
| `P4-a-norm-sharpness` | Nondecreasing; final \(\lambda_{\max}\) in `[4.5,6.0]` | **HIT** — \(\lambda\) \(2.1954286466134665\to5.967310256347902\); \(\mathrm{tr}(H^2)/d\) \(8.856225326736765\mathrm{e}{-7}\to5.364973956804868\mathrm{e}{-6}\) |
| `P4-a-density-neg` | Neg mass shrinks; final `[0,0.15)` | **MISS** — final \(0.06965100420936976\); shrink_0→9375=false |
| `P4-a-block-energy` | Family within-frac > random; excess `[0.05,0.30)` | **HIT** — excess_final=0.08727119221832025; fam_above=True; layer_above=True |
| `P4-lam-1526-earlyfit` | \(\lambda@1526\in[1.4,4.0)\) | **HIT** — \(2.1954286466134665\) |
| `P4-lam-9375-earlyfit` | \(\lambda@9375\in[0.3,4.0)\) | **MISS** — \(5.967310256347902\) (inside handout `[4.5,6.0]`) |

**MISS mechanisms (predictive — locked PREDICT kept).** Late sharpest stayed `head` (not attn\|mlp). Density cleanup direction failed. Earlyfit envelope undershot the rebound into the handout edge band.

### (b) Move \(\lambda_{\max}\) without moving the loss

**Baseline (**MEASURED** `media/p4/b-rescale-reference.json`):** \(\lambda_{\max}=5.967310256347902\), `probe_loss`\(=2.9474442899227142\), `scale={}`.

| Ledger ID | PREDICTION (locked) | Observed |
|-----------|---------------------|----------|
| `P4-invar-dL` | \(\lvert\Delta L_{\mathrm{val}}\rvert\in[0,0.002)\); \(\lambda\) ratio \(\ge 2\) or \(\le 0.5\) | **MISS** — `|ΔL|`=2.73853462934494; λ ratio=0.4658784038724411; λ 5.967310256347902→2.780040977639008; probe_loss 2.9474442899227142→5.685978919267654 |

### (c) SGDM top subspace

| Ledger ID | PREDICTION (locked) | Observed |
|-----------|---------------------|----------|
| `P4-top10-frac` | Mean \(\|P_{10}g\|/\|g\|\in[0.6,0.85)\) | **MISS** — mean top_frac=0.09296940652964016 |
| `P4-c-loss-rank` | full ≈ top < removed; gap `[0.02,0.20)` | **MISS** — L_full=3.582527816295624; L_top=5.502330541610718; L_removed=3.5821831226348877; gap=-0.000344693660736084 |

---

## Problem 5 (optional): Own prediction problem

**Phenomenon:** after early fork under `wsd0.0` already yields large \(B\), can a short matched cooldown collapse \(B\le 0.01\) (same valley walls) vs constant-LR control (true modes)?

**Baseline (**MEASURED** P1):** \((0,500)\) \(B=0.6237\); \((50,500)\) \(B=0.1299\); \((1000,500)\) \(B=-0.0825\).

| ID | PREDICTION | Observed |
|----|------------|----------|
| `P5-B-cool-x0` | \(B_C\le 0.01\) | **MISS** — B_C=0.7697049379348755 |
| `P5-dB-cool` | \(B_R-B_C\in[0.5,0.8)\) | **MISS** — dB=-0.14600493793487546 (B_R=0.6237 − B_C=0.7697049379348755) |
| `P5-B-const-ctrl` | \(B_K\ge 0.5\) | **HIT** — B_K=0.7320129871368408 |
| `P5-rank-arms` | \(B_C < B_R\) and \(B_C < B_K\) | **MISS** — B_C=0.7697049379348755, B_R=0.6237, B_K=0.7320129871368408 |
| `P5-G-cool-fitC` | if \(B_C>0.01\): \(\lvert G-(B+0.09)\rvert<0.05\) | **HIT** — G_C=0.8391239460323332; \|G−(B+0.09)\|=0.020580991902542256 |
| `P5-emergent-basin` | Same-valley if \(B_C\le 0.01\) and \(B_K>0.01\) | **MISS** — JSON emergent='true-modes' |
| `P5-L-cool-dir` | \(L_C<L_R\); \(\Delta L\in[0.02,0.15)\) | **INSUFFICIENT** — dL null / \(L_R,L_C\) missing |

**Takeaway.** Short matched cooldown failed to create connectivity on the worst P1 cell; constant-LR control stayed high-\(B\); Fit C still held; forced choice is true disconnected modes, not opposite walls of one valley. Code: `experiments/a3_optimization/p5_cooldown_barrier.py`. Media: `media/p5/`.

---

## Interpretation summary

**P1 (**MEASURED** + FITTED).** Weight averaging works on 26/30 cells. Failures = small-\(x\), large-\(y\) under constant LR; worst \(B=0.6237311363220215\) at \((0,500)\). Ensemble still wins there (\(G\) up to 0.7281). Fits A–C summarize growth in \(y\), exponential collapse in shared \(x\), and \(G\approx B+0.09\).

**P2 (**MEASURED** 0–6).** Decayed schedules beat constant (**HIT** rank); `wsd0.0` regret \(0.16930222511291504\) under-predicted (**MISS** band). Dip final ≈0 (**HIT**). Horizon-free \(\Delta L\sim 0.09\)–\(0.11\) vs linear (**MISS** bands) but track each other (**HIT** closeness). EMA / mid-dip still **PENDING**.

**P3 (**MEASURED** jobs 7–8).** SGD late \(\lambda_{\max}=245.16314737956807\) → **HIT** `[150,250)`. AdamW post-warmup \(\eta\lambda^{\mathrm{pre}}=30.987715914514\) → **HIT** `[30,50)`. η-change / ~18-run exploration **PENDING**.

**P4 (**MEASURED** 13/13 + b/c).** Rebound to \(\lambda@9375=5.967310256347902\) (**inside** handout `[4.5,6.0]`). Predictive MISSes kept locked: late sharpest=`head`; dens-neg no shrink; earlyfit@9375 past `[0.3,4.0)`. Norm **HIT**; block-energy **HIT** (excess_final=0.08727119221832025). `P4-invar-dL` **MISS** (|ΔL|=2.73853462934494); `P4-top10-frac` **MISS** (mean=0.09296940652964016); `P4-c-loss-rank` **MISS** (gap=-0.000344693660736084).

**P5.** Cooldown did not collapse barrier (`P5-B-cool-x0` / `P5-dB-cool` / `P5-rank-arms` / `P5-emergent-basin` **MISS**); constant-LR control stayed high-\(B\) (**HIT**); Fit C held (**HIT**); `P5-L-cool-dir` **INSUFFICIENT**.

---

## Compact quantitative card (all MEASURED / FITTED / scored)

### P2 scored counterfactuals

| ID | PREDICTION | Observed (exact) | Score |
|----|------------|------------------|-------|
| `B2-P2-dip-R` | \([-0.02,0.02)\) | \(0.0012464523315429688\) | **HIT** |
| `B2-P2-invsqrt-dL` | \([0.01,0.08)\) | \(0.09225320816040039\) | **MISS** |
| `B2-P2-sgdr-dL` | \([0.01,0.08)\) | \(0.10660195350646973\) | **MISS** |
| \|SGDR−inv\| | \([0,0.03)\) | \(0.014348745346069336\) | **HIT** |

### PENDING (do not invent)

| Topic | Status |
|-------|--------|
| Mid-dip \(R_{\mathrm{mid}}\) | **PENDING** |
| EMA gap (`P2-ema-gap-frac`) | **PENDING** |
| P3 η-change / P3(c) | **PENDING** |
| `P5-L-cool-dir` | **INSUFFICIENT** (dL=null) |

### Appendix highlight — P1 full cell table (**MEASURED**)

| x | y | L(θ₁) | L(θ₂) | L_ens | L(θ_avg) | B |
|---|---|-------|-------|-------|----------|---|
| 0 | 25 | 6.4380 | 6.4653 | 6.4077 | 6.4228 | −0.0288 ✓ |
| 0 | 50 | 5.6329 | 5.6864 | 5.5798 | 5.6180 | −0.0416 ✓ |
| 0 | 100 | 5.0282 | 5.0396 | 4.9238 | 5.0064 | −0.0275 ✓ |
| 0 | 250 | 4.3202 | 4.3092 | 4.2046 | 4.5319 | 0.2173 |
| 0 | 500 | 3.7511 | 3.7483 | 3.6454 | 4.3734 | 0.6237 |
| 50 | 25 | 5.2708 | 5.2877 | 5.1962 | 5.1784 | −0.1009 ✓ |
| 50 | 50 | 5.0453 | 5.0449 | 4.9441 | 4.9398 | −0.1053 ✓ |
| 50 | 100 | 4.6943 | 4.6940 | 4.5872 | 4.6031 | −0.0910 ✓ |
| 50 | 250 | 4.1278 | 4.1245 | 4.0231 | 4.1316 | 0.0055 ✓ |
| 50 | 500 | 3.7009 | 3.6898 | 3.5943 | 3.8253 | 0.1299 |
| 100 | 25 | 4.8268 | 4.8219 | 4.7374 | 4.7311 | −0.0932 ✓ |
| 100 | 50 | 4.6870 | 4.6719 | 4.5790 | 4.5776 | −0.1019 ✓ |
| 100 | 100 | 4.4887 | 4.4439 | 4.3591 | 4.3691 | −0.0972 ✓ |
| 100 | 250 | 3.9961 | 3.9747 | 3.8808 | 3.9342 | −0.0512 ✓ |
| 100 | 500 | 3.6670 | 3.6582 | 3.5634 | 3.6945 | 0.0319 |
| 250 | 25 | 4.2400 | 4.2449 | 4.1581 | 4.1461 | −0.0963 ✓ |
| 250 | 50 | 4.1383 | 4.1451 | 4.0505 | 4.0311 | −0.1106 ✓ |
| 250 | 100 | 3.9972 | 3.9767 | 3.8869 | 3.8709 | −0.1161 ✓ |
| 250 | 250 | 3.7429 | 3.7357 | 3.6447 | 3.6615 | −0.0779 ✓ |
| 250 | 500 | 3.5740 | 3.5712 | 3.4844 | 3.5204 | −0.0522 ✓ |
| 500 | 25 | 3.7266 | 3.7154 | 3.6522 | 3.6463 | −0.0747 ✓ |
| 500 | 50 | 3.7105 | 3.6971 | 3.6226 | 3.6123 | −0.0915 ✓ |
| 500 | 100 | 3.6654 | 3.6611 | 3.5798 | 3.5680 | −0.0953 ✓ |
| 500 | 250 | 3.5767 | 3.5748 | 3.4918 | 3.4847 | −0.0910 ✓ |
| 500 | 500 | 3.4862 | 3.4857 | 3.4037 | 3.4097 | −0.0763 ✓ |
| 1000 | 25 | 3.4786 | 3.4796 | 3.4221 | 3.4194 | −0.0597 ✓ |
| 1000 | 50 | 3.4730 | 3.4757 | 3.4051 | 3.3974 | −0.0769 ✓ |
| 1000 | 100 | 3.4619 | 3.4611 | 3.3867 | 3.3760 | −0.0855 ✓ |
| 1000 | 250 | 3.4241 | 3.4225 | 3.3449 | 3.3378 | −0.0855 ✓ |
| 1000 | 500 | 3.3812 | 3.3810 | 3.3017 | 3.2986 | −0.0825 ✓ |

Full configs / W&B IDs / family-\(\lambda\) tables remain in source MD (`a3-experimental-appendix.md`) — not re-dumped here.
