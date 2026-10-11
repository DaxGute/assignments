# A3 Predictive Intuition Guide

**Primary learning resource** for Assignment 3 prediction mastery. Organized by handout problem (P1–P5), with a quantitative appendix merged in so this is the one comprehensive study PDF.

### PDF / doc policy (max 2–3 final PDFs)

| # | Final PDF candidate | Source in store |
|---|---------------------|-----------------|
| 1 | Assignment report (≤20 pages) | `docs/a3-assignment-report.md` (+ appendix MD, not a separate study PDF) |
| 2 | **This guide** (comprehensive) | `docs/a3-predictive-intuition-guide.md` — includes Appendix Q |
| 3 | Optional practice quiz + key | `docs/a3-counterfactual-prediction-quiz.md` |

**Not PDFs:** prediction ledger (`docs/a3-prediction-ledger.md` + `.csv`), checklist (`internal/a3-requirements-checklist.md`), project context, experimental appendix tables, compact quant card `docs/a3-quantitative-prediction-reference.md` (MD only — mirrors Appendix Q MEASURED/HIT tables). Do not export overlapping study PDFs.

**Do not invent measured numbers.** Section **A** stays **PENDING MEASUREMENT** for any quantity not yet observed; when a stage lands, label those facts **MEASURED** and leave the rest **PENDING**. Sections **B–G** are mechanism hypotheses grounded in the handout (`internal/a3_optimization2.txt`). Pre-registered numerical intervals are labeled **PREDICTION** here and mirrored in the ledger (MD/CSV).

## A–G schema (every subproblem)

| Letter | Name | Fill rule |
|--------|------|-----------|
| **A** | Experimental result | **PENDING MEASUREMENT** — placeholders only; never invent values |
| **B** | Landscape / geometry | Minima, basins, barriers, spectrum shape; equations from handout |
| **C** | Dynamics / optimizer interaction | Schedules, momentum, Adam preconditioning, bouncing |
| **D** | Quantitative prediction template | Intervals, rankings, thresholds tagged **PREDICTION** |
| **E** | Counterfactual levers | Single knobs that should flip the outcome |
| **F** | Failure modes / confounds | Bad LR, short horizon, probe noise, seed |
| **G** | Cross-links & further tests | Ties to other problems; next cheap check |

Default setup unless overridden: d8 (~35M), ~614M tokens, AdamW \(\beta_1,\beta_2=(0.9,0.95)\), \(\epsilon=10^{-8}\), context 1024, batch 64, WD 0.1 (masked), grad clip 1.0; val loss ↓ better.

---

# Problem 1 — Mode Connectivity

Handout probe: for convex \(F\), Jensen gives \(F((x+y)/2)\le (F(x)+F(y))/2\). Deep nets are nonconvex; midpoint barrier \(B\) tests whether two low-loss checkpoints share a basin in **parameter** space.

## P1(a) — Forked-run sweep; endpoints + prediction ensemble

**A. Experimental result** — **MEASURED** (Batch 1 measure complete)  
- **MEASURED (prefix):** run `model-d8-lr0.003-tok614M-wsd0.0-stop1000` (W&B `rq1k0bz1`); end `train_loss=3.43004`, `val_loss=3.48832`. Modal `ap-L6RKrCcPCbhhgBvr3DLJfU`.  
- **MEASURED (branches):** 12/12 keep-ckpts; Modal `ap-57b3QORBZcKneFVkuuq7F3`.  
- **MEASURED (30-cell grid):** raw JSON `media/p1/a3-mode-connectivity.json` (repo copy under `experiments/a3_optimization/results/`, 20091 B @ 2026-10-10 17:21 PDT); Modal measure app `ap-ZU7fsZIl4Mq2Wi5veNzXCb`. Full table: `docs/a3-experimental-appendix.md` / `media/p1/barrier-table.csv`.  
- **MEASURED endpoints / ensemble (examples):** $(0,500)$: $L_1=3.7511$, $L_2=3.7483$, $L_{\mathrm{ens}}=3.6454$; $(1000,25)$: $L_1=3.4786$, $L_2=3.4796$, $L_{\mathrm{ens}}=3.4221$; $(250,100)$: $L_{\mathrm{ens}}=3.8869$. When $B>0.01$, ensemble is substantially below weight-avg (e.g. $(0,500)$: $L_{\mathrm{ens}}=3.6454$ vs $L(\theta_{\mathrm{avg}})=4.3734$).

**B. Landscape / geometry**  
For next-token probabilities,
\[
p_{\mathrm{ens}}(\cdot\mid c)=\tfrac12 p_{\theta_1}(\cdot\mid c)+\tfrac12 p_{\theta_2}(\cdot\mid c).
\]
Ensemble averaging is in **function space**. Two networks can disagree in \(\theta\) (high barrier later in P1b) yet agree on soft predictions → ensemble can stay low-loss while \(\theta_{\mathrm{avg}}\) sits on a ridge. Shared long prefix \(x\) keeps branches in a common basin; early forks (\(x=0\)) maximize opportunity for mode separation.

**C. Dynamics / optimizer interaction**  
Required schedule: `wsd0.0` (constant after warmup). Fork copies **model + optimizer state** at step \(x\); branches use different `batch_order_seed` but the same data distribution. Constant LR keeps late-training bounce amplitude high (river-valley; P2), so stochastic data-order differences can accumulate into distinct wall-orbits even when final losses match.  
**FITTED (Batch 1):** at early forks the ensemble stays low while \(B\) grows with post-fork horizon \(y\) (Appendix Q1.1 Fit A); shared-prefix length \(x\) collapses that barrier exponentially at fixed \(y=500\) (Fit B). Plots: `media/p1/plots/B_vs_y_curves.png`, `ensemble_vs_avg.png`.

**D. Quantitative prediction template** — calibrate against MEASURED where resolved  
- **PREDICTION (ranking; still open):** for fixed \(y\), as \(x\) increases, \(L(p_{\mathrm{ens}})\) approaches the better endpoint (shared-basin soft agreement).  
- **Ledger `P1-grid`:** PREDICTION was fraction \(B\le 0.01\) in \([0.2,0.5)\). **MEASURED:** \(26/30=0.867\in[0.8,1]\) — prediction under-estimated connectivity (still concentrated: failures only at small \(x\), large \(y\)).  
- **Ordering:** where modes separate, \(L(p_{\mathrm{ens}}) < L(\theta_{\mathrm{avg}})\). **MEASURED** on all four \(B>0.01\) cells (gaps below in P1b / ledger `P1-ens-gap`).

**E. Counterfactual levers**  
Change only one of: \(x\), \(y\), `batch_order_seed` pair, peak LR (keep schedule `wsd0.0` for the required grid). Optional later: same cells under weight averaging (P1b) or decay schedules (P1d).

**F. Failure modes / confounds**  
Untuned peak LR making both endpoints bad; comparing cells with unequal total steps \(x+y\); evaluating before branches finish \(y\) updates; mixing train vs val loss when reporting endpoints.

**G. Cross-links & further tests**  
Ensemble softness foreshadows P2 EMA (parameter-space averaging along a trajectory). Next cheap check: same cells under \(\theta_{\mathrm{avg}}\) and \(B\) (P1b). Cross-read: P4 \(\lambda_{\max}\) may differ across basins even when \(L\) matches.

---

## P1(b) — Weight averaging and midpoint barrier \(B\)

**A. Experimental result** — **MEASURED**  
- Definition used: $B=L(\theta_{\mathrm{avg}})-\tfrac12(L_1+L_2)$ at midpoint $\alpha=0.5$ from measure JSON.  
- **MEASURED:** $B_{\min}=-0.1161$ at $(250,100)$; $B_{\max}=0.6237$ at $(0,500)$; mean $-0.0334$.  
- **MEASURED:** **26 / 30** cells have $B\le 0.01$. Failures ($B>0.01$): $(0,250)$ $B=0.2173$, $(0,500)$ $B=0.6237$, $(50,500)$ $B=0.1299$, $(100,500)$ $B=0.0319$.  
- **MEASURED pattern:** large $B$ concentrated at small $x$ and large $y$; all $x\ge 250$ cells (and all $x=1000$) satisfy $B\le 0.01$. Table: `media/p1/barrier-table.csv`.

**B. Landscape / geometry**  
\[
B(\theta_1,\theta_2)=L\!\left(\tfrac{\theta_1+\theta_2}{2}\right)-\tfrac{L(\theta_1)+L(\theta_2)}{2},\qquad
\theta_{\mathrm{avg}}=\tfrac12\theta_1+\tfrac12\theta_2.
\]  
Handout criterion: \(B\le 0.01\) ⇔ “weight averaging works.” Large \(B\) ⇒ linear path crosses a ridge between modes. \(B\) is a **midpoint** probe only — full curves in P1(c).

**C. Dynamics / optimizer interaction**  
High constant LR encourages bouncing on valley walls (Wen et al. river-valley image in handout). Branches that bounce differently accumulate parameter disagreement \(\|\theta_1-\theta_2\|\) even if \(L(\theta_1)\approx L(\theta_2)\). Longer shared prefix synchronizes both weights and Adam moments before the fork.  
**FITTED rates (MEASURED grid only):**  
- At \(x=0\), growth regime \(y\ge 100\): \(B\approx -0.190+0.00163\,y\) (\(R^2\approx 1.00\), \(n=3\)) — barrier rises ~0.16 per +100 post-fork steps.  
- At \(y=500\): \(B\approx -0.066+0.687\,e^{-0.0232\,x}\) (\(R^2=0.995\)) — shared prefix halves the excess barrier on a ~30-step \(e\)-folding.  
- At \(x\ge 500\), slope of \(B\) vs \(y\) is consistent with zero (\(R^2\le 0.26\)) — late forks stay connected across the measured \(y\) range.  
Heatmap: `media/p1/plots/B_heatmap.png`. Full equations / residuals: Appendix Q1.1.

**D. Quantitative prediction template** — PREDICTION vs MEASURED vs FITTED  
- Threshold map intuition: large \(x\), small \(y\) → \(B\le 0.01\); small \(x\), large \(y\) → \(B>0.01\). **MEASURED:** matches direction (all failures have \(x\le 100\) and \(y\ge 250\)).  
- Ledger `P1-B-early-long`:  
  - **PREDICTION:** \(B(0,500)\in[0.05,0.2)\). **MEASURED:** \(B(0,500)=0.6237\in[0.2,\infty)\) — higher than predicted.  
  - **PREDICTION:** \(B(1000,25)\in(-\infty,0.01]\). **MEASURED:** \(B(1000,25)=-0.0597\) — match.  
- Ledger `P1-ens-gap` (cells with \(B>0.01\)): let \(G=L(\theta_{\mathrm{avg}})-L(p_{\mathrm{ens}})\).  
  - **PREDICTION:** \(G\ge 0.01\) and median \(G\in[0.02,0.08)\).  
  - **MEASURED gaps:** \((100,500)\,0.1311\); \((50,500)\,0.2309\); \((0,250)\,0.3273\); \((0,500)\,0.7281\); median \(G\approx 0.279\in[0.2,\infty)\). Direction (\(G\ge 0.01\)) matched; magnitude under-predicted.  
- **FITTED:** \(G\approx 0.090+1.033\,B\) on all 30 cells (\(R^2=0.989\)); on \(B>0.01\) cells \(G\approx 0.102+1.007\,B\) (\(R^2=0.9997\)).  
- **PREDICTION (extrapolated; ledger):** `P1-B-x0-y750`, `P1-B-x25-y500`, `P1-B-x2000-y500`, `P1-G-x0-y750` — intervals from Fits A–C beyond the measured \((x,y)\) grid (Appendix Q1.1).

**E. Counterfactual levers**  
Peak LR; batch size; schedule after fork (P1d); measuring only midpoints vs full \(\alpha\) curves (P1c).

**F. Failure modes / confounds**  
Barrier criterion is on **validation** loss — train-loss barriers can differ. Checkpoint timing must match \(x+y\). Cherry-picking only low-\(B\) cells for narrative. Treating \(B\le 0.01\) as continuous quality rather than a binary handout threshold.

**G. Cross-links & further tests**  
Where \(B\) is large, expect ensembles ≪ weight avg (P1a). Ties to P4: different basins may show different \(\lambda_{\max}\) under the same loss band. P2: decaying LR should shrink typical \(B\) at fixed \((x,y)\) (P1d).
---

## P1(c) — Full interpolation curves \(\theta(\alpha)\)

**A. Experimental result** — **MEASURED** (5-point $\alpha$ curves in JSON)  
- Every cell has val losses at $\alpha\in\{0,0.25,0.5,0.75,1\}$ in `media/p1/a3-mode-connectivity.json`.  
- **MEASURED examples:** $(0,500)$ interpolation `[3.7511, 3.9156, 4.3734, 3.8961, 3.7483]` with $L_{\mathrm{ens}}=3.6454$ (large midpoint barrier). $(1000,500)$: `[3.3812, 3.317, 3.2986, 3.3172, 3.381]` with $L_{\mathrm{ens}}=3.3017$, $B=-0.0825$.  
- Full per-cell curves: raw JSON; midpoint summary: appendix / `barrier-table.csv`.

**B. Landscape / geometry**  
\[
\theta(\alpha)=(1-\alpha)\theta_1+\alpha\theta_2,\quad \alpha\in\{0,0.25,0.5,0.75,1\}.
\]  
Unimodal barrier peaking near \(\alpha=0.5\) ⇒ distinct basins along the linear path. Near-flat or concave path ⇒ mode connectivity / shared basin. Asymmetric peak (\(\alpha\neq 0.5\)) means midpoint \(B\) understates the true path max.

**C. Dynamics / optimizer interaction**  
Endpoints from the same constant-LR fork often have similar \(\|\theta\|\) scale, so linear interpolation is a meaningful geometric probe. Midpoint alone can miss asymmetric barriers; denser \(\alpha\) later if asymmetry appears.  
**FITTED link:** when midpoint \(B\) is large, Fit C gives \(G\approx B+0.10\), so the \(\alpha=0.5\) ridge height almost fully appears as ensemble–weight-avg gap (see `media/p1/plots/ensemble_vs_avg.png`).

**D. Quantitative prediction template** — PREDICTION vs MEASURED (ledger `P1-interp-peak`)  
For a representative cell with midpoint \(B\in[0.05,0.2)\) — **MEASURED** cell \((50,500)\), \(B=0.1299\):  
- **PREDICTION (ranking):** \(L(\theta(0.5))\) strictly largest among five \(\alpha\). **MEASURED:** \([3.7009, 3.7345, 3.8253, 3.7265, 3.6898]\) — \(\alpha=0.5\) worst; match.  
- **PREDICTION (height):** barrier in \([0.05,0.2)\). **MEASURED:** \(0.1299\) — match.  
- **PREDICTION:** \(L(p_{\mathrm{ens}})\) below barrier peak. **MEASURED:** \(L_{\mathrm{ens}}=3.5943 < 3.8253=L(\theta(0.5))\).  
High-\(B\) extreme \((0,500)\): peak still at \(\alpha=0.5\), but height \(0.6237\in[0.2,\infty)\) (outside the preregistered mid band).

**E. Counterfactual levers**  
Which \((x,y)\) cells are “representative”; optional denser \(\alpha\) grid; compare a low-\(B\) cell (flat path) vs high-\(B\) cell (barrier).

**F. Failure modes / confounds**  
Cherry-picking only low-\(B\) cells; reporting train loss on the path; numerical issues if endpoints have wildly different scales (unlikely same prefix).

**G. Cross-links & further tests**  
Use curves to choose P1(d) ablations. Connect to P2: EMA averages along a **trajectory**, not across independently forked modes — geometry differs from \(\theta(\alpha)\).

---

## P1(d) — What controls same-basin membership

**A. Experimental result** — PENDING MEASUREMENT  
- TODO: LR / batch / schedule ablations at fixed fork geometry; patterns; proposed follow-ups.

**B. Landscape / geometry**  
Basin identity is not final loss alone — two low-loss solutions can sit across a ridge (\(B>0.01\) with matched endpoints). Knobs that keep trajectories closer in parameter space raise the chance of \(B\le 0.01\).

**C. Dynamics / optimizer interaction**  
- Lower peak LR or decay schedules → less wall-bounce → branches stay closer.  
- Larger batch → less gradient noise → slower stochastic mode separation.  
- `wsd0.0` after fork maximizes continued bounce; cosine / `wsd0.2` settle toward the river late (P2).  
A2 lesson: retune peak LR after batch changes or conclusions are LR artifacts.  
**FITTED baseline (constant `wsd0.0` only):** Fit B says excess barrier at \(y=500\) decays as \(\sim e^{-0.023\,x}\); schedule ablations should be scored as \(\Delta B\) vs this constant-LR map (Appendix Q1.1), not vs a blank slate.

**D. Quantitative prediction template** — **PREDICTION** (ledger `P1-d-dB-cos`)  
Fix \((x,y)=(250,250)\), same seeds, same peak LR; arm W = `wsd0.0` after fork; arm C = cosine after fork; \(\Delta B=B_W-B_C\).  
- **PREDICTION:** \(\Delta B\in[0.02,0.1)\) (cosine lowers barrier).  
- **PREDICTION:** \(B_C\le 0.01\) only if \(B_W\) was modestly above 0.01 — else cosine helps but may not cross the threshold (forced choice: “depends on \(B_W\) band”).  
Open pre-reg (ledger notes): peak-LR \(\times 2\) and batch \(\times 2\) \(\Delta B\) intervals before launch.

**E. Counterfactual levers**  
Peak LR \(\times 2\); batch \(\times 2\); `wsd0.2` vs `wsd0.0`; seed diversity; warmup length (secondary).

**F. Failure modes / confounds**  
Changing multiple knobs at once; not retuning LR after batch change; claiming “schedule causes basins” without matched token budgets / evaluation protocol.

**G. Cross-links & further tests**  
P2 regret/EMA: same bounce mechanism. P3: edge-of-stability regime (\(\eta\lambda\) large) may correlate with high \(B\) under constant LR. Cheap next: one cosine vs `wsd0.0` cell before a full grid.

---

# Problem 2 — River-Valley Dynamics

Handout picture (Wen et al.): steep hillsides + flat river. Large step sizes bounce between walls while drifting along the river; decay settles iterates toward the river.

**Batch 2 label snapshot (finished 9/9 required + P3 sharpness — do not invent the rest):**

| Status | Items |
|--------|-------|
| **MEASURED** | Four schedule finals; dip **final** \(R\); invsqrt \(\Delta L\); **SGDR** \(\Delta L\) + vs-invsqrt; P3 SGD \(\lambda_{\max}\); AdamW \(\eta\lambda^{\mathrm{pre}}\); **P4(a)** stage-a **13/13** scored; **P5** 6/7 (`P5-L-cool-dir` **INSUFFICIENT**) |
| **PENDING** (not MEASURED) | **mid-dip** \(R_{\mathrm{mid}}\); EMA gap; P3 η-change arms; **P4(b)** after-arm / `P4-invar-dL`; **P4(c)** (P4(b) soft baseline MEASURED — see §P4(b)) |

## P2(a) — Schedule comparison (constant / linear / cosine / plateau-then-decay)

**A. Experimental result** — **MEASURED** (four schedule arms complete; Batch 2 overall **9/9** + P3 sharpness)  
- **MEASURED finals** (`media/p2/batch2_partial_finals.csv`):  

| Rank | Arm | W&B | Final val | \(R\) vs best=`linear` |
|------|-----|-----|-----------|------------------------|
| 1 | `linear` | `r0294p93` | 2.9282028675079346 | 0 |
| 2 | `cos` | `sktw67q8` | 2.934119462966919 | 0.005916595458984375 |
| 3 | `wsd0.2` | `soosjcxr` | 2.93774151802063 | 0.009538650512695312 |
| 4 | `wsd0.0` | `widt420l` | 3.0975050926208496 | 0.16930222511291504 |

- **After measurement (ledger `B2-P2-*-R` / `P2-sched-rank`; scorecard):**  
  - **Ranking HIT** (`P2-sched-rank`): \(\min(L_{\mathrm{cos}},L_{\mathrm{wsd0.2}},L_{\mathrm{linear}}) < L_{\mathrm{wsd0.0}}\) — direction of “decayed beats constant” matched.  
  - **`wsd0.0` \(R\) band MISS:** MEASURED \(R=0.16930222511291504\approx 0.17\) vs PREDICTION \([0.02,0.08)\); still worst of four (worst **HIT**). Constant LR regret was under-estimated.  
  - **`wsd0.2` top-2 MISS:** band \(R\in[0,0.02)\) **HIT**, but rank #3 (top-2 with `cos` **MISS**); `linear`/`cos` took the top-2.  
  - **`linear` / `cos`:** both band **HIT**; `linear` best (\(R=0\)).  
- **Oct 20 quiz implication:** when scoring constant-LR vs decayed regret, prefer **higher** \(R\) bands than the locked `[0.02,0.08)` — the miss signal is that constant LR regret was under-estimated (here \(\approx 0.17\)). Keep ranking direction (“decayed \(<\) constant”) unless a new measure flips it.  
- Batch 2 **9/9**: dip final \(R\), invsqrt \(\Delta L\), SGDR, and both P3 sharpness arms also MEASURED (see P2(c)/(d) + P3 §A). Still **PENDING**: EMA / mid-dip.  
- TODO still: LR/train/val curves; peak-LR robustness; EMA gap for P2(b).

**B. Landscape / geometry**  
Constant late LR keeps large bounce amplitude → iterates spend time up the hillsides → higher final val possible even if mid-training loss looks competitive. Decay schedules shrink hillside steps → settle nearer the river floor.

**C. Dynamics / optimizer interaction**  
Starter schedules: `wsd0.0` (constant after warmup), `linear`, `cos`, `wsd0.2` (decay last 20%). Compare at equal token counts. AdamW unchanged. Peak LR must be tuned enough that the ranking is not an under-tuned artifact (handout).

**D. Quantitative prediction template** — locked PREDICTION vs **MEASURED** (Batch 2 lock `2026-10-11T00:32:07Z`)

| Ledger ID | Batch 2 job | Quantity | PREDICTION band | Status |
|-----------|-------------|----------|-----------------|--------|
| `B2-P2-wsd0.0-R` | `a3-b2-p2-wsd0.0` | Final \(R\) vs best of four | \(R\in[0.02,0.08)\); worst | **MEASURED** \(R=0.1693\) band **MISS**; worst **HIT** |
| `B2-P2-linear-R` | `a3-b2-p2-linear` | Final \(R\) | \(R\in[0,0.02)\); \(<L_{\mathrm{wsd0.0}}\) | **MEASURED** **HIT** (\(R=0\)) |
| `B2-P2-cos-R` | `a3-b2-p2-cos` | Final \(R\) | \(R\in[0,0.02)\); top-2 | **MEASURED** **HIT** |
| `B2-P2-wsd0.2-R` | `a3-b2-p2-wsd0.2` | Final \(R\) | \(R\in[0,0.02)\); top-2 | **MEASURED** band **HIT**; top-2 **MISS** |
| `P2-sched-rank` | jobs 0–3 | Ranking | \(\min(L_{\mathrm{cos}},L_{\mathrm{wsd0.2}},L_{\mathrm{linear}}) < L_{\mathrm{wsd0.0}}\) | **MEASURED** **HIT** |

Best raw final → \(\theta^{\mathrm{ref}}\) for \(R_M(t)\). Cross-refs: dip / HF in P2(c)/(d); P3 sharpness in P3 §A. Still **PENDING**: mid-dip \(R_{\mathrm{mid}}\), EMA.

**E. Counterfactual levers**  
Peak LR; decay fraction (`wsd0.2` vs other); token budget; warmup fraction (secondary).

**F. Failure modes / confounds**  
Declaring a schedule winner at a bad peak LR; unequal evaluation protocols; reading train loss as if it were val; stopping mid-cooldown inconsistently; promoting mid-run W&B as finals.

**G. Cross-links & further tests**  
Reference run feeds P2(b) EMA/regret. Horizon-free alternatives in P2(d). Same bounce story explains P1(d) schedule effect on \(B\).

---

## P2(b) — Eval-only EMA and regret

**A. Experimental result** — **PENDING MEASUREMENT** (still blank after Batch 2 **9/9**)  
- TODO: raw vs EMA val for \(\beta\in\{0.99,0.999\}\); regret curves; residual schedule gaps after EMA.  
- **PREDICTION** locked before Batch 2 completion (jobs log eval-only EMA). Do **not** treat schedule **MEASURED** finals as EMA results.

**B. Landscape / geometry**  
\[
\bar\theta_0=\theta_0,\qquad
\bar\theta_t=\beta_{\mathrm{EMA}}\bar\theta_{t-1}+(1-\beta_{\mathrm{EMA}})\theta_t.
\]  
EMA averages along the bounced trajectory → closer to the river centerline in parameter space (distinct from P1’s cross-mode \(\theta_{\mathrm{avg}}\)).

**C. Dynamics / optimizer interaction**  
EMA is **evaluation-only** (no training-state change); starter logs `val_ema{decay}_loss`. Longer timescale \(\beta=0.999\) is smoother but laggier than \(0.99\). Regret uses the P2(a) best raw final as reference unless stated otherwise.

**D. Quantitative prediction template** — **PREDICTION** (ledger `P2-ema-gap-frac`; B2-locked)

| Quantity | PREDICTION band / ranking |
|----------|---------------------------|
| Gap-closed fraction \(f=1-G_{\mathrm{EMA}}/G_{\mathrm{raw}}\) | \([0.5,0.75)\) |
| After best EMA, lowest schedules | `cos` or `wsd0.2` still favored |
| Absolute final val | not invented — relative gaps only |

**E. Counterfactual levers**  
EMA decays; which schedule is reference; when EMA is logged; peak LR retune before claiming gap closure.

**F. Failure modes / confounds**  
Accidentally training with EMA weights; comparing EMA of one run to raw of another; including warmup-dominated early EMA in “final gap.”

**G. Cross-links & further tests**  
Function-space ensemble (P1) vs parameter EMA (P2). Dip/rewarm (P2c) tests whether “settling” is reversible. P3: lowering effective \(\eta\) during bounce changes \(\eta\lambda\).

---

## P2(c) — Temporary LR dip then rewarm

**A. Experimental result** — **MEASURED** final \(R\) · mid-dip **PENDING** (Batch 2 job 4 finished)  
- **MEASURED** (`media/p2/batch2_partial_finals.csv`; ledger `B2-P2-dip-R` / `P2-dip-final-R`):  
  - Job 4 `a3-b2-p2-wsd0.2-dip` W&B `rms7iyyt`: final val \(2.938987970352173\).  
  - Control job 3 `wsd0.2` `soosjcxr`: final val \(2.93774151802063\).  
  - Final \(R=L_{\mathrm{dip}}-L_{\mathrm{wsd0.2}}=0.0012464523315429688\in[-0.02,0.02)\) — band **HIT**.  
- Mid-dip \(R_{\mathrm{mid}}\) — **PENDING** / **not MEASURED** (no mid-run snapshot in partial CSV).  
- **Oct 20 quiz implication:** final dip regret near zero matches the locked band — temporary dip does not hurt (or help) the final vs matched `wsd0.2` cooldown; do not invent a mid-dip score until a snapshot exists.

**B. Landscape / geometry**  
Dip → less bounce → drop toward river; rewarm → bounce resumes. Final position may return near the control’s river-wall orbit — transient gains need not survive.

**C. Dynamics / optimizer interaction**  
`lr_dip=(start,bottom,end,scale)` multiplies any schedule; times as fractions of training; model/optimizer state **not** reset. Handout requires the same final cooldown on control and dipped run for fair finals.

**D. Quantitative prediction template** — PREDICTION vs MEASURED (final only)

| Quantity | Tag | Band / value |
|----------|-----|--------------|
| Final \(R=L_{\mathrm{dip}}-L_{\mathrm{ctrl}}\) | **MEASURED** **HIT** | \(0.0012464523315429688\in[-0.02,0.02)\) (PREDICTION band matched) |
| Mid-dip \(R_{\mathrm{mid}}\) | **PENDING** / **PREDICTION** only | Band \((-\infty,-0.02)\); Observed blank — not in partial CSV; **not MEASURED** |
| Larger/longer dips | **PREDICTION** (qualitative) | larger transient \(|R|\) mid-run; finals still near 0 if rewarm+cooldown matched |

**E. Counterfactual levers**  
Dip scale, duration, start time; peak LR of the parent schedule; cooldown length.

**F. Failure modes / confounds**  
Missing final cooldown on control; resetting optimizer; measuring only train loss; starting dip before warmup finishes.

**G. Cross-links & further tests**  
EoS (P3): dip changes \(\eta\) → expected \(\lambda\) retarget (cousin of P3(a)ii). Horizon-free schedules (P2d) avoid budget-specific cooldowns.

---

## P2(d) — Horizon-free schedules (invsqrt, SGDR)

**A. Experimental result** — **MEASURED** (Batch 2 @ 9/9; invsqrt + SGDR finished; P3 sharpness also **MEASURED**)  
- **MEASURED invsqrt** (`media/p2/batch2_partial_finals.csv`; ledger `B2-P2-invsqrt-dL`):  
  - Job 5 `a3-b2-p2-invsqrt` W&B `1jri34em`: final val \(3.020456075668335\).  
  - Tuned ref = best of `{linear, cos, wsd0.2}` = `linear` (\(L^\star=2.9282028675079346\)).  
  - \(\Delta L=L_{\mathrm{invsqrt}}-L_{\mathrm{linear}}=0.09225320816040039\) vs PREDICTION \([0.01,0.08)\) — band **MISS** (too high; in \([0.08,\infty)\)).  
- **MEASURED SGDR** (`B2-P2-sgdr-dL` / job 6 W&B `51b3qq4n`): final val \(3.0348048210144043\).  
  - \(\Delta L=L_{\mathrm{sgdr}}-L_{\mathrm{linear}}=0.10660195350646973\) — band **MISS** (in \([0.08,\infty)\)).  
  - \(\lvert L_{\mathrm{sgdr}}-L_{\mathrm{invsqrt}}\rvert=0.014348745346069336\in[0,0.03)\) — closeness **HIT**.  
- **Oct 20 quiz implication:** both HF \(\Delta L\) gaps were **larger** than `[0.01,0.08)`; SGDR stayed close to invsqrt. Keep direction “HF worse than budget-tuned.”  
- TODO still: multi-budget transfer without cooldown.

**B. Landscape / geometry**  
Unknown horizon ⇒ cannot rely on a final “settle to river” phase sized to the budget; iterates may remain partially bouncing at stop → higher val than a budget-tuned cosine/`wsd0.2` with a long cooldown.

**C. Dynamics / optimizer interaction**  
Invsqrt (Vaswani et al.): for \(t\ge 1\),
\[
\alpha_t=\alpha_{\max}\min\bigl\{t/w,\sqrt{w/t}\bigr\}.
\]  
SGDR: after warmup, cycle
\[
\alpha_t=\alpha_{\min}+\tfrac12(\alpha_{\max}-\alpha_{\min})\bigl[1+\cos(\pi s/T)\bigr];
\]  
restart **LR only**, not model/optimizer. AdamW unchanged. Choose \(w,T,\alpha_{\max},\alpha_{\min}\) independently of final budget. Starter: `warmup_steps=94`, `sgdr2000`, `min_lr_ratio=0.1`.

**D. Quantitative prediction template** — invsqrt + SGDR **MEASURED** (B2-locked `2026-10-11T00:32:07Z`)

| Ledger ID | Batch 2 job | Quantity | PREDICTION band | Status |
|-----------|-------------|----------|-----------------|--------|
| `B2-P2-invsqrt-dL` | `a3-b2-p2-invsqrt` | \(\Delta L=L_{\mathrm{hf}}-L_{\mathrm{tuned}}\) at \(T_{\mathrm{long}}\) | \([0.01,0.08)\) | **MEASURED** \(0.09225320816040039\) **MISS** |
| `B2-P2-sgdr-dL` | `a3-b2-p2-sgdr2000` | same \(\Delta L\) | \([0.01,0.08)\) | **MEASURED** \(0.10660195350646973\) **MISS** |
| `B2-P2-sgdr-dL` | vs invsqrt | \(\lvert L_{\mathrm{sgdr}}-L_{\mathrm{invsqrt}}\rvert\) | \([0,0.03)\) | **MEASURED** \(0.014348745346069336\) **HIT** |
| `P2-horizon-dL` | mid budget (not B2 stop) | usefulness | \(\lvert\Delta L\rvert<0.02\) competitive | **PREDICTION** only (full-stop HF both MISS) |

Transfer test: freeze \((w,T)\) across budgets — usefulness = unknown-horizon setting, not cherry-picked \(T\).

**E. Counterfactual levers**  
\(w\), \(T\), \(\alpha_{\max},\alpha_{\min}\); stopping budgets; accidental cooldown (must avoid).

**F. Failure modes / confounds**  
Accidentally adding cooldown; restarting optimizer state on SGDR; retuning \(w,T\) per budget (defeats the point); unfair peak-LR mismatch vs tuned baselines.

**G. Cross-links & further tests**  
Practice-quiz style: transfer a \((w,T)\) pair across budgets. Tie to P3: SGDR restarts briefly change \(\eta\lambda^{\mathrm{pre}}\). Tie to P2(c): both probe reversible vs permanent settling.

---

# Problem 3 — The Edge of Stability

Quadratic GD: stable for \(\eta\lambda_{\max}<2\). Cohen et al.: full-batch GD progressively sharpens until \(\eta\lambda_{\max}\approx 2\) and rides the edge. Adam \(\beta_1=0.9\) frozen-preconditioner story: edge at \(\eta\lambda^{\mathrm{pre}}_{\max}\approx 38\).

**Label rule:** P3 SGD clean-edge \(\lambda\) and AdamW post-warmup \(\eta\lambda^{\mathrm{pre}}\) are **MEASURED HIT**; P3(a)ii η-change arms stay **PREDICTION** / **PENDING**.

## P3(a)i — Full-batch clean case (\(N=32\), \(\eta=0.01\))

**A. Experimental result** — **MEASURED** (Batch 2 job `a3-b2-p3-0-sgd` / W&B `0dewkykj`; ledger `B2-P3-sgd-lam` / `P3-clean-edge` **HIT**)  
- Median \(\lambda_{\max}\) over updates 1200–1500: **245.16314737956807** (n=12; window \(\eta\lambda\) median \(2.451631473795681\); `media/p2/p3_sgd_sharpness_summary.json`).  
- PREDICTION band \([150,250)\) → **HIT**; near \(2/\eta=200\); late train loss ↓ (final train \(1.268615484237671\) @ step 1499; `internal/batch2-p3-sgd-lam-score-status.md`).  
- TODO still: publish \(\lambda_{\max}\) vs update / vs train-loss plots with dashed threshold at 200.

**B. Landscape / geometry**  
Sharpness \(\lambda_{\max}=\lambda_{\max}(H(\theta))\) on a fixed probe. Progressive sharpening: curvature rises until the optimizer’s stability wall. Riding the edge: loss can still decrease while \(\lambda_{\max}\) hovers near \(2/\eta\).

**C. Dynamics / optimizer interaction**  
`full_batch(32, 1500, learning_rate=0.01)`: GD on first \(N\) sequences, 16× embedding init, 1500 updates. Not the default AdamW minibatch recipe — this is the Cohen-style clean case.

**D. Quantitative prediction template** — **MEASURED** (locked PREDICT `2026-10-11T00:32:07Z`; ledger `B2-P3-sgd-lam` / `P3-clean-edge` **HIT**)

| Quantity | Target / mechanism | PREDICTION band | Observed |
|----------|--------------------|-----------------|----------|
| Median \(\lambda_{\max}\) updates 1200–1500 | edge \(2/\eta=200\) at \(\eta=0.01\) | \([150,250)\) | **245.16314737956807** (**HIT**) |
| Progressive sharpening then hover | Cohen EoS | direction: rise → ride near 200 | window \(\eta\lambda\) median \(2.451631473795681\); summarize plateau \(2.547386438768327\) |
| Late train loss while at edge | Cohen-style | continues to decrease overall | final train \(1.268615484237671\) @ step 1499 (↓) |
| Batch 2 job | `a3-b2-p3-0-sgd` | — | `0dewkykj` finished |

Source: `media/p2/p3_sgd_sharpness_summary.json` / `internal/batch2-p3-sgd-lam-score-status.md` / scorecard `B2-P3-sgd-lam`.

**E. Counterfactual levers**  
\(N\), \(\eta\), init scale (16× embedding init per handout), update count.

**F. Failure modes / confounds**  
Probe ≠ train sequences; too few Lanczos iterations; mistaking transient spikes for the edge; plotting during warmup as if at edge; promoting mid-run sharpness as final median.

**G. Cross-links & further tests**  
Mid-run \(\eta\) changes (a-ii); minibatch Adam (b). River-valley bounce is the macroscopic cousin of edge-riding oscillations.

---

## P3(a)ii — Change \(\eta\) mid-run (×0.5 / ×2 at 750)

**A. Experimental result** — **PENDING MEASUREMENT** / not started  
- TODO: \(\lambda_{\max}\) trajectories after half / double; symmetry check (time-to-new-edge, overshoot).  
- **Not in Batch 2 required 9-job manifest** (expand starter / later batch). Intervals below are **PREDICTION** only — **not MEASURED**.

**B. Landscape / geometry**  
Edge location is \(2/\eta\). Halving \(\eta\) raises the allowed sharpness ceiling; doubling lowers it. Response need **not** be symmetric in time or overshoot (handout asks).

**C. Dynamics / optimizer interaction**  
`lr_schedule="step750x0.5"` vs double at update 750. After \(\eta\leftarrow\eta/2\): expect further sharpening toward new edge. After \(\eta\leftarrow 2\eta\): expect \(\lambda_{\max}\) to drop or become unstable until below new edge.

**D. Quantitative prediction template** — **PREDICTION** (ledgers `P3-eta-half`, `P3-eta-double`; not B2 required)

| Intervention | Target \(2/\eta\) | PREDICTION median \(\lambda_{\max}\) (1200–1500) |
|--------------|-------------------|--------------------------------------------------|
| \(\eta\leftarrow 0.005\) at 750 | 400 | \([350,450)\) |
| \(\eta\leftarrow 0.02\) at 750 | 100 | \([80,120)\) |
| Time-to-retarget symmetry | — | **not** assumed equal |

**E. Counterfactual levers**  
Step location (750); factor (0.5 vs 2); \(N\); whether first edge was reached before the step.

**F. Failure modes / confounds**  
Changing \(\eta\) before the first edge is reached; comparing absolute \(\lambda\) without \(\eta\lambda\); insufficient steps after the change to retarget.

**G. Cross-links & further tests**  
P2 LR dip is the noisy minibatch cousin of this intervention. P4: \(\lambda_{\max}\) moves can also come from reparameterization without dynamics (invariance).

---

## P3(b) — Minibatch AdamW preconditioned sharpness

**A. Experimental result** — **MEASURED** (Batch 2 job `a3-b2-p3-1-adamw` / W&B `foew12y0`; ledger `B2-P3-adamw-pre` / `P3-mb-edge` **HIT**)  
- Post-warmup median \(\eta\lambda^{\mathrm{pre}}\): **30.987715914514** (n=25 post-warmup; primary scoring metric; `media/p2/p3_adamw_sharpness_summary.json`).  
- PREDICTION band \([30,50)\) → **HIT**; near Adam threshold 38. Secondary (not scoring): all-measured median \(31.813196147279584\); 100–180M-tok window median \(22.496516434600995\).  
- TODO still: publish \(\eta\lambda^{\mathrm{pre}}\) vs tokens plot with dashed threshold at 38.

**B. Landscape / geometry**  
\[
\lambda^{\mathrm{pre}}_{\max}=\lambda_{\max}(D^{-1/2}HD^{-1/2}),\quad D=\sqrt{\hat v}+\epsilon.
\]  
Preconditioned sharpness is the relevant stability quantity for Adam-style steps.

**C. Dynamics / optimizer interaction**  
`minibatch()`: constant LR, shorter horizon (compute). For Adam with \(\beta_1=0.9\) and frozen-preconditioner story:
\[
\eta\lambda^{\mathrm{pre}}_{\max}\approx 2(1+\beta_1)/(1-\beta_1)=38.
\]  
Minibatch noise may blur the edge vs full-batch GD’s tight ride at 2.

**D. Quantitative prediction template** — **MEASURED** (ledger `B2-P3-adamw-pre` / `P3-mb-edge` **HIT**)

| Quantity | PREDICTION band / ranking | Observed |
|----------|---------------------------|----------|
| Post-warmup median \(\eta_t\lambda^{\mathrm{pre}}_{\max}\) | \([30,50)\) (near 38) | **30.987715914514** (**HIT**) |
| Secondary medians (not primary score) | all-measured / 100–180M tok | \(31.813196147279584\) / \(22.496516434600995\) |
| Vs full-batch GD \(\eta\lambda\approx 2\) ride | noisier / wider band — not stuck near 2, not \(\to\infty\) | post-warmup \(\eta\lambda^{\mathrm{pre}}\approx 31\) near 38; not at 2 |
| Batch 2 job | `a3-b2-p3-1-adamw` | `foew12y0` finished |

Source: `media/p2/p3_adamw_sharpness_summary.json` / `internal/batch2-p3-adamw-score-status.md` / scorecard `B2-P3-adamw-pre`.

**E. Counterfactual levers**  
Batch size, \(\beta_1\), LR schedule (constant in starter recipe), horizon length.

**F. Failure modes / confounds**  
Including warmup in edge plots; reading raw \(\lambda_{\max}\) instead of preconditioned; short-horizon artifacts; wrong \(\beta_1\) in the threshold formula.

**G. Cross-links & further tests**  
P3(c) hparam exploration; P4 spectrum explains *which* directions dominate \(\lambda_{\max}\). P2 schedules modulate \(\eta_t\) hence the instantaneous edge.

---

## P3(c) — Open-ended hyperparameter exploration (~18 runs)

**A. Experimental result** — **PENDING MEASUREMENT** / not started  
- TODO: factors that help enter/stay at the edge; plots with threshold lines; ≤18 runs budget honesty.  
- Example axis templates in §D are **PREDICTION** sketches only — zero MEASURED P3(c) outcomes.

**B. Landscape / geometry**  
Anything that changes effective curvature or step size relative to curvature moves time-to-edge and stability. Goal: factors affecting tendency to **enter and stay** at the edge (handout).

**C. Dynamics / optimizer interaction**  
Candidate axes (one knob per cluster): batch size, peak LR, \(\beta_1/\beta_2\), weight decay, schedule, init scale, probe size. Prefer axes that change \(\eta\lambda^{\mathrm{pre}}\) interpretable against threshold 38.

**D. Quantitative prediction template** — **PREDICTION** (pre-register ≤3 axes in ledger before launch)  
Example templates (do not treat as measured):  
- Batch↑ at fixed peak LR: **PREDICTION** — harder to stay tightly at edge (noisier \(\eta\lambda^{\mathrm{pre}}\); median band widens).  
- Peak LR↑: **PREDICTION** — faster time-to-edge; risk of instability overshoot.  
- \(\beta_1\)↑: threshold \(2(1+\beta_1)/(1-\beta_1)\) moves — compare medians to the **new** threshold, not blindly to 38.

**E. Counterfactual levers**  
One knob per run cluster; keep compute budget honest (~18 runs).

**F. Failure modes / confounds**  
Multiple simultaneous changes; under-tuning LR so “no edge” is just “too small \(\eta\)”; claiming edge absence without plotting the correct threshold.

**G. Cross-links & further tests**  
Feed P4(b) interventions that move \(\lambda_{\max}\) without hurting loss. Report should separate “dynamics edge” from “parameterization sharpness” (P4b).

---

# Problem 4 — Inside the Hessian

Objects: sharpness, normalized sharpness \(\mathrm{tr}(H^2)/d\), negative eigenvalues, outliers, block structure (family & layer). Tools: Lanczos HVPs, Hutchinson; probe = 64 sequences; FP32, TF32 off.

**Label rule:** AdamW Hessian steps **0 / 3 / 15 / 61 / 153 / 305 / 763 / 1526 / 3052 / 4578 / 6104 / 7630 / 9375** are **MEASURED** from `media/p4/stage-a/` JSON (+ summaries). Stage a **13/13 DONE**. Handout final bands \(\lambda_{\max}\in[4.5,6.0]\), val \(\in[2.925,2.931]\) are **assignment-given**. Rebound note: `internal/p4-lambda-rebound-13of13.md`. Ledger HIT/MISS copied from `internal/p4-score-dryrun.md` (operator; `--apply` does not mutate docs) → `internal/p4-9375-hit-miss-summary.md`.

**Trajectory (**MEASURED** 13/13).** Collapse by 15 → dip \(\sim 2.195\) @1526 (earlyfit **HIT**) → rebound to \(\lambda@9375=5.967310256347902\) (**inside** handout `[4.5,6.0]`). Norm **HIT**; block-energy **HIT** (excess_final=0.08727119221832025). Rebound: `internal/p4-lambda-rebound-13of13.md`.

**MISS mechanisms (predictive — PREDICT bands locked; scored vs MEASURED).** From preregister (`docs/a3-p4-preregister-draft.md`) + file-exact `internal/p4-9375-hit-miss-summary.md`:
- **`P4-a-family-sharpest`:** PREDICT late ∈ {`attention`,`mlp`} (progressive sharpening in large residual blocks; embed/head early). MEASURED late=`head` (\(5.641603135007765\)); early_top2=`embedding`,`attention`; ratio \(7.38436579244244\notin[1.5,5)\). Residual-block late locus **failed**.
- **`P4-a-density-neg`:** PREDICT neg mass shrinks 0→9375; final ∈`[0,0.15)`. MEASURED \(0.004510114309467541\to0.01491374251600709\to0.9344596369432798\to0.06965100420936976\); final band OK, shrink_0→9375=**false**. Saddle-cleanup direction **failed**.
- **`P4-lam-9375-earlyfit`:** PREDICT `[0.3,4.0)` (early decay / preferred asymptote \(\sim 3.51\)). MEASURED \(5.967310256347902\) — **MISS** the earlyfit envelope, **HIT** the handout edge band `[4.5,6.0]` by rebound.

## P4(a) — What the Hessian looks like over training (0 train runs; measure ckpts)

**A. Experimental result** — **MEASURED** (steps 0, 3, 15, 61, 153, 305, 763, 1526, 3052, 4578, 6104, 7630, 9375; stage a **13/13 DONE**)  
- **MEASURED \(\lambda_{\max}\) trajectory** (summaries / stage-a JSON):

| Step | Source | \(\lambda_{\max}\) (exact) |
|------|--------|----------------------------|
| 0 | `media/p4/a-adamw-0.summary.json` | \(1431.5030691763686\) |
| 3 | `media/p4/a-adamw-3.summary.json` | \(1571.7420262270407\) |
| 15 | `media/p4/a-adamw-15.summary.json` | \(53.36426571742344\) |
| 61 | `media/p4/a-adamw-61.summary.json` | \(18.054717868295125\) |
| 153 | `media/p4/stage-a/a-adamw-153.json` | \(6.502395300121604\) |
| 305 | `media/p4/stage-a/a-adamw-305.json` | \(4.32926630998023\) |
| 763 | `media/p4/a-adamw-763.summary.json` | \(3.02398152405625\) |
| 1526 | `media/p4/a-adamw-1526.summary.json` | \(2.1954286466134665\) |
| 3052 | `media/p4/a-adamw-3052.summary.json` | \(3.1637529312879873\) |
| 4578 | `media/p4/a-adamw-4578.summary.json` | \(3.7423705126158477\) |
| 6104 | `media/p4/a-adamw-6104.summary.json` | \(4.155164815834522\) |
| 7630 | `media/p4/a-adamw-7630.summary.json` | \(5.281516516770698\) |
| 9375 | `media/p4/a-adamw-9375.summary.json` | \(5.967310256347902\) |

![MEASURED early P4 \(\lambda_{\max}\) (steps 0 / 3 / 15 / 61 / 153 / 305 / 763 / 1526 / 3052 / 4578 / 6104 / 7630 / 9375; Batch3 stage a 13/13)](../media/p4/p4_lambda_max_early.png)

![MEASURED early P4 family \(\lambda_{\max}\) (embedding / attention / mlp / head; steps 0 / 3 / 15 / 61 / 153 / 305 / 763 / 1526 / 3052 / 4578 / 6104 / 7630 / 9375; Batch3 stage a 13/13)](../media/p4/p4_family_lambda_early.png)

- **Pattern (MEASURED 13/13):** collapse by 15 → minimum @1526 (\(2.195\)) → rebound to \(\lambda@9375=5.967310256347902\) (**inside** handout). Late family top = **head** (\(5.642\)) then attention (\(1.208\)). `density_runs` at **0, 153, 1526, 9375**.
- **FIT context (not MEASURED):** preferred early-fit asymptote \(\sim 3.51\) from `internal/p4-lambda-early-fit.md` — **fit / PREDICTION only**, never treat as an Observed \(\lambda\).  
- **Early-fit vs MEASURED** (match `internal/p4-lambda-early-fit.md`; stage a **13/13**):  
  - **Preferred** FITTED exp→asymptote on \(s\ge 61\): point PREDICT \(\approx\mathbf{3.51}\) at 1526 and 9375 — **FIT only**; **undershoots** late MEASURED.  
  - MEASURED 1526 = \(2.1954286466134665\) (**HIT** in `[1.4,4.0)`); preferred ≈3.51 note-only miss. Rebound continues through **7630** (\(5.2815\)) → **9375** (\(\mathbf{5.9673}\)) — still **above** ≈3.51 and **inside** handout `[4.5,6.0]`.  
  - **Bands:** `P4-lam-1526-earlyfit` **`[1.4,4.0)`** **HIT**; `P4-lam-9375-earlyfit` **`[0.3,4.0)`** **MISS** (Observed \(5.967310256347902\)).  
  - Aggressive power-only points ≈**1.43** / ≈**0.36** were **band floors only** — not the preferred PREDICT.  
  - Fit asymptote \(\sim 3.51\) **undershoots** late MEASURED through **9375**. Dens rose mid-train then fell to \(0.06965100420936976\) @9375 — shrink vs init fails → dens-neg **MISS**.  

- **Scored (ledger):** `P4-lam-1526-earlyfit` **HIT**; `P4-lam-9375-earlyfit` **MISS** (\(5.967310256347902\)); `P4-a-norm-sharpness` **HIT**; `P4-a-family-sharpest` **MISS** (late=`head`); `P4-a-density-neg` **MISS**; `P4-a-block-energy` **HIT** (excess_final=0.08727119221832025; fam_above=True; layer_above=True).
- **MEASURED density:** neg_frac 0→153→1526→9375 = \(0.004510114309467541\to 0.01491374251600709\to 0.9344596369432798\to 0.06965100420936976\) — final in `[0,0.15)` but **no shrink** vs init → **MISS**.
- **Counterfactual prediction lessons (predictive MISSes; bands unchanged):**  
  1. **Earlyfit @1526 HIT** — late rebound breaks the early-decay PREDICT: \(\lambda@9375=5.967310256347902\notin[0.3,4.0)\) yet lands in handout `[4.5,6.0]` (`P4-lam-9375-earlyfit` **MISS**).  
  2. **Family PREDICT attn\|mlp late** fails: sharpest=`head` (\(5.641603135007765\)); ratio \(7.38436579244244\notin[1.5,5)\).  
  3. **Dens-neg PREDICT “shrink 0→9375”** fails: final \(0.06965100420936976>\) init \(0.004510114309467541\) (final still ∈`[0,0.15)`).  
  4. Norm mid→final nondecreasing + handout final λ → **HIT** (separate ID).  
- **Scored:** block-energy MEASURED **HIT** — excess_final=0.08727119221832025; fam_above=True; layer_above=True (`a-random-energy-1526/9375`).

**B. Landscape / geometry**  
Families: embed, attn projections, QK-norm gains, MLP, other RMSNorm gains, output head. Off-diagonal blocks = dependencies between variables. If within-block Frobenius energy ≫ between-block (and above random partitions of same sizes), curvature is meaningfully block-diagonal.  
**MEASURED geometry lessons (13/13):**  
- Curvature collapses from init, **minimum** @1526, then **rebounds** through **9375** (\(\lambda=5.967310256347902\)). Late block \(\lambda_{\max}\) dominated by **head**. Fit asymptote \(\sim 3.51\) **undershoots** (FIT only).
- Dens rose to ~0.93 @1526 then \(0.06965\) @9375 — no shrink vs init (**MISS**). Random partition MEASURED — excess_final=0.08727119221832025 (**HIT**).

**C. Dynamics / optimizer interaction**  
Provided AdamW checkpoints at 13 steps from 0→9375 (614M tokens; step 153≈10M, 1526≈100M). AdamW trajectory shapes which blocks sharpen. SGDM comparison ckpts provided for P4(c). See **Mechanism** above for the **13/13** \(\lambda\) minimum@1526 + rebound through **9375** (\(5.967310256347902\)).

**D. Quantitative prediction template** — PREDICT bands locked; MISSes scored file-exact  
- **`P4-lam-*` earlyfit:** bands **`[1.4,4.0)`** / **`[0.3,4.0)`** (preferred FIT ≈3.51 note-only). @1526 **HIT**; @9375 **MISS** — MEASURED \(5.967310256347902\) rebounded into handout `[4.5,6.0]` (early-decay mechanism under-predicted late edge).  
- **`P4-a-family-sharpest`:** PREDICT late attn\|mlp + ratio `[1.5,5)` — **MISS** (late=`head` \(5.641603135007765\); ratio \(7.38436579244244\)). Early embed/head-among-top-2 story partly holds (early_top2=`embedding`,`attention`).  
- **`P4-a-density-neg`:** PREDICT shrink 0→9375 + final `[0,0.15)` — **MISS** on shrink (final \(0.06965100420936976>\) init); final band OK.  
- **`P4-a-block-energy`:** PREDICT family within > random, excess `[0.05,0.30)` — **HIT** (excess_final=0.08727119221832025; fam_above=True; layer_above=True).  
- Handout finals (**assignment-given**): \(\lambda_{\max}\in[4.5,6.0]\), val \(\in[2.925,2.931]\).

**E. Counterfactual levers**  
Checkpoint step choice; `density_runs`; family vs layer partition; extremes vs density-only jobs.

**F. Failure modes / confounds**  
TF32 on; incomplete probe (<64 rows); comparing AdamW vs SGDM without noting optimizer; under-iterated Lanczos residuals.

**G. Cross-links & further tests**  
Sets baseline for P4(b) loss-band interventions. Top eigenspace feeds P4(c). Edge (P3) is about \(\lambda_{\max}\) dynamics; here about full spectrum / blocks.

---

## P4(b) — Move \(\lambda_{\max}\) without moving the loss

**A. Experimental result** — **MEASURED** (`P4-invar-dL` **MISS**)
- Before/after from dry-run `internal/p4-b-score-dryrun.md`: λ 5.967310256347902→2.780040977639008; |ΔL|=2.73853462934494; ratio=0.4658784038724411 → **MISS**.
- Paths: `/cursor/stores/bc-01947f99-cdd8-4c04-97a8-01a1028bf8af/media/p4/b-rescale-reference.json`, `/cursor/stores/bc-01947f99-cdd8-4c04-97a8-01a1028bf8af/media/p4/b-rescale-head-x0.25.json`.
- Training interventions (`P4-b-train-2x`) still **PENDING** (untouched).

**B. Landscape / geometry**  
Loss invariances (lecture / handout hint): reparameterizations that preserve predictions can rescale curvature of affected blocks — so \(\lambda_{\max}\) is **not** a pure quality metric across runs or parameterizations.

**C. Dynamics / optimizer interaction**  
Two routes: (1) hand transform on a final checkpoint (no further training); (2) training-recipe change that induces sharpening “for free.” Hyperparams may get partway; code-level interventions may be stronger (handout).

**D. Quantitative prediction template** — **PREDICTION** (ledger `P4-invar-dL`; Observed **PENDING**) + **MEASURED MISS**  
Hand transform on final ckpt (before = MEASURED reference above):  
- **PREDICTION:** \(\lvert\Delta L_{\mathrm{val}}\rvert\in[0,0.002)\) (scorer uses file-exact \(\lvert\Delta\mathrm{probe\_loss}\rvert\) until val lands).  
- **PREDICTION:** \(\lambda_{\max}^{\mathrm{after}}/\lambda_{\max}^{\mathrm{before}}\in[2,\infty)\) **or** \((0,0.5)\) (direction = rescale choice; \(\ge 2\times\) move).  
- **MEASURED:** |ΔL|=2.73853462934494; λ ratio=0.4658784038724411; λ 5.967310256347902→2.780040977639008; probe_loss 2.9474442899227142→5.685978919267654 → **MISS** (from `internal/p4-b-score-dryrun.md`).
Training route: final \(\lambda_{\max}\) outside \([4.5,6.0]\) by \(\ge 2\times\) factor vs baseline mean, loss inside \([2.925,2.931]\) — pre-register which knob before launch (`P4-b-train-2x`).

**E. Counterfactual levers**  
Which family to rescale; WD / init / LR as soft sharpeners; optional small code change; seed (band is across three seeds).

**F. Failure modes / confounds**  
Moving loss out of band; changing evaluation probe; claiming \(\lambda_{\max}\) comparable across inequivalent parameterizations; mistaking a failed invariance (predictions changed) for a curvature result.

**G. Cross-links & further tests**  
Never treat raw \(\lambda_{\max}\) as comparable without stating parameterization. P3 edge thresholds use \(\eta\lambda\) / \(\eta\lambda^{\mathrm{pre}}\) — still parameterization-sensitive via \(H\). Quiz implication: sharpness ≠ quality.

---

## P4(c) — Training in / out of the top subspace (SGDM)

**A. Experimental result** — **MEASURED**
- From `internal/p4-c-score-dryrun.md` / `media/p4/stage-c/`: mean top_frac=0.09296940652964016 → **MISS**; L_full=3.582527816295624, L_top=5.502330541610718, L_removed=3.5821831226348877, gap=-0.000344693660736084 → **MISS**.
- Optional `P4-c-k-ablate` / stale-evecs still **PENDING** (untouched).

**B. Landscape / geometry**  
Gur-Ari et al.: gradient lies mostly in span of top few Hessian eigenvectors. Top subspace ≈ directions of highest curvature / progress. Removing that span should starve updates of useful progress if the claim holds.

**C. Dynamics / optimizer interaction**  
SGD+momentum easier to reason about than Adam. From SGDM ckpt @ 100M tokens: **zero momentum**, continue 300 steps with transform ∈ {`full`,`top`,`removed`}; refresh eigenvectors every 25 steps. Log losses, \(\|P_{10}g\|/\|g\|\), \(\lambda_{\max}\) vs \(2(1+\beta)/\eta_t\).

**D. Quantitative prediction template** — **PREDICTION** (ledger `P4-top10-frac`) + **MEASURED** + **MEASURED**  
- **PREDICTION:** mean \(\|P_{10}g\|/\|g\|\in[0.6,0.85)\) over the continue.  
- **PREDICTION (ranking):** full \(\approx\) top-10 \(<\) removed (removed worst).  
- **MEASURED (`P4-top10-frac`):** mean=0.09296940652964016 → **MISS**.
- **MEASURED (`P4-c-loss-rank`):** L_full=3.582527816295624; L_top=5.502330541610718; L_removed=3.5821831226348877; gap=-0.000344693660736084 → **MISS**.
- If surprises: vary \(k\), refresh, LR, start point (handout warns results may be unexpected).

**E. Counterfactual levers**  
\(k\), refresh interval, \(\eta\), start checkpoint, \(\beta\).

**F. Failure modes / confounds**  
Stale eigenvectors; Adam instead of SGDM; forgetting to zero momentum at continue; comparing to AdamW \(\lambda_{\max}\) bands without care.

**G. Cross-links & further tests**  
River-valley: top directions may be the “wall-normal” bounce axes. Optional P5 quiz from an unexpected subspace result. Edge (P3): heavy-ball threshold line is the SGDM cousin of \(\eta\lambda=2\).

---

# Problem 5 — Optional own prediction problem

**Label rule:** P5 is **MEASURED** (6/7) from `media/p5/` / `internal/p5-score-dryrun.md`. `P5-B-cool-x0` **MISS**; `P5-dB-cool` **MISS**; `P5-B-const-ctrl` **HIT**; `P5-rank-arms` **MISS**; `P5-G-cool-fitC` **HIT**; `P5-emergent-basin` **MISS**; `P5-L-cool-dir` **INSUFFICIENT** (`dL=null` / \(L_R,L_C\) missing — no HIT/MISS).

**MISS mechanisms (predictive — PREDICT bands locked; scored vs MEASURED).** From `media/p5/` / `internal/p5-score-dryrun.md`:
- **Cooldown did not collapse barrier:** `P5-B-cool-x0` / `P5-dB-cool` / `P5-rank-arms` all **MISS** — B_C=0.7697049379348755 stays ≫0.01; dB=-0.14600493793487546 (negative vs PREDICT `[0.5,0.8)`); ranking \(B_C < B_R\) and \(B_C < B_K\) fails.
- **Constant-LR control stayed high-\(B\):** `P5-B-const-ctrl` **HIT** (B_K=0.7320129871368408 ∈`[0.5,∞)`).
- **Fit C held:** `P5-G-cool-fitC` **HIT** (|G−(B+0.09)|=0.020580991902542256 < 0.05).
- **`P5-emergent-basin`:** PREDICT same-valley walls → MEASURED not-same-valley / JSON emergent='true-modes' (**MISS**).
- **`P5-L-cool-dir`:** **INSUFFICIENT** (dL=null) — no direction takeaway.

## P5(a–b) — Question writeup + compact resolving code

**A. Experimental result** — **MEASURED**
- File-exact: B_R=0.6237, B_C=0.7697049379348755, B_K=0.7320129871368408, G_C=0.8391239460323332, dB=-0.14600493793487546; dL=null.
- Scores: `P5-B-cool-x0` **MISS**; `P5-dB-cool` **MISS**; `P5-B-const-ctrl` **HIT**; `P5-rank-arms` **MISS**; `P5-G-cool-fitC` **HIT**; `P5-emergent-basin` **MISS**; `P5-L-cool-dir` **INSUFFICIENT** (dL / (L_R, L_C) missing).
- Takeaway: cooldown did not collapse the barrier; constant-LR control stayed high-\(B\); Fit C held; emergent='true-modes' (not same-valley).
- Source: `internal/p5-score-dryrun.md` / `media/p5/`.


**B. Landscape / geometry**  
Choose a phenomenon already seen in P1–P4 (barrier, EMA gap, edge retarget, invariance, subspace). Prefer something where the linear path / threshold / ranking is unambiguous.

**C. Dynamics / optimizer interaction**  
Prefer a single clear intervention with a stable predicted interval. Graded runs: seed 42; executable from starter + small diff (handout).

**D. Quantitative prediction template** — **PREDICTION** (author before coding)   + **MEASURED**
- **MEASURED (`P5-B-cool-x0`):** B_C=0.7697049379348755 → **MISS** (cooldown did not collapse barrier to \(\le 0.01\)).
- **MEASURED (`P5-dB-cool`):** dB=-0.14600493793487546 (B_R=0.6237 [JSON] − B_C=0.7697049379348755) → **MISS** (negative; no large collapse vs PREDICT `[0.5,0.8)`).
- **MEASURED (`P5-B-const-ctrl`):** B_K=0.7320129871368408 → **HIT** (constant-LR control stayed high-\(B\)).
- **MEASURED (`P5-rank-arms`):** B_C=0.7697049379348755 B_R=0.6237 B_K=0.7320129871368408 → **MISS** (cooldown arm not below raw or control).
- **MEASURED (`P5-G-cool-fitC`):** G_C=0.8391239460323332 B_C=0.7697049379348755 |G-(B+0.09)|=0.020580991902542256 → **HIT** (Fit C held).
- **MEASURED (`P5-emergent-basin`):** B_C=0.7697049379348755 B_K=0.7320129871368408 → not-same-valley (JSON emergent='true-modes') → **MISS**.
- **INSUFFICIENT (`P5-L-cool-dir`):** dL / (L_R, L_C) missing (`media/p5/` dL=null) — no HIT/MISS.
Mirror A2 quiz structure:  
1. “From the assignment” (reference cell).  
2. “This question changes” (one knob).  
3. Interval / ranking / threshold options with adjacent-half scoring.  
Pre-register the answer interval in the ledger before running the resolving script.

**E. Counterfactual levers**  
Keep scope to <100 LOC; one seed; one metric printed.

**F. Failure modes / confounds**  
Ambiguous grading; needing huge sweeps; answer depending on flaky seeds; inventing “measured” answers in the writeup before the diff runs.

**G. Cross-links & further tests**  
Staff may reuse; align style with `docs/a3-counterfactual-prediction-quiz.md`. Good seeds: P1 ens-vs-avg gap, P2 dip final \(R\), P3 η retarget asymmetry, P4 invariance \(\Delta L\).

---

## Cross-problem prediction habits

1. Name the **reference cell** and the **single changed knob**.  
2. Prefer **interval / ranking / threshold** answers tied to handout constants (\(B\le 0.01\), \(\eta\lambda=2\), \(\eta\lambda^{\mathrm{pre}}=38\)).  
3. Tag every number: measured / fitted / extrapolated / **PREDICTION**.  
4. Pre-register in `docs/a3-prediction-ledger.csv` (see `docs/a3-prediction-ledger.md`) before launch — **MD/CSV only, not a PDF**.  
5. Never paste study-guide speculation into the official report as if measured.  
6. Practice items: optional `docs/a3-counterfactual-prediction-quiz.md` (PDF #3 if exported).  
7. Batch 1 measure landed: P1 §A and ledger Observed filled only from `media/p1/*` — do not backfill invented precision. Batch 1 **FITTED** curves live in Appendix Q1.1; out-of-grid rows are Kind=`extrapolated`.  
8. Batch 2 PREDICTIONs locked `2026-10-11T00:32:07Z`. **MEASURED** @ **9/9**: schedule \(R\), dip **final** \(R\) (**HIT**), both HF \(\Delta L\) (**MISS**) + vs-invsqrt (**HIT**), P3 SGD \(\lambda\) (**HIT**), AdamW \(\eta\lambda^{\mathrm{pre}}\) (**HIT**). Batch 3: **P4(a)** stage-a **13/13** MEASURED + scored; **P5** MEASURED 6/7 (`P5-L-cool-dir` **INSUFFICIENT**). Still **PENDING**: **mid-dip** \(R_{\mathrm{mid}}\), EMA, P3 η-change, **P4(b)/(c)**.

---

# Appendix Q — Quantitative prediction card

Equations and thresholds; compact MD twin: `docs/a3-quantitative-prediction-reference.md` (not a PDF). P1 empirical domain filled from MEASURED Batch 1. P2 + P3 starters **MEASURED** @ 9/9 (both sharpness HITs); **P4(a)** stage-a **13/13** MEASURED + scored; **P5** MEASURED 6/7 (`P5-L-cool-dir` **INSUFFICIENT**). Still **PENDING MEASUREMENT**: **mid-dip / EMA / P3 η-change / P4(b)/(c)**.

## Q1. Mode connectivity (P1)

\[
B(\theta_1,\theta_2)
=
L\!\left(\frac{\theta_1+\theta_2}{2}\right)
-
\frac{L(\theta_1)+L(\theta_2)}{2}
\]

| Symbol / rule | Meaning |
|---------------|---------|
| \(L\) | Validation loss |
| \(B\le 0.01\) | Weight averaging “works” |

\[
\theta_{\mathrm{avg}}=\tfrac12\theta_1+\tfrac12\theta_2,\quad
\theta(\alpha)=(1-\alpha)\theta_1+\alpha\theta_2,\ \alpha\in\{0,0.25,0.5,0.75,1\}
\]

\[
p_{\mathrm{ens}}(\cdot\mid c)=\tfrac12 p_{\theta_1}(\cdot\mid c)+\tfrac12 p_{\theta_2}(\cdot\mid c)
\]

**Grid:** \(x\in\{0,50,100,250,500,1000\}\), \(y\in\{25,50,100,250,500\}\), `wsd0.0`; fork copies model + optimizer state.

**MEASURED domain (Batch 1):**  
- \(26/30\) cells with \(B\le 0.01\); failures \((0,250),(0,500),(50,500),(100,500)\).  
- \(B\in[-0.1161,0.6237]\), mean \(-0.0334\).  
- On failure cells, \(G=L(\theta_{\mathrm{avg}})-L_{\mathrm{ens}}\in[0.1311,0.7281]\) (median \(\approx 0.279\)).  
- Full tables: `docs/a3-experimental-appendix.md`, `media/p1/barrier-table.csv`.  
- Plots (matplotlib): `media/p1/plots/B_heatmap.png`, `B_vs_y_curves.png`, `B_vs_x_y500.png`, `ensemble_vs_avg.png`.

**Ledger:** `P1-B-early-long`, `P1-ens-gap`, `P1-interp-peak`, `P1-grid` → Kind=`measured`, resolved; `P1-d-dB-cos` still **PREDICTION**; extrapolated rows `P1-B-x0-y750`, `P1-B-x25-y500`, `P1-B-x2000-y500`, `P1-G-x0-y750` (Kind=`extrapolated`).

### Q1.1 FITTED relationships (Batch 1 MEASURED only)

No invented measurements. All coefficients from `media/p1/barrier-table.csv` / `barrier-summary.json`. Power laws not forced; preferred forms chosen by \(R^2\) and residual structure among simple candidates (linear / quadratic / \(\log(1+x)\) / \(\exp\)).

#### Fit A — \(B\) vs \(y\) at fixed \(x=0\)

| Model | Domain | Equation | \(R^2\) | RMSE | Residuals (by \(y\)) |
|-------|--------|----------|---------|------|----------------------|
| Quadratic (preferred full curve) | \(x=0\), \(y\in\{25,50,100,250,500\}\) | \(B=-0.0765+7.61\times10^{-4}\,y+1.30\times10^{-6}\,y^2\) | 0.991 | 0.0243 | \(+0.028,-0.006,-0.040,+0.023,-0.004\) |
| Linear growth regime | \(x=0\), \(y\ge 100\) (\(n=3\)) | \(B=-0.1901+0.001628\,y\) | \(\approx 1.000\) | 0.0003 | \(-0.0002,+0.0004,-0.0001\) |

**Validity:** growth-regime linear is almost exact on the three MEASURED points with \(y\ge 100\) but \(n=3\) — treat out-of-grid use as **extrapolated** with widening bands. For \(y\in\{25,50\}\), \(B\) is mildly negative (connected); do not apply the growth-regime line there.  
**Sibling slopes (linear \(B=a+by\), full \(y\) grid):** \(x=50\): \(R^2=0.989\), slope \(5.14\times10^{-4}\); \(x=100\): \(R^2=0.968\), slope \(2.83\times10^{-4}\); \(x=250\): \(R^2=0.833\); \(x\ge 500\): \(R^2\le 0.26\) (flat — no useful \(y\)-trend).

#### Fit B — \(B\) vs \(x\) at fixed \(y=500\)

| Model | Domain | Equation | \(R^2\) | RMSE |
|-------|--------|----------|---------|------|
| Exp decay (**preferred**) | \(y=500\), \(x\in\{0,50,100,250,500,1000\}\) | \(B=-0.0663+0.6870\,e^{-0.02316\,x}\) | 0.995 | 0.0174 |
| \(\log(1+x)\) | same | \(B=0.5850-0.1079\log(1+x)\) | 0.964 | 0.0472 |
| \(a+b/\sqrt{x+1}\) | same | \(B=-0.0651+0.7000/\sqrt{x+1}\) | 0.960 | — |

Exp residuals (by \(x\)): \(+0.003,-0.020,+0.030,+0.012,-0.010,-0.016\).  
**Validity:** preferred on MEASURED \(x\le 1000\) at \(y=500\). As \(x\to\infty\), Fit B asymptotes to \(B\approx -0.066\) (connected). At smaller \(y\), \(\log(1+x)\) fits weaken (\(R^2=0.28\) at \(y=25\); \(0.95\) at \(y=250\)) — barrier-vs-prefix structure is strongest at large \(y\).

#### Fit C — ensemble–avg gap \(G\) vs \(B\)

\[
G=L(\theta_{\mathrm{avg}})-L(p_{\mathrm{ens}}).
\]

| Model | Domain | Equation | \(R^2\) | RMSE |
|-------|--------|----------|---------|------|
| Linear (all cells) | 30 MEASURED cells | \(G=0.0897+1.0328\,B\) | 0.989 | 0.0151 |
| Linear (\(B>0.01\)) | 4 failure cells | \(G=0.1020+1.0066\,B\) | 0.9997 | 0.0039 |

**Identity check:** \(G=B+(\bar L-L_{\mathrm{ens}})\) with \(\bar L=\tfrac12(L_1+L_2)\). MEASURED \(\bar L-L_{\mathrm{ens}}\) has mean \(0.0886\) (std \(0.0158\)), nearly independent of \(B\) (\(R^2=0.086\)) — so \(G\approx B+0.09\) is the mechanistic reading of Fit C.

#### Extrapolations beyond the MEASURED grid (**PREDICTION** / Kind=`extrapolated`)

| Ledger ID | Cell | Point from fits | Interval (grounded) | Models |
|-----------|------|-----------------|---------------------|--------|
| `P1-B-x0-y750` | \((0,750)\) | quad \(1.22\); growth-lin \(1.03\) | \(B\in[0.9,1.4)\) | Fit A |
| `P1-B-x25-y500` | \((25,500)\) | exp \(0.319\); lin-interp MEASURED endpoints \(0.377\) | \(B\in[0.25,0.40)\) | Fit B + MEASURED \((0,500),(50,500)\) |
| `P1-B-x2000-y500` | \((2000,500)\) | exp asymptote \(-0.066\) | \(B\in[-0.12,-0.01)\) | Fit B |
| `P1-G-x0-y750` | \((0,750)\) | \(G(B_{\mathrm{quad}})=1.35\); \(G(B_{\mathrm{lin}})=1.15\) | \(G\in[1.0,1.5)\) | Fit C ∘ Fit A |

Not yet measured — do not treat as Observed.

## Q2. River-valley regret and EMA (P2)

\[
R_M(t)=L(\theta^M_t)-L(\theta^{\mathrm{ref}}_t)
\]

\[
\bar\theta_0=\theta_0,\qquad
\bar\theta_t=\beta_{\mathrm{EMA}}\bar\theta_{t-1}+(1-\beta_{\mathrm{EMA}})\theta_t
\]

Starter EMA decays: \(\{0.99,0.999\}\) (eval only).

**Invsqrt:** \(\alpha_t=\alpha_{\max}\min\{t/w,\sqrt{w/t}\}\) for \(t\ge 1\).  
**SGDR:** \(\alpha_t=\alpha_{\min}+\tfrac12(\alpha_{\max}-\alpha_{\min})[1+\cos(\pi s/T)]\); restart LR only.  
**LR dip:** `lr_dip=(start,bottom,end,scale)`; match control’s final cooldown.  

| Tag | Domain @ Batch 2 finished 9/9 + P3 sharpness |
|-----|------------------------------|
| **MEASURED** | Four schedules; dip **final** \(R=0.0012464523315429688\) (**HIT**); invsqrt \(\Delta L=0.09225320816040039\) (**MISS**); SGDR \(\Delta L=0.10660195350646973\) (**MISS**); \(\lvert L_{\mathrm{sgdr}}-L_{\mathrm{inv}}\rvert=0.014348745346069336\) (**HIT**); P3 SGD \(\lambda=245.16314737956807\) (**HIT**); AdamW \(\eta\lambda^{\mathrm{pre}}=30.987715914514\) (**HIT**) |
| **PENDING** | **Mid-dip** \(R_{\mathrm{mid}}\) (not logged); EMA gap-closed fraction; P3 η-change arms |

### Q2.1 Batch 2 PREDICTION table (locked `2026-10-11T00:32:07Z`)

PREDICTION bands locked before Batch 2 completion (mid-run W&B not used to tune). Manifest: 7 P2 + 2 P3 jobs. Status column separates resolved MEASURED from still-PENDING.

| ID | Job | PREDICTION | Status @ 9/9 |
|----|-----|------------|--------------|
| `B2-P2-wsd0.0-R` | `a3-b2-p2-wsd0.0` | Final \(R\in[0.02,0.08)\); worst of four | **MEASURED** (band MISS; worst HIT) |
| `B2-P2-linear-R` | `a3-b2-p2-linear` | Final \(R\in[0,0.02)\); \(<L_{\mathrm{wsd0.0}}\) | **MEASURED** HIT |
| `B2-P2-cos-R` | `a3-b2-p2-cos` | Final \(R\in[0,0.02)\); top-2 | **MEASURED** HIT |
| `B2-P2-wsd0.2-R` | `a3-b2-p2-wsd0.2` | Final \(R\in[0,0.02)\); top-2 | **MEASURED** (band HIT; top-2 MISS) |
| `P2-ema-gap-frac` | jobs 1–4 EMA | \(f\in[0.5,0.75)\) | **PENDING** (not MEASURED) |
| `B2-P2-dip-R` | dip job | Final \(R\in[-0.02,0.02)\) | **MEASURED** HIT |
| `B2-P2-dip-R` (mid) | same run | mid-dip \(R_{\mathrm{mid}}<-0.02\) | **PENDING** (not MEASURED) |
| `B2-P2-invsqrt-dL` | `a3-b2-p2-invsqrt` | \(\Delta L\in[0.01,0.08)\) vs tuned | **MEASURED** MISS |
| `B2-P2-sgdr-dL` | `a3-b2-p2-sgdr2000` | \(\Delta L\in[0.01,0.08)\); \(\lvert\Delta_{\mathrm{hf}}\rvert\in[0,0.03)\) | **MEASURED** \(\Delta L\) MISS; vs invsqrt HIT |

**PREDICTION pointers:** `P2-sched-rank`, `P2-ema-gap-frac`, `P2-dip-final-R`, `P2-horizon-dL`, `B2-P2-*`. See Q2.1b for MEASURED schedule / dip / invsqrt / SGDR (@ 9/9).

### Q2.1b Batch 2 MEASURED — schedules + dip/invsqrt/SGDR (P2 slice; full batch 9/9)

Jobs 0–6 finished (`media/p2/batch2_partial_finals.csv` exact vals). P3 sharpness also **MEASURED** (see Q3). **P4(a)** stage-a **13/13** MEASURED + scored (see Q4). **P5** MEASURED 6/7 (`P5-L-cool-dir` **INSUFFICIENT**). Still **PENDING**: **mid-dip / EMA / P4(b)/(c)**.  
\(R=L-L^\star\) with \(L^\star=L_{\mathrm{linear}}=2.9282028675079346\).

| Rank | Job | Label | W&B | Final train | Final val | \(R\) | vs PREDICTION |
|------|-----|-------|-----|-------------|-----------|-------|---------------|
| 1 | 1 | `a3-b2-p2-linear` | `r0294p93` | 2.886288642883301 | 2.9282028675079346 | 0 | band \([0,0.02)\) **HIT**; \(<L_{\mathrm{wsd0.0}}\) **HIT** |
| 2 | 2 | `a3-b2-p2-cos` | `sktw67q8` | 2.8933908939361572 | 2.934119462966919 | 0.005916595458984375 | band **HIT**; top-2 **HIT** |
| 3 | 3 | `a3-b2-p2-wsd0.2` | `soosjcxr` | 2.8900909423828125 | 2.93774151802063 | 0.009538650512695312 | band **HIT**; top-2 w/ cos **MISS** |
| 4 | 0 | `a3-b2-p2-wsd0.0` | `widt420l` | 3.0593721866607666 | 3.0975050926208496 | 0.16930222511291504 | worst **HIT**; band \([0.02,0.08)\) **MISS** |

| Job | Label | W&B | Final val | Quantity | vs PREDICTION |
|-----|-------|-----|-----------|----------|---------------|
| 4 | `wsd0.2`+dip | `rms7iyyt` | 2.938987970352173 | \(R_{\mathrm{dip}}=0.0012464523315429688\) vs job3 | \([-0.02,0.02)\) **HIT** |
| 5 | invsqrt | `1jri34em` | 3.020456075668335 | \(\Delta L=0.09225320816040039\) vs linear | \([0.01,0.08)\) **MISS** |
| 6 | SGDR \(T=2000\) | `51b3qq4n` | 3.0348048210144043 | \(\Delta L=0.10660195350646973\) vs linear | \([0.01,0.08)\) **MISS** |
| 6 | (vs invsqrt) | `51b3qq4n` | 3.0348048210144043 | \(\lvert\Delta_{\mathrm{hf}}\rvert=0.014348745346069336\) | \([0,0.03)\) **HIT** |

**`P2-sched-rank` HIT.** Mid-dip / EMA still blank (**PENDING**). Both P3 starter \(\lambda\) / \(\eta\lambda^{\mathrm{pre}}\) **HIT**.

## Q3. Edge of stability (P3)

**Status:** Batch 2 starter IDs **MEASURED HIT** — `B2-P3-sgd-lam` median \(\lambda_{\max}=245.16314737956807\); `B2-P3-adamw-pre` post-warmup median \(\eta\lambda^{\mathrm{pre}}=30.987715914514\). η-change arms (`P3-eta-half` / `P3-eta-double`) remain **PENDING** / **PREDICTION**.

\[
\lambda_{\max}=\lambda_{\max}\bigl(H(\theta)\bigr),\qquad
\eta\lambda_{\max}<2\ \text{(GD/SGD quadratic edge)}
\]

\[
\lambda^{\mathrm{pre}}_{\max}=\lambda_{\max}(D^{-1/2}HD^{-1/2}),\quad D=\sqrt{\hat v}+\epsilon
\]

\[
\eta\lambda^{\mathrm{pre}}_{\max}<2(1+\beta_1)/(1-\beta_1)=38\quad(\beta_1=0.9,\ \text{frozen-precond story})
\]

| Clean-case defaults | Value |
|---------------------|-------|
| \(N\), \(\eta\), updates | 32, 0.01, 1500 |
| Edge at \(\eta=0.01\) | \(\lambda=200\) |
| After \(\eta\leftarrow 0.005\) at 750 | target \(\lambda=400\) |
| After \(\eta\leftarrow 0.02\) at 750 | target \(\lambda=100\) |

**Domain:** SGD clean-edge + AdamW post-warmup \(\eta\lambda^{\mathrm{pre}}\) = **MEASURED** (`media/p2/p3_*_sharpness_summary.json`). Half vs double asymmetry still **PENDING**.

### Q3.1 Batch 2 PREDICTION table (locked `2026-10-11T00:32:07Z`)

| ID | Job | PREDICTION | Status |
|----|-----|------------|--------|
| `B2-P3-sgd-lam` / `P3-clean-edge` | `a3-b2-p3-0-sgd` | Median \(\lambda_{\max}\) (1200–1500) \(\in[150,250)\); late train loss ↓ | **HIT** · \(\lambda=245.16314737956807\) |
| `B2-P3-adamw-pre` / `P3-mb-edge` | `a3-b2-p3-1-adamw` | Post-warmup median \(\eta\lambda^{\mathrm{pre}}\in[30,50)\) | **HIT** · \(30.987715914514\) |
| `P3-eta-half` | *not in B2 required 9* | After \(\eta\leftarrow 0.005\): median \(\lambda\in[350,450)\) | **PENDING** / **PREDICTION** |
| `P3-eta-double` | *not in B2 required 9* | After \(\eta\leftarrow 0.02\): median \(\lambda\in[80,120)\) | **PENDING** / **PREDICTION** |

**Pointers:** `P3-clean-edge` / `B2-P3-sgd-lam` and `P3-mb-edge` / `B2-P3-adamw-pre` both **MEASURED HIT**; η-change arms still PREDICTION / PENDING.

## Q4. Hessian structure (P4)

**Status:** Stage a **13/13** MEASURED. Ledger (from dry-run DRAFT_*): `P4-a-norm-sharpness` **HIT**; `P4-a-family-sharpest` / `P4-a-density-neg` / `P4-lam-9375-earlyfit` **MISS**; `P4-lam-1526-earlyfit` **HIT**; `P4-a-block-energy` **HIT** (excess_final=0.08727119221832025; fam_above=True; layer_above=True). See `internal/p4-9375-hit-miss-summary.md`.

**Predictive MISS notes (bands locked):** late sharpest=`head` not attn\|mlp; dens-neg no shrink 0→9375 (final \(0.06965100420936976>\) init); earlyfit@9375 \(5.967310256347902\) rebounded into handout `[4.5,6.0]` past `[0.3,4.0)`.

- Probe: 64 sequences; FP32, TF32 off; Lanczos + Hutchinson.  
- Handout baseline band (**assignment-given**): \(\lambda_{\max}\in[4.5,6.0]\), val \(\in[2.925,2.931]\).  
- **Early fit:** `P4-lam-1526-earlyfit` **HIT**; `P4-lam-9375-earlyfit` **MISS** (\(5.967310256347902\)).  
- P4(b)/(c) scored: `P4-invar-dL` **MISS**. `P4-a-block-energy` **HIT** (excess_final=0.08727119221832025). Optional train/k-ablate still PENDING.
- Heavy-ball log line: \(2(1+\beta)/\eta_t\).  
**Domain:** resolvable `P4-a-*` + earlyfit scored; block-energy **HIT**; b waiting `internal/p4-b-waiting.md`.

**PREDICTION pointers (still open):** `P4-invar-dL` (PENDING after-arm). `P4-a-block-energy` scored **HIT**.

## Q5. Quick threshold card

Handout / mechanism thresholds only (not Batch-2 MEASURED outcomes). P5 answer = **MEASURED** 6/7 (`P5-L-cool-dir` **INSUFFICIENT**).

| Name | Value | Problem |
|------|-------|---------|
| Averaging works | \(B\le 0.01\) | P1 |
| GD / SGD quadratic edge | \(\eta\lambda_{\max}=2\) | P3 |
| Adam β1=0.9 precond edge | \(\eta\lambda^{\mathrm{pre}}_{\max}=38\) | P3 |
| EMA decays (starter) | \(0.99,\ 0.999\) | P2 |
| P4 loss band | val \(\in[2.925,2.931]\) | P4 |
| P4 sharpness band | \(\lambda_{\max}\in[4.5,6.0]\) | P4 |
| P4(b) move | \(\lambda_{\max}\) change \(\ge 2\times\) | P4 |

## Q6. Ledger pointer (not part of this PDF’s “results”)

Machine-editable rows: `docs/a3-prediction-ledger.csv`. Human schema/notes: `docs/a3-prediction-ledger.md`. Fill **Observed** only from measurements.  
- P1 resolvable (`P1-B-early-long`, `P1-ens-gap`, `P1-interp-peak`, `P1-grid`): Kind=`measured`, Status=`resolved`.  
- Batch 2 scored (`B2-P2-*`, `B2-P3-sgd-lam`, `B2-P3-adamw-pre`, …): Observed filled only where finals exist — mark `resolved` + HIT/MISS.  
- Fit-extrapolated (`P1-B-x0-y750`, …): Kind=`extrapolated`, Observed empty until a future measure.  
- Still **PENDING** / Kind=`predicted`: EMA, mid-dip, P3 η-change, **`P4-invar-dL`** (after-arm), **P4(c)**. (**P4(a)** resolvable rows MEASURED/scored @ 13/13; P4(b) soft baseline MEASURED `scale={}`; `P4-a-block-energy` **HIT**; **P5** MEASURED 6/7; `P5-L-cool-dir` **INSUFFICIENT**.)
