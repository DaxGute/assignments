# A3 quantitative prediction reference

**Role:** Compact MEASURED / FITTED / scored card for CS312 A3. MD only — **not** a PDF (PDF policy: report, intuition guide, optional quiz). Full A–G narrative lives in `docs/a3-predictive-intuition-guide.md` (Appendix Q).  
**Rule:** Every number below is copied from store evidence. No invented coeffs or finals.  
**Sources:** `media/p1/*`, `docs/a3-experimental-appendix.md`, intuition-guide Appendix Q1.1, `media/p2/batch2_partial_finals.csv` (jobs 0–6), `media/p2/p3_*_sharpness_summary.json`, `docs/a3-batch2-prediction-scorecard.md`, `internal/batch2-7of9-sgdr-score-status.md`, `internal/p1-scaling-fit-status.md`, `media/p4/a-adamw-{0,3,15,61,153,305,763,1526,3052,4578,6104,7630,9375}.summary.json` (+ `media/p4/stage-a/`), `internal/p4-lambda-early-fit.md`, `internal/p4-density-soft-0-1526.md`.

Labels: **MEASURED** = observed run/grid; **FITTED** = curve fit to MEASURED; **PREDICTION** = preregistered band; **HIT/MISS** = scored vs PREDICTION; **PENDING** = no Observed yet (Resolved blank).

---

## P1 — Mode connectivity (**MEASURED** Batch 1)

\[
B = L(\theta_{\mathrm{avg}}) - \tfrac12\bigl(L(\theta_1)+L(\theta_2)\bigr),\quad
\theta_{\mathrm{avg}}=\tfrac12(\theta_1+\theta_2).
\]

### MEASURED summary (30-cell grid)

From `media/p1/barrier-summary.json` / `barrier-table.csv` / appendix heatmap:

| Stat | Value |
|------|-------|
| Cells | 30 |
| \(B_{\min}\) | −0.1160578727722168 at \((x,y)=(250,100)\) |
| \(B_{\max}\) | 0.6237311363220215 at \((0,500)\) |
| \(B_{\mathrm{mean}}\) | −0.033404584725697836 |
| \(B\le 0.01\) | **26 / 30** |
| Failures (\(B>0.01\)) | \((0,250)\), \((0,500)\), \((50,500)\), \((100,500)\) |

Heatmap (rounded display; ✓ = \(B\le 0.01\)):

| \(x\) \\ \(y\) | 25 | 50 | 100 | 250 | 500 |
|----------------|------|------|------|------|------|
| 0 | −0.0288 ✓ | −0.0416 ✓ | −0.0275 ✓ | 0.2173 | 0.6237 |
| 50 | −0.1009 ✓ | −0.1053 ✓ | −0.0910 ✓ | 0.0055 ✓ | 0.1299 |
| 100 | −0.0932 ✓ | −0.1019 ✓ | −0.0972 ✓ | −0.0512 ✓ | 0.0319 |
| 250 | −0.0963 ✓ | −0.1106 ✓ | −0.1161 ✓ | −0.0779 ✓ | −0.0522 ✓ |
| 500 | −0.0747 ✓ | −0.0915 ✓ | −0.0953 ✓ | −0.0910 ✓ | −0.0763 ✓ |
| 1000 | −0.0597 ✓ | −0.0769 ✓ | −0.0855 ✓ | −0.0855 ✓ | −0.0825 ✓ |

Plots: `media/p1/plots/B_heatmap.png`, `B_vs_y_curves.png`, `B_vs_x_y500.png`, `ensemble_vs_avg.png`.

### FITTED models A–C (Batch 1 MEASURED only)

Coefficients from intuition-guide **Appendix Q1.1** / `internal/p1-scaling-fit-status.md`. **Never label FITTED as MEASURED.**

#### Fit A — \(B\) vs \(y\) at \(x=0\)

| Model | Domain | Equation | \(R^2\) | RMSE |
|-------|--------|----------|---------|------|
| Quadratic (preferred full curve) | \(x=0\), \(y\in\{25,50,100,250,500\}\) | \(B=-0.0765+7.61\times10^{-4}\,y+1.30\times10^{-6}\,y^2\) | 0.991 | 0.0243 |
| Linear growth regime | \(x=0\), \(y\ge 100\) (\(n=3\)) | \(B=-0.1901+0.001628\,y\) | \(\approx 1.000\) | 0.0003 |

#### Fit B — \(B\) vs \(x\) at \(y=500\)

| Model | Domain | Equation | \(R^2\) | RMSE |
|-------|--------|----------|---------|------|
| Exp decay (**preferred**) | \(y=500\), six MEASURED \(x\) | \(B=-0.0663+0.6870\,e^{-0.02316\,x}\) | 0.995 | 0.0174 |

(Weaker candidates in Q1.1: \(\log(1+x)\), \(a+b/\sqrt{x+1}\) — not preferred.)

#### Fit C — ensemble–avg gap \(G\) vs \(B\)

\[
G=L(\theta_{\mathrm{avg}})-L(p_{\mathrm{ens}}).
\]

| Model | Domain | Equation | \(R^2\) | RMSE |
|-------|--------|----------|---------|------|
| Linear (all cells) | 30 MEASURED cells | \(G=0.0897+1.0328\,B\) | 0.989 | 0.0151 |
| Linear (\(B>0.01\)) | 4 failure cells | \(G=0.1020+1.0066\,B\) | 0.9997 | 0.0039 |

Out-of-grid intervals (`P1-B-x0-y750`, …) are Kind=`extrapolated` **PREDICTION** — not Observed (see ledger / Q1.1).  
P1(d) cosine \(\Delta B\) (`P1-d-dB-cos`) — **PENDING** (no cosine-fork MEASURED).

---

## P2 — Schedule ranking (**MEASURED** @ Batch 2 9/9)

Source: `media/p2/batch2_partial_finals.csv` jobs 0–3; scorecard `docs/a3-batch2-prediction-scorecard.md`.  
Best raw final \(L^\star=2.9282028675079346\) (`linear`). Regret \(R=L-L^\star\).

| Rank | Arm | W&B | Final `val_loss` (exact) | \(R\) vs linear |
|------|-----|-----|--------------------------|-----------------|
| 1 | `linear` | `r0294p93` | 2.9282028675079346 | 0 |
| 2 | `cos` | `sktw67q8` | 2.934119462966919 | 0.005916595458984375 |
| 3 | `wsd0.2` | `soosjcxr` | 2.93774151802063 | 0.009538650512695312 |
| 4 | `wsd0.0` | `widt420l` | 3.0975050926208496 | 0.16930222511291504 |

**Ranking (lower val loss better):**  
\(\mathrm{linear}\prec\mathrm{cos}\prec\mathrm{wsd0.2}\prec\mathrm{wsd0.0}\).

**`P2-sched-rank`:** \(\min(L_{\mathrm{cos}},L_{\mathrm{wsd0.2}},L_{\mathrm{linear}}) < L_{\mathrm{wsd0.0}}\) — **HIT**.

---

## P2 scored counterfactuals (jobs 4–6 — **MEASURED**)

SGDR final (job 6): `a3-b2-p2-sgdr2000` W&B `51b3qq4n`, `val_loss=3.0348048210144043`. Tuned baseline = `linear`.

| ID | Quantity | PREDICTION band | Observed (exact) | Score |
|----|----------|-----------------|------------------|-------|
| `B2-P2-dip-R` / `P2-dip-final-R` | \(R=L_{\mathrm{dip}}-L_{\mathrm{wsd0.2}}\) | \([-0.02,0.02)\) | \(0.0012464523315429688\) (= \(2.938987970352173-2.93774151802063\)) | **HIT** |
| `B2-P2-invsqrt-dL` | \(\Delta L=L_{\mathrm{invsqrt}}-L_{\mathrm{linear}}\) | \([0.01,0.08)\) | \(0.09225320816040039\) (= \(3.020456075668335-2.9282028675079346\)) | **MISS** (falls in \([0.08,\infty)\)) |
| `B2-P2-sgdr-dL` | \(\Delta L=L_{\mathrm{sgdr}}-L_{\mathrm{linear}}\) | \([0.01,0.08)\) | \(0.10660195350646973\) (= \(3.0348048210144043-2.9282028675079346\)) | **MISS** (falls in \([0.08,\infty)\)) |
| `B2-P2-sgdr-dL` (vs invsqrt) | \(\lvert L_{\mathrm{sgdr}}-L_{\mathrm{invsqrt}}\rvert\) | \([0,0.03)\) | \(0.014348745346069336\) (= \(\lvert 3.0348048210144043-3.020456075668335\rvert\)) | **HIT** |

`P2-horizon-dL` (both HF at full stop): invsqrt + SGDR \(\Delta L\) both **MISS**; \|SGDR−inv\| **HIT**.  
Mid-dip \(R_{\mathrm{mid}}\) — **PENDING** (not in partial CSV). EMA gap (`P2-ema-gap-frac`) — **PENDING**.

---

## P3 — Edge of stability (**MEASURED** starters; η-change **PENDING**)

Sources: `media/p2/p3_sgd_sharpness_summary.json`, `media/p2/p3_adamw_sharpness_summary.json`.

| ID | Quantity | PREDICTION band | Observed (exact) | Score |
|----|----------|-----------------|------------------|-------|
| `B2-P3-sgd-lam` / `P3-clean-edge` | Median \(\lambda_{\max}\) updates 1200–1500 | \([150,250)\) | \(245.16314737956807\) | **HIT** |
| `B2-P3-adamw-pre` / `P3-mb-edge` | Post-warmup median \(\eta\lambda^{\mathrm{pre}}\) | \([30,50)\) | \(30.987715914514\) | **HIT** |
| `P3-eta-half` / `P3-eta-double` | Mid-run \(\eta\) ×0.5 / ×2 | see ledger | — | **PENDING** (not in B2 required 9) |
| P3(c) hparam exploration | — | — | — | **PENDING** |

Handout thresholds (mechanism, not Observed): GD edge \(\eta\lambda=2\) ⇒ \(\lambda=200\) at \(\eta=0.01\); Adam \(\beta_1=0.9\) precond edge \(38\).

---

## P4 — Hessian trajectory (**MEASURED** stage a **13/13**)

Sources: `media/p4/stage-a/a-adamw-{0,3,15,61,153,305,763,1526,3052,4578,6104,7630,9375}.json` (+ summaries). Stage a **DONE 13/13**.

### MEASURED card (\(\lambda_{\max}\))

| Job | Step | \(\lambda_{\max}\) (exact) | Status |
|-----|------|----------------------------|--------|
| `a-adamw-0` | 0 | 1431.5030691763686 | **MEASURED** |
| `a-adamw-3` | 3 | 1571.7420262270407 | **MEASURED** |
| `a-adamw-15` | 15 | 53.36426571742344 | **MEASURED** |
| `a-adamw-61` | 61 | 18.054717868295125 | **MEASURED** |
| `a-adamw-153` | 153 | 6.502395300121604 | **MEASURED** |
| `a-adamw-305` | 305 | 4.32926630998023 | **MEASURED** |
| `a-adamw-763` | 763 | 3.02398152405625 | **MEASURED** |
| `a-adamw-1526` | 1526 | 2.1954286466134665 | **MEASURED** |
| `a-adamw-3052` | 3052 | 3.1637529312879873 | **MEASURED** |
| `a-adamw-4578` | 4578 | 3.7423705126158477 | **MEASURED** |
| `a-adamw-6104` | 6104 | 4.155164815834522 | **MEASURED** |
| `a-adamw-7630` | 7630 | 5.281516516770698 | **MEASURED** |
| `a-adamw-9375` | 9375 | 5.967310256347902 | **MEASURED** |

**Trajectory:** collapse by 15 → minimum @1526 (\(2.195\)) → rebound \(2.20\to 3.16\to 3.74\to 4.16\to 5.28\to\mathbf{5.9673}\) through **9375** (exact \(5.967310256347902\); **inside** handout `[4.5,6.0]`). Preferred early-fit asymptote ≈\(3.51\) still **undershoots**. See `internal/p4-lambda-rebound-13of13.md`.

**Density (file-exact → ledger):** neg_frac 0→153→1526→9375 = \(0.004510114309467541\to 0.01491374251600709\to 0.9344596369432798\to 0.06965100420936976\). Final in `[0,0.15)` but **did not shrink** vs init → `P4-a-density-neg` **MISS**.

### Early-fit scorecard

| ID | Band | Preferred point | Observed | Score |
|----|------|-----------------|----------|-------|
| `P4-lam-1526-earlyfit` | `[1.4,4.0)` | ≈3.51 | **2.1954286466134665** | **HIT** (preferred note-only miss) |
| `P4-lam-9375-earlyfit` | `[0.3,4.0)` | ≈3.51 | **5.967310256347902** | **MISS** (preferred also miss; inside handout `[4.5,6.0]`) |

### P4-a scorecard (from `internal/p4-score-dryrun.md` → ledger)

| ID | PREDICT (locked) | Score | Observed (file-exact) | Predictive mechanism note |
|----|------------------|-------|------------------------|---------------------------|
| `P4-a-family-sharpest` | late attn\|mlp; ratio `[1.5,5)` | **MISS** | late=`head` (\(5.641603135007765\)); early_top2=`embedding`,`attention`; ratio=\(7.38436579244244\) | Pre-reg: residual-block late sharpening. Fail: head stayed sharpest @9375. |
| `P4-a-norm-sharpness` | mid→final nondec.; final λ in `[4.5,6.0]` | **HIT** | \(\lambda\) \(2.1954286466134665\to5.967310256347902\); tr nondecreasing; final in `[4.5,6.0]` | Edge-band + mid→final sharpening held. |
| `P4-a-density-neg` | shrink 0→9375; final `[0,0.15)` | **MISS** | \(0.004510114309467541\to\cdots\to0.06965100420936976\); shrink_0→9375=false; final band OK | Pre-reg: saddle cleanup. Fail: final neg_frac > init. |
| `P4-lam-9375-earlyfit` | λ@9375 ∈`[0.3,4.0)` | **MISS** | \(5.967310256347902\) (inside handout `[4.5,6.0]`) | Pre-reg: early-decay / asymptote \(\sim 3.51\). Fail: rebound into handout edge. |
| `P4-a-block-energy` | family>random; excess `[0.05,0.30)` | **HIT** | family_within-random excess_final=0.08727119221832025; fam_above=True; layer_above=True | MEASURED `a-random-energy-1526/9375`. |

Operator copied DRAFT_* into ledger (script `--apply` does not mutate docs). Summary: `internal/p4-9375-hit-miss-summary.md`. PREDICT bands unchanged.

### P4(b) soft baseline (mirrors report — not an invar score)

| Field | Value |
|-------|-------|
| Source | `media/p4/b-rescale-reference.json` |
| Status | **MEASURED** baseline only (`scale={}`) |
| \(\lambda_{\max}\) | \(5.967310256347902\) (≈5.967; matches stage-a @9375) |
| `probe_loss` | \(2.9474442899227142\) |
| Hand-rescale arms | `b-rescale-head-x0.25` / `x0.5` / `x2` — **launched / waiting** (`ap-lSeukZDR7yWisgODr61i1k`) |
| `P4-invar-dL` | **MISS** — |ΔL|=2.73853462934494; λ ratio=0.4658784038724411; λ 5.967310256347902→2.780040977639008; probe_loss 2.9474442899227142→5.685978919267654 (from `internal/p4-b-score-dryrun.md`) |
| Dry-run scorer | `internal/p4-score-invar-from-media.py` (exit 2 until after-arm) |
| Waiting note | `internal/p4-b-waiting.md` |

PREDICT (locked, unscored): \(\lvert\Delta L\rvert\in[0,0.002)\); λ ratio \(\in[2,\infty)\cup(0,0.5)\).


### P4 b/c scorecard (from dry-run → deliverables)

| ID | PREDICT (locked) | Score | Observed (file-exact) |
|----|------------------|-------|------------------------|
| `P4-top10-frac` | mean frac ∈[0.6,0.85) | **MISS** | mean top_frac=0.09296940652964016 (source=c-full) |
| `P4-c-loss-rank` | full≈top<removed; gap ∈[0.02,0.20) | **MISS** | L_full=3.582527816295624; L_top=5.502330541610718; L_removed=3.5821831226348877; gap=-0.000344693660736084 |




## P5 — Cooldown barrier (MEASURED from dry-run)

Source: `media/p5/` / `internal/p5-score-dryrun.md`.

| ID | PREDICT (locked) | Score | Observed (file-exact) |
|----|------------------|-------|------------------------|
| `P5-B-cool-x0` | B_C ∈ (-∞,0.01] | **MISS** | B_C=0.7697049379348755 |
| `P5-dB-cool` | ΔB ∈ [0.5,0.8) | **MISS** | dB=-0.14600493793487546 (B_R=0.6237 [JSON] − B_C=0.7697049379348755) |
| `P5-B-const-ctrl` | B_K ∈ [0.5,∞) | **HIT** | B_K=0.7320129871368408 |
| `P5-rank-arms` | B_C < B_R and B_C < B_K | **MISS** | B_C=0.7697049379348755 B_R=0.6237 B_K=0.7320129871368408 |
| `P5-G-cool-fitC` | if B_C≤0.01: G∈[0,0.12); else |G-(B+0.09)|<0.05 | **HIT** | G_C=0.8391239460323332 B_C=0.7697049379348755 |G-(B+0.09)|=0.020580991902542256 |
| `P5-L-cool-dir` | L_C<L_R; ΔL ∈ [0.02,0.15) | **INSUFFICIENT** | dL / (L_R, L_C) missing (`media/p5/` dL=null) — no HIT/MISS |
| `P5-emergent-basin` | same-valley (B_C≤0.01 and B_K>0.01) | **MISS** | B_C=0.7697049379348755 B_K=0.7320129871368408 → not-same-valley (JSON emergent='true-modes') |




## PENDING (no Observed — do not invent)

| Topic | Status | Notes |
|-------|--------|-------|
| Mid-dip \(R_{\mathrm{mid}}\) | **PENDING** | Not in `batch2_partial_finals.csv` |
| EMA gap (`P2-ema-gap-frac`) | **PENDING** | No scored EMA series |
| P3 η-change / P3(c) | **PENDING** | Not in required 9-job finals |
| **P4** `P4-a-block-energy` | **HIT** | excess_final=0.08727119221832025; fam_above=True; layer_above=True |
| **P5** `P5-L-cool-dir` | **INSUFFICIENT** | dL=null / (L_R, L_C) missing — no invent HIT/MISS |

Quiz answer key: Resolved filled only for MEASURED / FITTED / EXTRAPOLATED rows; `P4-a-block-energy` **HIT** (Resolved filled); `P5-L-cool-dir` stays Resolved = — · PENDING/INSUFFICIENT (**P5** otherwise MEASURED 6/7).

---

## Cross-links

| Resource | Path |
|----------|------|
| Full A–G + Appendix Q | `docs/a3-predictive-intuition-guide.md` |
| Practice quiz + key | `docs/a3-counterfactual-prediction-quiz.md` |
| Experimental tables | `docs/a3-experimental-appendix.md` |
| Batch 2 scorecard | `docs/a3-batch2-prediction-scorecard.md` |
| Ledger | `docs/a3-prediction-ledger.md` / `.csv` |
| Fit status | `internal/p1-scaling-fit-status.md` |
| P4 early λ fit | `internal/p4-lambda-early-fit.md` |
| P1 media | `media/p1/` |
| P2 finals CSV | `media/p2/batch2_partial_finals.csv` |
| P3 sharpness JSON | `media/p2/p3_*_sharpness_summary.json` |
