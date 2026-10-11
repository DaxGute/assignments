# A3 Prediction Ledger

**Format:** Markdown notes + machine table `docs/a3-prediction-ledger.csv`.  
**Not a PDF** — keep as store MD/CSV only (see PDF policy in `docs/a3-predictive-intuition-guide.md`).

Pre-register counterfactual predictions **before** launching the corresponding runs. Never invent measured outcomes — leave Observed empty until data exist. Prefer editing the CSV when resolving rows; keep this MD as the human-readable schema + mirror.

**Quiz alignment:** practice quiz `docs/a3-counterfactual-prediction-quiz.md` cites ledger IDs; CSV column `quiz_ref` lists `Q#` parts.

---

## Batch 2 lock (P2/P3) — PREDICTION only

| Field | Value |
|-------|-------|
| **Locked at (UTC)** | `2026-10-11T00:32:07Z` |
| **Lock note** | Pre-registered **before Batch 2 completion**. No Batch 2 finals used. Mid-run W&B from `internal/mac-gpu-status.md` was **not** used as outcomes or to tune bands. |
| **Kind** | `predicted` (human label: **PREDICTION**). Observed empty until MEASURED finals. |
| **Evidence basis** | Handout (`internal/a3_optimization2.txt`) + guide P2/P3; A2-style intervals. Manifest: `internal/batch2-prep.md` / `internal/manifests/batch2_manifest.csv` (9 jobs). |
| **Status file** | `internal/p2-p3-preregister-status.md` |

### Batch 2 job → PREDICTION map (9 jobs)

Regret \(R\) for schedule jobs is vs the best raw final among `{wsd0.0, linear, cos, wsd0.2}`. Horizon-free \(\Delta L\) vs that same best **budget-tuned** schedule at the full 614M-token stop. Absolute final val losses are **not** invented.

| # | Manifest label | Prediction ID | Direction (PREDICTION) | Magnitude band (PREDICTION) | Quiz |
|---|----------------|---------------|------------------------|------------------------------|------|
| 1 | `a3-b2-p2-wsd0.0` | `B2-P2-wsd0.0-R` | Worst raw final among four | Final \(R\in[0.02,0.08)\) | Q22a |
| 2 | `a3-b2-p2-linear` | `B2-P2-linear-R` | Better than `wsd0.0` | Final \(R\in[0,0.02)\) | Q22b |
| 3 | `a3-b2-p2-cos` | `B2-P2-cos-R` | Among top-2 with `wsd0.2` | Final \(R\in[0,0.02)\) | Q22b |
| 4 | `a3-b2-p2-wsd0.2` | `B2-P2-wsd0.2-R` | Among top-2 with `cos` | Final \(R\in[0,0.02)\) | Q22b |
| 5 | `a3-b2-p2-wsd0.2` (dip) | `B2-P2-dip-R` (=`P2-dip-final-R`) | Final gain does not survive rewarm | Final \(R\in[-0.02,0.02)\); mid-dip \(R_{\mathrm{mid}}\in(-\infty,-0.02)\) | Q6 |
| 6 | `a3-b2-p2-invsqrt` | `B2-P2-invsqrt-dL` | Worse than best tuned cooldown | \(\Delta L\in[0.01,0.08)\) | Q23a / Q11 |
| 7 | `a3-b2-p2-sgdr2000` | `B2-P2-sgdr-dL` | Same HF story as invsqrt | \(\Delta L\in[0.01,0.08)\); \(\lvert L_{\mathrm{sgdr}}-L_{\mathrm{invsqrt}}\rvert\in[0,0.03)\) | Q23b |
| 8 | `a3-b2-p3-0-sgd` | `B2-P3-sgd-lam` (=`P3-clean-edge`) | Progressive sharpen → ride \(2/\eta=200\) | Median \(\lambda_{\max}\) (1200–1500) \(\in[150,250)\); late train loss ↓ | Q24a / Q12 |
| 9 | `a3-b2-p3-1-adamw` | `B2-P3-adamw-pre` (=`P3-mb-edge`) | Near Adam threshold 38 | Post-warmup median \(\eta\lambda^{\mathrm{pre}}\in[30,50)\) | Q24b / Q8 |

**Cross-cutting EMA (jobs 1–4):** `P2-ema-gap-frac` — \(f\in[0.5,0.75)\); after EMA, cos/`wsd0.2` still favored (Q5).  
**Not in Batch 2 required 9:** `P3-eta-half`, `P3-eta-double` (P3(a)ii).

## Schema

| Column | Meaning |
|--------|---------|
| ID | Stable row id (`P{problem}-{theme}-{short}`) |
| Theme | P1–P4 theme |
| Baseline cell | Reference setting from the handout / starter |
| Intervention | Single change vs baseline (counterfactual) |
| Predicted interval / ranking | **Numerical** interval, ranking, or threshold choice — written before the run |
| Observed | Fill only from measured / fitted data; leave blank until then |
| Kind | `predicted` \| `measured` \| `fitted` \| `extrapolated` |
| Status | `preregistered` \| `running` \| `resolved` \| `superseded` \| `partial` (soft early evidence; not scored HIT/MISS) |
| Mechanism cite | Handout section / equation |
| Quiz ref | Practice quiz item(s), e.g. `Q1a,Q1b` |

## Legend (Kind)

| Tag | Meaning | Quiz answer-key label |
|-----|---------|------------------------|
| predicted | Written before the run | **PREDICTION** |
| measured | Direct experimental outcome | **MEASURED** |
| fitted | Curve/fit derived from measured points (in-sample) | **FITTED** |
| extrapolated | Out-of-sample from a fit (not measured) | **EXTRAPOLATED** |

Interval convention (A2-style): pick one option; adjacent interval = half credit when graded.

---

## Pre-registered counterfactuals (quiz-aligned)

Observed filled only from MEASURED Batch 1 JSON / FITTED Appendix Q1.1 where applicable. Keep original PREDICT bands; Status=`resolved` when Observed is set from real cells. P2–P4 Batch 3 arms remain preregistered / Observed empty. P4 Batch 3 draft rows merged below (`docs/a3-p4-preregister-draft.md`). Batch 2 job-level `B2-*` PREDICTIONs: status `internal/p2-p3-preregister-status.md` · locked `2026-10-11T00:32:07Z` (section above + table below).

| ID | Theme | Quiz | Baseline cell | Intervention | Predicted interval / ranking | Observed | Kind | Status | Mechanism cite |
|----|-------|------|---------------|--------------|------------------------------|----------|------|--------|----------------|
| P1-B-early-long | P1 | Q1 | `(x,y)=(1000,25)` vs `(0,500)`, `wsd0.0`, seeds `{1,2}` | Contrast barriers at the two cells | **(i)** \(B(0,500)\): **PREDICT `[0.05,0.2)`**; **(ii)** \(B(1000,25)\): **PREDICT `(−∞,0.01]`** | **MEASURED:** \(B(0,500)=0.6237\in[0.2,\infty)\) (miss); \(B(1000,25)=-0.0597\in(-\infty,0.01]\) (hit) | measured | resolved | P1 \(B\le0.01\); shared prefix |
| P1-ens-gap | P1 | Q2 | Cells with measured \(B>0.01\) (after grid) | \(L(\theta_{\mathrm{avg}})-L(p_{\mathrm{ens}})\) | Interval for gap: **PREDICT `[0.02,0.08)`** (ensemble better; gap grows with \(B\)) | **MEASURED:** gaps \(0.1311,0.2309,0.3273,0.7281\); median \(\approx0.279\in[0.2,\infty)\) (miss magnitude; all \(G\ge0.01\)) | measured | resolved | P1(a–b) ens vs weight avg |
| P1-interp-peak | P1 | Q3 | High-\(B\) representative cell | Full \(\alpha\) curve | Ranking: **PREDICT** \(\alpha=0.5\) worst; height **PREDICT `[0.05,0.2)`** | **MEASURED** on mid-band cell \((50,500)\): \(\alpha=0.5\) worst; height \(B=0.1299\in[0.05,0.2)\) (hit). Extreme \((0,500)\): height \(0.6237\in[0.2,\infty)\) | measured | resolved | P1(c) |
| P1-d-dB-cos | P1 | Q4 | `(x,y)=(250,250)`, `wsd0.0` branches | Both branches use cosine decay after fork | \(\Delta B=B_{\mathrm{wsd0.0}}-B_{\mathrm{cos}}\): **PREDICT `[0.02,0.1)`** | — (note: MEASURED \(B_{\mathrm{wsd}}(250,250)=-0.0779\) already \(\le0.01\); cosine arm unmeasured) | predicted | preregistered | P1(d)+river-valley |
| P1-ens-vs-endpoint | P1 | Q13 | Cell `(0,500)`, endpoints + ensemble | \(G_{\mathrm{ens}}=L_\star-L(p_{\mathrm{ens}})\); rank ens vs \(\theta_{\mathrm{avg}}\) | **PREDICT** \(G_{\mathrm{ens}}\in[0,0.01)\); **PREDICT** \(L(p_{\mathrm{ens}})<L(\theta_{\mathrm{avg}})-0.01\) | **MEASURED:** \(G_{\mathrm{ens}}=0.1029\ge0.01\) (miss); \(L_{\mathrm{ens}}=3.6454\) vs \(L_{\mathrm{avg}}=4.3734\) (hit) | measured | resolved | P1(a) |
| P1-B-x0-y250 | P1 | Q18 | Grid cell `(0,250)` | Early fork, medium branch | \(B(0,250)\): **PREDICT `[0.2,\infty)`**; ranking vs \(y\in\{100,250,500\}\) at \(x=0\): **PREDICT** \(B(0,100)<B(0,250)<B(0,500)\) | **MEASURED:** \(B=0.2173\in[0.2,\infty)\); ranking holds (\(B(0,100)=-0.0275\)) | measured | resolved | P1(b) |
| P1-B-x100-y500 | P1 | Q19 | Grid cell `(100,500)` | Mildest failure cell | \(B\): **PREDICT `(0.01,0.05)`**; \(G\): **PREDICT `[0.05,0.15)`** | **MEASURED:** \(B=0.0319\); \(G=0.1311\) | measured | resolved | P1(a–b) |
| P1-G-fitC-mid | P1 | Q21 | Cell `(50,500)`, \(B=0.1299\) MEASURED | Apply Fit C \(G\)–vs–\(B\) | \(G\): **PREDICT `[0.20,0.25)`**; mechanism **PREDICT** \(G\approx B+0.09\) | **FITTED** \(G\approx0.233\); **MEASURED** \(G=0.2309\in[0.20,0.25)\) | fitted | resolved | Q1.1 Fit C |
| P2-ema-gap-frac | P2 | Q5 | Four schedules; peak LR tuned; ref = best raw final | Eval-only EMA \(\beta\in\{0.99,0.999\}\) | Fraction of raw best−worst schedule gap closed by best EMA: **PREDICT `[0.5,0.75)`** | — | predicted | preregistered | P2(b) EMA · **B2-locked 2026-10-11T00:32:07Z** |
| P2-dip-final-R | P2 | Q6 | Control + matched final cooldown | Temporary `lr_dip` then rewarm | Final regret \(R=L_{\mathrm{dip}}-L_{\mathrm{ctrl}}\): **PREDICT `[−0.02,0.02)`** | **MEASURED:** \(R=0.0012464523315429688\) vs `wsd0.2` control; band **HIT**; mid-dip unscored (not in CSV) | measured | resolved | P2(c) · (=`B2-P2-dip-R`) |
| P3-eta-half | P3 | Q7a | Full-batch GD \(N=32\), \(\eta=0.01\), at edge ~200 | Halve \(\eta\) at step 750 | Median \(\lambda_{\max}\) on steps 1200–1500: **PREDICT `[350,450)`** | — | predicted | preregistered | P3(a)ii; \(2/\eta\) · not in B2 required 9 |
| P3-eta-double | P3 | Q7b | Same clean case | Double \(\eta\) at step 750 | Median \(\lambda_{\max}\) on steps 1200–1500: **PREDICT `[80,120)`** | — | predicted | preregistered | P3(a)ii · not in B2 required 9 |
| P3-mb-edge | P3 | Q8 / Q24b | `minibatch()` AdamW, constant LR, post-warmup | Measure \(\eta_t\lambda^{\mathrm{pre}}_{\max}\) | Median post-warmup \(\eta\lambda^{\mathrm{pre}}\): **PREDICT `[30,50)`** (near 38) | **MEASURED:** \(30.987715914514\) (**HIT**; n=25) | measured | resolved | P3(b); threshold 38 · **B2-locked** (=`B2-P3-adamw-pre`); `p3_adamw_sharpness_summary.json` |
|  P4-invar-dL | P4 | Q9 | Final AdamW ckpt; handout bands \(\lambda\in[4.5,6.0]\), val \(\in[2.925,2.931]\) | Hand loss-invariant family rescale (`b-rescale`; no further training) | \(\lvert\Delta L_{\mathrm{val}}\rvert\): **PREDICT `[0,0.002)`**; \(\lambda\) ratio **PREDICT `[2,\infty)` or `(0,0.5)`**; scaled blocks show largest relative \(\lambda\) change | **MEASURED:** |ΔL|=2.73853462934494; λ ratio=0.4658784038724411; λ 5.967310256347902→2.780040977639008; probe_loss 2.9474442899227142→5.685978919267654 → **MISS** (from dry-run) | measured | resolved | P4(b) invariance  |
|  P4-top10-frac | P4 | Q10a | SGDM @ 100M tok (step 1526); continue 300; refresh 25; \(k=10\) | Mean grad fraction in top-10 Hessian span | Mean \(\|P_{10}g\|/\|g\|\): **PREDICT `[0.6,0.85)`** (ranking → `P4-c-loss-rank`) | **MEASURED:** mean top_frac=0.09296940652964016 (source=c-full) → **MISS** (from dry-run) | measured | resolved | P4(c); Gur-Ari  |
| P2-horizon-dL | P2 | Q11 / Q23 | Budget-tuned cos/`wsd0.2` with long cooldown | Fixed invsqrt/SGDR \((w,T)\) no budget cooldown at \(T_{\mathrm{long}}\) | \(\Delta L=L_{\mathrm{hf}}-L_{\mathrm{tuned}}\): **PREDICT `[0.01,0.08)`**; mid budget: **PREDICT** \(\lvert\Delta L\rvert<0.02\) | **MEASURED:** invsqrt \(\Delta L=0.09225320816040039\) **MISS**; SGDR \(\Delta L=0.10660195350646973\) **MISS**; \(\lvert\mathrm{SGDR}-\mathrm{inv}\rvert=0.014348745346069336\) **HIT**; mid-budget unscored | measured | resolved | P2(d); both HF vs linear |
| P3-clean-edge | P3 | Q12 / Q24a | Full-batch GD \(N=32\), \(\eta=0.01\), 1500 updates | No mid-run η change; median \(\lambda_{\max}\) on 1200–1500 | **PREDICT `[150,250)`** (toward \(2/\eta=200\)); late train loss ↓ | **MEASURED:** median \(\lambda_{\max}=245.16314737956807\) (**HIT**); late train ↓ | measured | resolved | P3(a)i · **B2-locked** (=`B2-P3-sgd-lam`) |

**Count (this section):** 17 quiz-aligned counterfactual rows (+ 9 `B2-*` job rows below). P1 Batch 1 + all nine Batch 2 `B2-*` P2/P3 rows with finals are `resolved`; EMA / mid-dip stay preregistered (Observed blank).

---

## Batch 2 job-level rows

Locked `2026-10-11T00:32:07Z` before Batch 2 completion. Schedule \(R\), dip final \(R\) (**HIT**), invsqrt/SGDR \(\Delta L\), P3 SGD \(\lambda\) (**HIT**), and AdamW \(\eta\lambda^{\mathrm{pre}}\) (**HIT**) filled from CSV + `media/p2/p3_*_sharpness_summary.json`. EMA / mid-dip Observed blank — do not invent. CSV mirrors these IDs.

| ID | Theme | Quiz | Baseline cell | Intervention | Predicted interval / ranking | Observed | Kind | Status | Mechanism cite |
|----|-------|------|---------------|--------------|------------------------------|----------|------|--------|----------------|
| B2-P2-wsd0.0-R | P2 | Q22a | Best of `{linear,cos,wsd0.2,wsd0.0}` | Job `a3-b2-p2-wsd0.0` | Final \(R\): **PREDICT `[0.02,0.08)`**; worst of four | **MEASURED:** \(R=0.16930222511291504\) vs best=`linear`; worst HIT; band **MISS** (too high) | measured | resolved | P2(a); `batch2_partial_finals.csv` |
| B2-P2-linear-R | P2 | Q22b | Best of four | Job `a3-b2-p2-linear` | Final \(R\): **PREDICT `[0,0.02)`**; \(L_{\mathrm{linear}}<L_{\mathrm{wsd0.0}}\) | **MEASURED:** \(R=0\) (best); \(L_{\mathrm{lin}}<L_{\mathrm{wsd0.0}}\) HIT; band HIT | measured | resolved | P2(a); CSV |
| B2-P2-cos-R | P2 | Q22b | Best of four | Job `a3-b2-p2-cos` | Final \(R\): **PREDICT `[0,0.02)`**; top-2 with `wsd0.2` | **MEASURED:** \(R=0.005916595458984375\in[0,0.02)\) HIT; rank #2 (top-2 HIT) | measured | resolved | P2(a); CSV |
| B2-P2-wsd0.2-R | P2 | Q22b | Best of four | Job `a3-b2-p2-wsd0.2` (no dip) | Final \(R\): **PREDICT `[0,0.02)`**; top-2 with `cos` | **MEASURED:** \(R=0.009538650512695312\in[0,0.02)\) HIT; rank #3 (top-2 **MISS**) | measured | resolved | P2(a); CSV |
| B2-P2-dip-R | P2 | Q6 | Matched `wsd0.2` control (`soosjcxr`) | Job4 `lr_dip` `rms7iyyt` | Final \(R\): **PREDICT `[-0.02,0.02)`**; mid-dip **PREDICT** \((-\infty,-0.02)\) | **MEASURED:** \(R=0.0012464523315429688\in[-0.02,0.02)\) **HIT**; mid-dip not logged | measured | resolved | P2(c) alias `P2-dip-final-R` |
| B2-P2-invsqrt-dL | P2 | Q23a / Q11a | Best of `{linear,cos,wsd0.2}` = linear | Job5 `a3-b2-p2-invsqrt` | \(\Delta L\): **PREDICT `[0.01,0.08)`** | **MEASURED:** \(\Delta L=0.09225320816040039\); band **MISS** (in \([0.08,\infty)\)) | measured | resolved | P2(d); CSV |
| B2-P2-sgdr-dL | P2 | Q23b | Same tuned ref (=linear) | Job6 `a3-b2-p2-sgdr2000` `51b3qq4n` | \(\Delta L\): **PREDICT `[0.01,0.08)`**; vs invsqrt \(\lvert\Delta\rvert\in[0,0.03)\) | **MEASURED:** \(\Delta L=0.10660195350646973\) vs linear **MISS**; \(\lvert L_{\mathrm{sgdr}}-L_{\mathrm{inv}}\rvert=0.014348745346069336\) **HIT** | measured | resolved | P2(d); CSV |
| B2-P3-sgd-lam | P3 | Q24a | Cohen \(2/\eta=200\) | Job `a3-b2-p3-0-sgd` | Median \(\lambda_{\max}\): **PREDICT `[150,250)`**; late train loss ↓ | **MEASURED:** median \(\lambda_{\max}=245.16314737956807\) (**HIT**); late train ↓ | measured | resolved | P3(a)i alias `P3-clean-edge`; `p3_sgd_sharpness_summary.json` |
| B2-P3-adamw-pre | P3 | Q24b | Adam threshold 38 | Job `a3-b2-p3-1-adamw` | Post-warmup median \(\eta\lambda^{\mathrm{pre}}\): **PREDICT `[30,50)`** | **MEASURED:** \(30.987715914514\) (**HIT**; n=25) | measured | resolved | P3(b) alias `P3-mb-edge`; `p3_adamw_sharpness_summary.json` |

---

## P4 Batch 3 pre-registrations (from draft)

Source: `docs/a3-p4-preregister-draft.md` · dry-run `internal/p4-score-dryrun.md` · runbook `internal/p4-on-9375-runbook.md` · summary `internal/p4-9375-hit-miss-summary.md`.  
Stage a **13/13** MEASURED (`a-adamw-9375.json`). Apply-gate **READY**. `P4-a-block-energy` **HIT** (MEASURED `random_within*` from `a-random-energy-1526/9375`; excess_final=0.08727119221832025). Scorer `--apply` resolved can_resolve `P4-a-*`. Overlap IDs `P4-invar-dL` / `P4-top10-frac` also in quiz-aligned table above.

| ID | Theme | Quiz | Baseline cell | Intervention | Predicted interval / ranking | Observed | Kind | Status | Mechanism cite |
|----|-------|------|---------------|--------------|------------------------------|----------|------|--------|----------------|
| P4-a-family-sharpest | P4 | Q26 | AdamW ckpts; per-family block \(\lambda_{\max}\) | Late sharpest family + early vs late | Late: **PREDICT** `attention` or `mlp` sharpest (tied-ok); early: `embedding`/`head` in top-2 more often than at 614M; ratio sharpest/median final **PREDICT `[1.5,5)`** | **MEASURED:** late_sharpest=`head` (\(5.641603135007765\)); early_top2=`embedding`,`attention`; ratio=\(7.38436579244244\) → **MISS** | measured | resolved | P4(a) family blocks; `a-adamw-0/9375.json` |
| P4-a-norm-sharpness | P4 | — | \(\lambda_{\max}\) and \(\mathrm{tr}(H^2)/d\) vs tokens | Mid (1526) → final (9375) direction | Both **nondecreasing overall**; final \(\lambda_{\max}\) in handout **`[4.5,6.0]`** | **MEASURED:** \(\lambda\) \(2.1954286466134665\to5.967310256347902\) nondec.; final in `[4.5,6.0]`; \(\mathrm{tr}(H^2)/d\) \(8.856225326736765\mathrm{e}{-7}\to5.364973956804868\mathrm{e}{-6}\) nondec. → **HIT** | measured | resolved | P4(a) progressive sharpening; `a-adamw-1526/9375.json` |
| P4-lam-1526-earlyfit | P4 | — | Early MEASURED through step 763; preferred exp→asymptote (`internal/p4-lambda-early-fit.md`); MEASURED `a-adamw-1526` | Read \(\lambda_{\max}\) at ckpt step **1526** | \(\lambda_{\max}@1526\): band **`[1.4,4.0)`** (preferred point ≈3.51 note-only) | **2.1954286466134665** — band **HIT** (preferred point 3.51 note-only miss; not scored as MISS) | measured | resolved | P4(a) early-fit; `media/p4/a-adamw-1526.summary.json` |
| P4-lam-9375-earlyfit | P4 | — | Early fit band vs final MEASURED `a-adamw-9375` | Read \(\lambda_{\max}\) at step **9375** | Band **PREDICT `[0.3,4.0)`** (preferred ≈3.51 note-only) | **5.967310256347902** — band **MISS** (above `[0.3,4.0)`; inside handout `[4.5,6.0]`) | measured | resolved | P4(a) early-fit; `media/p4/a-adamw-9375.summary.json` |
| P4-a-density-neg | P4 | Q27a | Density at 0 / 153 / 1526 / 9375 | Negatives and outliers | Negatives at init; neg mass **shrinks** init→614M; final positive outliers; neg fraction final **PREDICT `[0,0.15)`** | **MEASURED:** neg_frac \(0.004510114309467541\to0.01491374251600709\to0.9344596369432798\to0.06965100420936976\); shrink_0→9375=false (final>init); final_in_`[0,0.15)`=true → **MISS** | measured | resolved | P4(a) spectrum; dens `a-adamw-{0,153,1526,9375}.json` |
| P4-a-block-energy | P4 | Q27b | Family/layer Frobenius energy | Within-block vs random partition | Family within-frac **>** random; excess **PREDICT `[0.05,0.30)`**; layer also **>** random | **MEASURED:** family_within-random excess_final=0.08727119221832025; fam_above=True; layer_above=True → **HIT** | measured | resolved | P4(a) block structure; `a-random-energy-1526/9375.json` |
| P4-b-train-2x | P4 | — | Default AdamW final in handout band | Training knob (WD / peak LR / emb-init; freeze `TrainConfig` before launch) | \(\lambda\) factor \(\ge 2\times\) vs ~5.25 (**PREDICT** ratio `[2,\infty)` or `(0,0.5]`); val stays **`[2.925,2.931]`** | — | predicted | preregistered | P4(b) sharpening in training |
| P4-b-invar-vs-train | P4 | — | Same final loss band | Contrast hand rescale vs training \(\lambda\) move | Hand \(\lvert\Delta L\rvert\) **\<** typical successful train-arm \(\lvert\Delta L\rvert\); both can hit \(2\times\) for different reasons | — | predicted | preregistered | P4(b) gauge vs basin |
|  P4-c-loss-rank | P4 | Q10b | full / top / removed continues from SGDM@1526 | End-of-continue loss ranking | **PREDICT** full \(\approx\) top \(<\) removed; \(L_{\mathrm{removed}}-L_{\mathrm{full}}\) **PREDICT `[0.02,0.20)`** | **MEASURED:** L_full=3.582527816295624; L_top=5.502330541610718; L_removed=3.5821831226348877; gap=-0.000344693660736084 → **MISS** (from dry-run) | measured | resolved | P4(c) subspace progress  |
| P4-c-k-ablate | P4 | — | \(k=10\), refresh 25, 300 steps | Repeat top/removed at \(k\in\{3,30\}\) | \(k=3\) frac **PREDICT `[0.35,0.65)`**; \(k=30\) **PREDICT `[0.75,0.95)`**; removed–full gap grows with \(k\) | — | predicted | preregistered | P4(c) investigate \(k\) |
| P4-c-stale-evecs | P4 | — | refresh every 25 | Refresh every 100 (or freeze basis) on top/removed | Top degrades vs refresh-25; extra loss **PREDICT `[0,0.05)`** | — | predicted | preregistered | P4(c) stale evecs |

**Count (this section):** 11 IDs in table (+ overlaps `P4-invar-dL` / `P4-top10-frac` in quiz-aligned section). Stage-a scored @13/13: `P4-a-norm-sharpness` **HIT**; `P4-a-family-sharpest` / `P4-a-density-neg` / `P4-lam-9375-earlyfit` **MISS**; `P4-lam-1526-earlyfit` **HIT** (prior); `P4-a-block-energy` **HIT** (excess_final=0.08727119221832025; fam_above=True; layer_above=True).

---

## P5 exploratory pre-registrations (optional) — LOCKED PREDICT

Source: `docs/a3-p5-exploratory-design.md` §3 / §8–9. **MEASURED** 6/7 from `media/p5/` / `internal/p5-score-dryrun.md` (HIT/MISS file-exact); `P5-L-cool-dir` **INSUFFICIENT** (`dL=null` / \(L_R,L_C\) missing — no invent HIT/MISS). Baseline \(B_R=0.6237\) at \((0,500)\) is Batch 1 MEASURED (not a P5 outcome). Practice quiz **Q25** PREDICTION key **1 / 4 / 1**.

| ID | Theme | Quiz | Baseline cell | Intervention | Predicted interval / ranking | Observed | Kind | Status | Mechanism cite |
|----|-------|------|---------------|--------------|------------------------------|----------|------|--------|----------------|
|  P5-B-cool-x0 | P5 | Q25a | MEASURED \((0,500)\) \(B=0.6237\), seeds `{1,2}` | +100-step LR cooldown both branches; no state reset | \(B_C\): **PREDICT `(−∞,0.01]`** | **MEASURED:** B_C=0.7697049379348755 → **MISS** (from `internal/p5-score-dryrun.md`) | measured | resolved | P1 \(B\) + P2 river settle  |
|  P5-dB-cool | P5 | Q25a | Same; \(B_R=0.6237\) | \(\Delta B=B_R-B_C\) | **PREDICT `[0.5,0.8)`** | **MEASURED:** dB=-0.14600493793487546 (B_R=0.6237 [JSON] − B_C=0.7697049379348755) → **MISS** (from `internal/p5-score-dryrun.md`) | measured | resolved | P1×P2  |
|  P5-B-const-ctrl | P5 | Q25b | Same endpoints | +100 steps still `wsd0.0` | \(B_K\): **PREDICT `[0.5,∞)`** | **MEASURED:** B_K=0.7320129871368408 → **HIT** (from `internal/p5-score-dryrun.md`) | measured | resolved | Fit A / continued bounce  |
|  P5-rank-arms | P5 | Q25c | Arms R/C/K | Ranking | **PREDICT** \(B_C < B_R\) and \(B_C < B_K\) | **MEASURED:** B_C=0.7697049379348755 B_R=0.6237 B_K=0.7320129871368408 → **MISS** (from `internal/p5-score-dryrun.md`) | measured | resolved | P5 emergent  |
|  P5-G-cool-fitC | P5 | — | After arm C | \(G_C\) vs Fit C | If \(B_C\le 0.01\): **PREDICT** \(G_C\in[0,0.12)\); else \(G\approx B+0.09\) | **MEASURED:** G_C=0.8391239460323332 B_C=0.7697049379348755 |G-(B+0.09)|=0.020580991902542256 → **HIT** (from `internal/p5-score-dryrun.md`) | measured | resolved | Q1.1 Fit C  |
| P5-L-cool-dir | P5 | — | Endpoints after C vs R | Mean endpoint \(\Delta L\) | **PREDICT** \(L_C<L_R\); \(\Delta L\in[0.02,0.15)\) | **INSUFFICIENT:** dL / (L_R, L_C) missing — no HIT/MISS (`media/p5/` dL=null; `internal/p5-score-dryrun.md`) | predicted | partial | P2 settle |
|  P5-emergent-basin | P5 | Q25c | Forced choice | Same-valley vs true modes | **PREDICT** same-valley walls (\(B_C\le 0.01\), \(B_K>0.01\)) | **MEASURED:** B_C=0.7697049379348755 B_K=0.7320129871368408 → not-same-valley (JSON emergent='true-modes') → **MISS** (from `internal/p5-score-dryrun.md`) | measured | resolved | P1+P2  |

**Count (this section):** 7 IDs (design §9). Optional stretch `P5-lam-mid-ratio` not appended. Practice quiz: **Q25**. 6/7 MEASURED (HIT/MISS); `P5-L-cool-dir` **INSUFFICIENT** (partial) — no invented MEASURED.

---

## Fit-extrapolated PREDICTIONs (beyond MEASURED \((x,y)\) grid)

Grounded in Appendix Q1.1 Fits A–C from Batch 1 MEASURED cells only. Kind=`extrapolated`; Observed empty until a future measure. Not invented measurements. Quiz Q14–Q17.

| ID | Theme | Quiz | Baseline cell | Intervention | Predicted interval / ranking | Observed | Kind | Status | Mechanism cite |
|----|-------|------|---------------|--------------|------------------------------|----------|------|--------|----------------|
| P1-B-x0-y750 | P1 | Q14 | MEASURED \(B(0,500)=0.6237\); Fit A growth | Extend post-fork to \(y=750\) at \(x=0\) | \(B(0,750)\): **PREDICT `[0.9,1.4)`** (growth-lin \(\approx 1.03\); quad \(\approx 1.22\)); vs \(B(0,500)\): **PREDICT** larger by \(\gtrsim 0.3\) | — | extrapolated | preregistered | Q1.1 Fit A |
| P1-B-x25-y500 | P1 | Q15 | MEASURED \(B(0,500)=0.6237\), \(B(50,500)=0.1299\) | Mid-prefix fork \(x=25\) at \(y=500\) | \(B(25,500)\): **PREDICT `[0.25,0.40)`** (Fit B exp \(\approx 0.319\); lin-interp \(\approx 0.377\)); ranking **PREDICT** \(B(50,500)<B(25,500)<B(0,500)\) | — | extrapolated | preregistered | Q1.1 Fit B |
| P1-B-x2000-y500 | P1 | Q16 | MEASURED \(B(1000,500)=-0.0825\); Fit B asymptote | Longer shared prefix \(x=2000\) at \(y=500\) | \(B(2000,500)\): **PREDICT `[-0.12,-0.01)`** (exp asymptote \(\approx -0.066\)); averaging **PREDICT** Yes | — | extrapolated | preregistered | Q1.1 Fit B |
| P1-G-x0-y750 | P1 | Q17 | Fit C \(G\approx 0.10+1.01\,B\) on failures; Fit A \(B(0,750)\) | Ensemble–avg gap at extrapolated \((0,750)\) | \(G(0,750)\): **PREDICT `[1.0,1.5)`** (\(G(B_{\mathrm{lin}})\approx 1.15\); \(G(B_{\mathrm{quad}})\approx 1.35\)); vs \(G(0,500)\): **PREDICT** clearly larger | — | extrapolated | preregistered | Q1.1 Fit C∘A |

---

## Required baseline grids (tracking; not pure counterfactuals)

| ID | Quiz | Setting | Predicted interval / ranking | Observed | Kind | Notes |
|----|------|---------|------------------------------|----------|------|-------|
| P1-grid | Q20 | Full \((x,y)\) heatmap, `wsd0.0` | Fraction of 30 cells with \(B\le0.01\): **PREDICT `[0.2,0.5)`**; failures at small \(x\), large \(y\) | **MEASURED:** \(26/30=0.867\in[0.8,1]\) (miss); failures \((0,250),(0,500),(50,500),(100,500)\) | measured | From `barrier-summary.json` |
| P2-sched-rank | Q22b | `wsd0.0` / linear / cos / `wsd0.2` at tuned peak LR | Final-val ranking: **PREDICT** \(\min(L_{\mathrm{cos}},L_{\mathrm{wsd0.2}},L_{\mathrm{linear}}) < L_{\mathrm{wsd0.0}}\) | **MEASURED:** linear \(2.9282028675079346\) \(<\) cos \(2.934119462966919\) \(<\) `wsd0.2` \(2.93774151802063\) \(<\) `wsd0.0` \(3.0975050926208496\); min(decayed)\(<L_{\mathrm{wsd0.0}}\) **HIT** | measured | P2(a); 4/4 schedule arms; dip HIT / invsqrt MISS; SGDR MEASURED @ Batch2 9/9 |

---

## Quiz ↔ ledger map (complete)

| Quiz | Ledger ID | Kind (current) | Status |
|------|-----------|----------------|--------|
| Q1 | `P1-B-early-long` | measured | resolved |
| Q2 | `P1-ens-gap` | measured | resolved |
| Q3 | `P1-interp-peak` | measured | resolved |
| Q4 | `P1-d-dB-cos` | predicted | preregistered |
| Q5 | `P2-ema-gap-frac` | predicted | preregistered · **B2-locked** |
| Q6 | `P2-dip-final-R` / `B2-P2-dip-R` | measured | resolved (Q6a HIT; Q6b mid-dip unscored) |
| Q7a | `P3-eta-half` | predicted | preregistered (not B2 required 9) |
| Q7b | `P3-eta-double` | predicted | preregistered (not B2 required 9) |
| Q8 | `P3-mb-edge` / `B2-P3-adamw-pre` | measured | resolved (Q8a HIT) |
| Q9 | `P4-invar-dL` | measured | resolved · **MISS** (|ΔL|=2.73853462934494; λ ratio=0.4658784038724411; λ 5.967310256347902→2.780040977639008; probe_loss 2.9474442899227142→5.685978919267654) |
| Q10a | `P4-top10-frac` | measured | resolved · **MISS** (mean top_frac=0.09296940652964016 (source=c-full)) |
| Q10b | `P4-c-loss-rank` | measured | resolved · **MISS** (L_full=3.582527816295624; L_top=5.502330541610718; L_removed=3.5821831226348877; gap=-0.000344693660736084) |
| Q11 | `P2-horizon-dL` / `B2-P2-invsqrt-dL` / `B2-P2-sgdr-dL` | measured | Q11a both HF \(\Delta L\) MISS; Q11b mid-budget unscored |
| Q12 | `P3-clean-edge` / `B2-P3-sgd-lam` | measured | resolved (Q12a \(\lambda\) HIT; Q12b late-loss ↓ HIT) |
| Q13 | `P1-ens-vs-endpoint` | measured | resolved |
| Q14 | `P1-B-x0-y750` | extrapolated | preregistered |
| Q15 | `P1-B-x25-y500` | extrapolated | preregistered |
| Q16 | `P1-B-x2000-y500` | extrapolated | preregistered |
| Q17 | `P1-G-x0-y750` | extrapolated | preregistered |
| Q18 | `P1-B-x0-y250` | measured | resolved |
| Q19 | `P1-B-x100-y500` | measured | resolved |
| Q20 | `P1-grid` | measured | resolved |
| Q21 | `P1-G-fitC-mid` | fitted | resolved |
| Q22a | `B2-P2-wsd0.0-R` | measured | resolved (band miss; worst HIT) |
| Q22b | `P2-sched-rank` / `B2-P2-{linear,cos,wsd0.2}-R` | measured | resolved (wsd0.2 top-2 miss) |
| Q23a | `B2-P2-invsqrt-dL` | measured | resolved (band MISS) |
| Q23b | `B2-P2-sgdr-dL` | measured | resolved (\(\Delta L\) MISS; vs-invsqrt HIT) |
| Q24a | `B2-P3-sgd-lam` | measured | resolved (HIT) |
| Q24b | `B2-P3-adamw-pre` | measured | resolved (HIT) |
| Q25a | `P5-B-cool-x0` / `P5-dB-cool` | measured | resolved · `P5-B-cool-x0` **MISS**; `P5-dB-cool` **MISS** |
| Q25b | `P5-B-const-ctrl` | measured | resolved · **HIT** (B_K=0.7320129871368408) |
| Q25c | `P5-rank-arms` / `P5-emergent-basin` | measured | resolved · `P5-rank-arms` **MISS**; `P5-emergent-basin` **MISS** |
| Q26 | `P4-a-family-sharpest` | measured | resolved · **MISS** (late=`head`) |
| Q27a | `P4-a-density-neg` | measured | resolved · **MISS** (neg_frac did not shrink 0→9375) |
| Q27b | `P4-a-block-energy` | measured | resolved · **HIT** (excess_final=0.08727119221832025) |
| Q28 | `P4-b-train-2x` | predicted | preregistered · b not started |
| Q29 | `P4-c-k-ablate` | predicted | preregistered · c not started |

P5 exploratory design rows: § “P5 exploratory pre-registrations” + CSV `P5-*` (`docs/a3-p5-exploratory-design.md`). Practice quiz item: **Q25** (a–c). Status: `internal/p5-ledger-quiz-status.md`. P4 9/13 quiz/ledger: `internal/p4-quiz-ledger-9of13.md` (prior 7/13: `internal/p4-quiz-ledger-7of13.md`).

---

## Open questions / next pre-registrations

1. ~~After first measured P1 heatmap: tighten P1-ens-gap and P1-interp-peak to specific cells.~~ Done.
2. ~~Fit Batch 1 scaling curves and pre-register out-of-grid cells.~~ Done — Q14–Q17 / Fit rows.
3. ~~Harden practice quiz answer key with MEASURED / FITTED / EXTRAPOLATED tags.~~ Done — see quiz + `internal/quiz-ledger-p1-status.md`.
4. Before P1(d) LR/batch: add \(\Delta B\) intervals for peak-LR \(\times 2\) and batch \(\times 2\); resolve `P1-d-dB-cos` when cosine cells exist.
5. ~~When `internal/p2-p3-preregister-status.md` lands: merge `B2-*` PREDICTION rows.~~ Done — locked `2026-10-11T00:32:07Z`; CSV synced; Batch 2 **not** used as PREDICTION evidence.
6. ~~Four schedule arms + dip + invsqrt @ 6/9.~~ Done: `B2-P2-{wsd0.0,linear,cos,wsd0.2}-R`, `P2-sched-rank`, `B2-P2-dip-R` (**HIT**), `B2-P2-invsqrt-dL` (**MISS**). After **9/9**: fill SGDR / P3 / EMA / mid-dip Observed only.
7. Before P3(c): ≤3 hparam axes with time-at-edge or median \(\eta\lambda\) intervals.
8. P4-a @ **13/13**: `P4-a-norm-sharpness` **HIT**; `P4-a-family-sharpest` / `P4-a-density-neg` / `P4-lam-9375-earlyfit` **MISS**; `P4-lam-1526-earlyfit` **HIT** (prior); `P4-a-block-energy` **HIT** (excess_final=0.08727119221832025). See `internal/p4-9375-hit-miss-summary.md`.
9. ~~Append P5 design §9 rows.~~ Done — 6/7 MEASURED; `P5-L-cool-dir` **INSUFFICIENT** (`dL=null`).
