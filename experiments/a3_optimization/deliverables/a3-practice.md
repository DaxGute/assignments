# A3 Practice: Counterfactual Prediction Quiz + Answer Key

CS 312 · Optimization 2 · **Hand-in PDF #2** (of 2) · Practice only (not the official Oct 20 quiz)  
Modeled on A2 *Outcome Prediction Quiz: Hyperparameter Scaling* (`internal/hyperparameter_scaling_prediction_quiz.txt`).

**PDF role:** practice quiz + answer key. Results / intuition / quantitative properties → [`a3-results-intuition.md`](a3-results-intuition.md) / `media/pdfs/a3-results-intuition.pdf`. See [`a3-deliverables-readme.md`](a3-deliverables-readme.md).

**Ledger alignment:** every item cites a stable ledger ID (`docs/a3-prediction-ledger.md` / `.csv`). Kind tags there: `predicted` · `measured` · `fitted` · `extrapolated`.

**Instructions.** Predict outcomes of the following experiments. Each question has two parts:

- **From the assignment** restates the assignment setting (reference only).
- **This question changes** lists every difference; everything else is unchanged.

Check one box per row. Scoring (when used for self-study): correct option = full credit; **adjacent interval = half**. Each question has three rationale fields: assignment problem cited, what that experiment showed (with numbers — fill only after measurements), and why it implies your answer.

**Default setup.** Course d8 LM: ~35M params, ~614M tokens unless stated, AdamW \(\beta_1,\beta_2=(0.9,0.95)\), \(\epsilon=10^{-8}\), context 1024, batch 64, WD 0.1 (masked), grad clip 1.0; lower validation loss is better. Problem-specific overrides (e.g. P1 `wsd0.0`, P3 full-batch GD) are stated in each item.

**Conventions.**

- Barrier \(B(\theta_1,\theta_2)=L((\theta_1+\theta_2)/2)-(L(\theta_1)+L(\theta_2))/2\); averaging works if \(B\le 0.01\).
- Gap \(G=L(\theta_{\mathrm{avg}})-L(p_{\mathrm{ens}})\); ensemble \(p_{\mathrm{ens}}=\tfrac12 p_{\theta_1}+\tfrac12 p_{\theta_2}\).
- Regret \(R_M(t)=L(\theta^M_t)-L(\theta^{\mathrm{ref}}_t)\).
- GD edge: \(\eta\lambda_{\max}=2\). Adam \(\beta_1=0.9\) frozen-precond story: \(\eta\lambda^{\mathrm{pre}}_{\max}=38\).

**Evidence labels (answer key).**

| Label | Meaning |
|-------|---------|
| **PREDICTION** | Pre-registered before the corresponding run / fit; mechanism + handout thresholds. Not ground truth. |
| **MEASURED** | Direct experimental cell from Batch 1 (`media/p1/*`) or Batch 2 finals (`media/p2/batch2_partial_finals.csv`). |
| **FITTED** | Value from Appendix Q1.1 Fits A–C on MEASURED points (in-sample). |
| **EXTRAPOLATED** | Out-of-grid interval from those fits (ledger Kind=`extrapolated`); not a new measurement. |

Do **not** invent MEASURED numbers. Leave Resolved blank / **PENDING** until the required final metric exists.

**Batch 2 lock (P2/P3).** Q5–Q8, Q11–Q12, and Q22–Q24 PREDICTION slots were locked at `2026-10-11T00:32:07Z` **before Batch 2 completion**. Mid-run W&B was **not** used to tune bands. At **9/9**: Q6a dip **HIT**; Q11a/Q23 HF \(\Delta L\) **MISS** + vs-invsqrt **HIT**; Q22 schedule resolved; Q12a/Q24a \(\lambda_{\max}=245.16314737956807\) **HIT**; Q8a/Q24b \(\eta\lambda^{\mathrm{pre}}=30.987715914514\) **HIT**. Still blank/PENDING: Q6b mid-dip, Q5 EMA. Ledger: `docs/a3-prediction-ledger.md` / `.csv`.

**P4 Batch 3 lock.** Q9–Q10 and Q26–Q29 PREDICTION bands are locked from [`docs/a3-p4-preregister-draft.md`](a3-p4-preregister-draft.md) **before** Batch 3 measure/train. Stage a **13/13** MEASURED (\(\lambda_{\max}@9375=5.967310256347902\)). Ledger apply (resolvable IDs): `P4-a-family-sharpest` **MISS** (late=`head`); `P4-a-norm-sharpness` **HIT**; `P4-a-density-neg` **MISS** (no shrink 0→9375); `P4-lam-9375-earlyfit` **MISS** (above `[0.3,4.0)`); `P4-lam-1526-earlyfit` **HIT** (prior). `P4-a-block-energy` **HIT** (excess_final=0.08727119221832025; fam_above=True; layer_above=True) — Q27b Resolved filled. Q9: `P4-invar-dL` **MISS** (|ΔL|=2.73853462934494; ratio=0.4658784038724411). Q10a **MISS** (mean=0.09296940652964016); Q10b **MISS** (gap=-0.000344693660736084). Q28–Q29 PENDING (optional arms untouched).

**P5 exploratory lock.** MEASURED from `internal/p5-score-dryrun.md` / `media/p5/`: `P5-B-cool-x0` **MISS**, `P5-dB-cool` **MISS**, `P5-B-const-ctrl` **HIT**, `P5-rank-arms` **MISS**, `P5-G-cool-fitC` **HIT**, `P5-emergent-basin` **MISS**; `P5-L-cool-dir` **INSUFFICIENT** (`dL=null` / \(L_R,L_C\) missing — no HIT/MISS). PREDICTION key retained **(a)1 · (b)4 · (c)1**. Keep locked bands; never erase a miss.

---

## Prediction

### Q1. Mode connectivity: early vs late fork  
**Difficulty: Easy · theme P1(b) · ledger `P1-B-early-long`**

**From the assignment (reference only).** P1 forked-run grid with `wsd0.0` (constant after warmup), shared prefix \(x\), branch length \(y\), different `batch_order_seed`, barrier criterion \(B\le 0.01\).

**This question changes.** Compare only two cells: \(A=(x,y)=(0,500)\) and \(C=(1000,25)\). Same peak LR, same seeds `{1,2}`.

**(a) Barrier at cell \(A=(0,500)\)** (10 points). Which interval contains \(B(0,500)\)?

· 1. \((-\infty, 0.01]\) · 2. \((0.01, 0.05)\) · 3. \([0.05, 0.2)\) · 4. \([0.2, \infty)\)

**(b) Barrier at cell \(C=(1000,25)\)** (10 points). Which interval contains \(B(1000,25)\)?

· 1. \((-\infty, 0.01]\) · 2. \((0.01, 0.05)\) · 3. \([0.05, 0.2)\) · 4. \([0.2, \infty)\)

Mark answers and rationale on the answer sheet.

---

### Q2. Ensemble vs weight average when \(B\) is large  
**Difficulty: Medium · theme P1(a–b) · ledger `P1-ens-gap`**

**From the assignment.** For each grid cell you evaluate endpoints, \(0.5/0.5\) weight average, and \(0.5/0.5\) prediction ensemble \(p_{\mathrm{ens}}=\tfrac12 p_{\theta_1}+\tfrac12 p_{\theta_2}\).

**This question changes.** Restrict to cells with \(B>0.01\). Let \(G=L(\theta_{\mathrm{avg}})-L(p_{\mathrm{ens}})\).

**(a) Sign / ranking** (8 points). Which do you predict?

· 1. \(G<0\) (weight avg better) · 2. \(G\in[0,0.01)\) · 3. \(G\ge 0.01\) and ensemble better · 4. Ensemble and weight avg within \(0.005\) always

**(b) Magnitude** (12 points). Which interval contains the median \(G\) over those cells?

· 1. \([0, 0.02)\) · 2. \([0.02, 0.08)\) · 3. \([0.08, 0.2)\) · 4. \([0.2, \infty)\)

---

### Q3. Interpolation peak vs midpoint barrier  
**Difficulty: Medium · theme P1(c) · ledger `P1-interp-peak`**

**From the assignment.** Full path \(\theta(\alpha)=(1-\alpha)\theta_1+\alpha\theta_2\), \(\alpha\in\{0,0.25,0.5,0.75,1\}\).

**This question changes.** Take one representative cell with midpoint \(B\in[0.05,0.2)\). Let \(L_\alpha=L(\theta(\alpha))\).

**(a) Ranking** (10 points). Which ranking is closest?

· 1. \(L_{0.5}\) strictly largest among the five \(\alpha\)  
· 2. \(L_{0.25}\) or \(L_{0.75}\) largest (asymmetric barrier)  
· 3. Path nearly flat: \(\max L_\alpha-\min L_\alpha<0.01\)  
· 4. Endpoints worst; midpoint best (inverted)

**(b) Barrier height check** (10 points). Which interval contains \(L_{0.5}-\frac{L_0+L_1}{2}\)?

· 1. \([0, 0.01)\) · 2. \([0.01, 0.05)\) · 3. \([0.05, 0.2)\) · 4. \([0.2, \infty)\)

---

### Q4. Schedule decay lowers barrier at fixed fork  
**Difficulty: Hard · theme P1(d)+P2 · ledger `P1-d-dB-cos`**  
**(Evidence: PREDICTION only — no cosine Batch 2 cells yet.)**

**From the assignment.** P1(d) asks which training choices control same-basin membership via \(B\) and interpolation curves.

**This question changes.** Fix \((x,y)=(250,250)\). Arm W uses `wsd0.0` after fork; arm C uses cosine decay after fork (same peak LR, same seeds). Define \(\Delta B=B_W-B_C\).

**(a)** (10 points). Which interval contains \(\Delta B\)?

· 1. \((-\infty, 0)\) · 2. \([0, 0.02)\) · 3. \([0.02, 0.1)\) · 4. \([0.1, \infty)\)

**(b)** (10 points). After the change, does arm C satisfy \(B_C\le 0.01\)?

· 1. Yes · 2. No · 3. Depends on \(B_W\) band · 4. Always yes regardless of \(B_W\)

*(Study note: MEASURED \(B_W(250,250)=-0.0779\le 0.01\) already under `wsd0.0`; cosine \(\Delta B\) is still unmeasured.)*

---

### Q5. EMA closes schedule gaps  
**Difficulty: Medium · theme P2(b) · ledger `P2-ema-gap-frac`**  
**(Evidence: PREDICTION — Batch 2 locked `2026-10-11T00:32:07Z`; jobs 1–4 schedule+EMA. No finals used.)**

**From the assignment.** Compare `wsd0.0`, `linear`, `cos`, `wsd0.2` at equal tokens; EMA decays \(\{0.99,0.999\}\) eval-only; regret vs best raw final.

**This question changes.** Let \(G_{\mathrm{raw}}=L^{\mathrm{raw}}_{\mathrm{worst}}-L^{\mathrm{raw}}_{\mathrm{best}}\) among the four schedules (best peak LR each). Let \(G_{\mathrm{EMA}}\) be the same gap using each run’s best EMA val. Define closed fraction \(f=1-G_{\mathrm{EMA}}/G_{\mathrm{raw}}\) (if \(G_{\mathrm{raw}}=0\), treat \(f=1\)).

**(a)** (10 points). Which interval contains \(f\)?

· 1. \([0, 0.25)\) · 2. \([0.25, 0.5)\) · 3. \([0.5, 0.75)\) · 4. \([0.75, 1]\)

**(b)** (10 points). After best EMA, which schedule still has the lowest val loss?

· 1. `wsd0.0` · 2. `linear` · 3. `cos` or `wsd0.2` · 4. All four within \(0.01\)

---

### Q6. LR dip then rewarm — final regret  
**Difficulty: Easy · theme P2(c) · ledger `B2-P2-dip-R` / `P2-dip-final-R`**  
**(Evidence: MEASURED final \(R\) — job4 `rms7iyyt` vs job3 `wsd0.2` control; mid-dip unscored.)**

**From the assignment.** Temporary `lr_dip` then rewarm without resetting state; control shares final cooldown; plot regret vs control.

**This question changes.** One dip depth/duration (moderate: scale \(0.1\), duration \(10\%\) of training, mid-run). Let final \(R=L_{\mathrm{dip}}-L_{\mathrm{ctrl}}\).

**(a) Final regret** (12 points). Which interval contains \(R\)?

· 1. \((-\infty, -0.02)\) · 2. \([-0.02, 0.02)\) · 3. \([0.02, 0.08)\) · 4. \([0.08, \infty)\)

**(b) During-dip regret** (8 points). Mid-dip \(R_{\mathrm{mid}}\) (at bottom of dip) is in:

· 1. \((-\infty, -0.02)\) · 2. \([-0.02, 0.02)\) · 3. \([0.02, 0.08)\) · 4. \([0.08, \infty)\)

---

### Q7. Edge of stability: mid-run \(\eta\) changes  
**Difficulty: Easy · theme P3(a)ii · ledger `P3-eta-half` / `P3-eta-double`**  
**(Evidence: PREDICTION / PENDING — η-change arms not in Batch 2 required 9; clean-edge P3 is MEASURED elsewhere as Q12/Q24a.)**

**From the assignment.** Full-batch GD, \(N=32\), \(\eta=0.01\), 1500 updates; progressive sharpening toward \(2/\eta=200\). At update 750: either `step750x0.5` or double \(\eta\).

**This question changes.** Report median \(\lambda_{\max}\) over updates \(1200\)–\(1500\).

**(a) After \(\eta\leftarrow 0.005\)** (10 points). Interval for median \(\lambda_{\max}\):

· 1. \([150, 250)\) · 2. \([250, 350)\) · 3. \([350, 450)\) · 4. \([450, \infty)\)

**(b) After \(\eta\leftarrow 0.02\)** (10 points). Interval for median \(\lambda_{\max}\):

· 1. \((0, 80)\) · 2. \([80, 120)\) · 3. \([120, 180)\) · 4. \([180, \infty)\)

---

### Q8. Minibatch AdamW preconditioned edge  
**Difficulty: Medium · theme P3(b) · ledger `B2-P3-adamw-pre` / `P3-mb-edge`**  
**(Evidence: MEASURED — job `a3-b2-p3-1-adamw` / `foew12y0`; post-warmup median \(\eta\lambda^{\mathrm{pre}}=30.987715914514\) from `media/p2/p3_adamw_sharpness_summary.json`.)**

**From the assignment.** `minibatch()` recipe; plot \(\eta_t\lambda^{\mathrm{pre}}_{\max}\) after warmup; Adam \(\beta_1=0.9\) frozen-precond threshold \(38\).

**This question changes.** Summarize the post-warmup median of \(\eta_t\lambda^{\mathrm{pre}}_{\max}\) on the held-out probe.

**(a)** (12 points). Which interval contains that median?

· 1. \([0, 20)\) · 2. \([20, 30)\) · 3. \([30, 50)\) · 4. \([50, \infty)\)

**(b)** (8 points). Relative to full-batch GD’s \(\eta\lambda\approx 2\) ride, minibatch AdamW is:

· 1. Exactly as tight around its threshold · 2. Noisier / wider band around ~38 · 3. Stuck near 2 (not 38) · 4. Diverges (\(\eta\lambda^{\mathrm{pre}}\to\infty\))

---

### Q9. Loss-invariant rescale moves sharpness  
**Difficulty: Hard · theme P4(b) · ledger `P4-invar-dL`**  
**(Evidence: MEASURED — `P4-invar-dL` **MISS**; |ΔL|=2.73853462934494; λ ratio=0.4658784038724411; λ 5.967310256347902→2.780040977639008; probe_loss 2.9474442899227142→5.685978919267654)

**From the assignment.** Default d8 AdamW finals: \(\lambda_{\max}\in[4.5,6.0]\), val \(\in[2.925,2.931]\) (handout bands). Find transforms / recipes that move \(\lambda_{\max}\) by \(\ge 2\times\) without leaving the loss band.

**This question changes.** Apply a hand loss-invariant rescale to one weight family on the final checkpoint (no further training; stage `b-rescale`).

**(a)** (10 points). \(\lvert\Delta L_{\mathrm{val}}\rvert\) is in:

· 1. \([0, 0.002)\) · 2. \([0.002, 0.01)\) · 3. \([0.01, 0.05)\) · 4. \([0.05, \infty)\)

**(b)** (10 points). Ratio \(\lambda_{\max}^{\mathrm{after}}/\lambda_{\max}^{\mathrm{before}}\) is in:

· 1. \((0, 0.5)\) · 2. \([0.5, 0.9)\) · 3. \([0.9, 1.1)\) · 4. \([2, \infty)\)  
*(If your rescale direction lowers sharpness, treat option 1 as the success band for \(\ge 2\times\) decrease; option 4 for \(\ge 2\times\) increase.)*

---

### Q10. Top-10 subspace continue (SGDM)  
**Difficulty: Medium · theme P4(c) · ledger `P4-top10-frac` / `P4-c-loss-rank`**  
**(Evidence: MEASURED — `P4-top10-frac` / `P4-c-loss-rank` from `internal/p4-c-score-dryrun.md`; mean=0.09296940652964016 **MISS**; gap=-0.000344693660736084 **MISS**)

**From the assignment.** From SGDM checkpoint at 100M tokens (step 1526): zero momentum; continue 300 steps with full / top-10 / removed updates; refresh eigenvectors every 25 steps (`k=10`).

**This question changes.** Report mean gradient fraction in the top-10 span over the continue, and final loss ranking.

**(a)** (10 points). Mean \(\|P_{10}g\|/\|g\|\) in:

· 1. \([0, 0.3)\) · 2. \([0.3, 0.6)\) · 3. \([0.6, 0.85)\) · 4. \([0.85, 1]\)

**(b)** (10 points). Final val-loss ranking (lowest → highest):

· 1. full \(\approx\) top-10 \(<\) removed  
· 2. full \(<\) removed \(<\) top-10  
· 3. removed \(<\) full \(\approx\) top-10  
· 4. all three within \(0.01\)

---

### Q11. Horizon-free vs budget-tuned cooldown  
**Difficulty: Hard · theme P2(d) · ledger `P2-horizon-dL` / `B2-P2-invsqrt-dL` / `B2-P2-sgdr-dL`**  
**(Evidence: MEASURED invsqrt + SGDR \(\Delta L\) vs linear @ Batch 2 9/9 — both band MISS; mid-budget stop still PREDICTION / PENDING.)**

**From the assignment.** Invsqrt \(\alpha_t=\alpha_{\max}\min\{t/w,\sqrt{w/t}\}\) and SGDR cosine warm restarts; \(w,T,\alpha\) bounds chosen independently of final budget; no budget-specific cooldown; AdamW unchanged; compare val and regret to schedules tuned to each budget.

**This question changes.** Fix one invsqrt \((w,\alpha_{\max})\) and one SGDR \((T,\alpha_{\max},\alpha_{\min})\) pair. Evaluate at two stopping budgets: mid \(T_{\mathrm{mid}}\) and long \(T_{\mathrm{long}}\) where the budget-tuned baseline gets a long final cooldown. Let \(\Delta L = L_{\mathrm{horizon\text{-}free}}-L_{\mathrm{tuned}}\) at each stop (best of invsqrt/SGDR vs best tuned cosine/`wsd0.2`).

**(a) At \(T_{\mathrm{long}}\)** (10 points). Which interval contains \(\Delta L\)?

· 1. \((-\infty, -0.01)\) · 2. \([-0.01, 0.01)\) · 3. \([0.01, 0.08)\) · 4. \([0.08, \infty)\)

**(b) At \(T_{\mathrm{mid}}\)** (10 points). Ranking of usefulness:

· 1. Horizon-free clearly better (\(\Delta L<-0.01\))  
· 2. Competitive: \(\lvert\Delta L\rvert<0.02\)  
· 3. Horizon-free clearly worse (\(\Delta L\ge 0.08\))  
· 4. Diverges / NaN

---

### Q12. Full-batch clean edge (\(N=32\), \(\eta=0.01\))  
**Difficulty: Easy · theme P3(a)i · ledger `B2-P3-sgd-lam` / `P3-clean-edge`**  
**(Evidence: MEASURED — job `a3-b2-p3-0-sgd` / `0dewkykj`; median \(\lambda_{\max}=245.16314737956807\) from `media/p2/p3_sgd_sharpness_summary.json`.)**

**From the assignment.** `full_batch(32, 1500, learning_rate=0.01)`; GD; 16× embedding init; plot \(\lambda_{\max}\) vs update and vs train loss; dashed threshold at \(2/\eta\).

**This question changes.** Report the median \(\lambda_{\max}\) over updates \(1200\)–\(1500\) on the clean case (no mid-run \(\eta\) change).

**(a)** (12 points). Which interval contains that median?

· 1. \([50, 150)\) · 2. \([150, 250)\) · 3. \([250, 350)\) · 4. \([350, 450)\)

**(b)** (8 points). While riding near the edge, train loss over those late updates:

· 1. Strictly increases  
· 2. Continues to decrease overall (Cohen-style)  
· 3. Exactly flat machine-epsilon  
· 4. Diverges to \(\infty\)

---

### Q13. Ensemble vs endpoints on an early-fork cell  
**Difficulty: Medium · theme P1(a) · ledger `P1-ens-vs-endpoint`**

**From the assignment.** Forked-run grid with `wsd0.0`; for each cell evaluate endpoints and \(0.5/0.5\) prediction ensemble \(p_{\mathrm{ens}}\).

**This question changes.** Restrict to cell \((x,y)=(0,500)\), seeds `{1,2}`. Let \(L_{\star}=\min(L(\theta_1),L(\theta_2))\) and \(G_{\mathrm{ens}}=L_{\star}-L(p_{\mathrm{ens}})\).

**(a) Sign** (8 points). Which do you predict?

· 1. \(G_{\mathrm{ens}}<0\) (ensemble worse than best endpoint)  
· 2. \(G_{\mathrm{ens}}\in[0,0.01)\)  
· 3. \(G_{\mathrm{ens}}\ge 0.01\) (ensemble clearly better than best endpoint)  
· 4. Ensemble undefined (probabilities not comparable)

**(b) Vs weight average** (12 points). On the same cell, ranking of \(L(p_{\mathrm{ens}})\) vs \(L(\theta_{\mathrm{avg}})\):

· 1. \(L(p_{\mathrm{ens}}) < L(\theta_{\mathrm{avg}})-0.01\)  
· 2. Within \(0.01\)  
· 3. \(L(p_{\mathrm{ens}}) > L(\theta_{\mathrm{avg}})+0.01\)  
· 4. \(\theta_{\mathrm{avg}}\) diverges

---

### Q14. Extrapolate barrier growth along \(x=0\)  
**Difficulty: Medium · theme P1(b) · ledger `P1-B-x0-y750`**  
**(Evidence: EXTRAPOLATED from Fit A — not MEASURED.)**

**From the assignment.** P1 grid includes \(y\in\{25,50,100,250,500\}\) at each fork prefix \(x\).

**This question changes.** Hold \(x=0\). Extend post-fork horizon to \(y=750\) (beyond the measured grid). Use the Batch 1 growth pattern of \(B\) vs \(y\) at \(x=0\) (Appendix Q1.1 Fit A).

**(a)** (12 points). Which interval contains \(B(0,750)\)?

· 1. \([0.5, 0.9)\) · 2. \([0.9, 1.4)\) · 3. \([1.4, 2.0)\) · 4. \([2.0, \infty)\)

**(b)** (8 points). Relative to MEASURED \(B(0,500)\), you expect \(B(0,750)\):

· 1. Smaller (barrier shrinks with more branch steps)  
· 2. About equal (\(\lvert\Delta B\rvert<0.05\))  
· 3. Larger by \(\gtrsim 0.3\)  
· 4. Negative (averaging becomes safe)

---

### Q15. Mid-prefix fork at fixed long branch  
**Difficulty: Medium · theme P1(b) · ledger `P1-B-x25-y500`**  
**(Evidence: EXTRAPOLATED from Fit B + MEASURED endpoints — not MEASURED at \(x=25\).)**

**From the assignment.** Grid samples \(x\in\{0,50,100,250,500,1000\}\) at \(y=500\).

**This question changes.** Insert an unmeasured mid-prefix cell \((x,y)=(25,500)\). Anchor on MEASURED \(B(0,500)\) and \(B(50,500)\) and the exponential \(B\)–vs–\(x\) fit at \(y=500\) (Fit B).

**(a)** (12 points). Which interval contains \(B(25,500)\)?

· 1. \([0.10, 0.25)\) · 2. \([0.25, 0.40)\) · 3. \([0.40, 0.55)\) · 4. \([0.55, \infty)\)

**(b)** (8 points). Ranking vs the two MEASURED anchors:

· 1. \(B(25,500) < B(50,500) < B(0,500)\)  
· 2. \(B(50,500) < B(25,500) < B(0,500)\)  
· 3. \(B(25,500) > B(0,500)\)  
· 4. \(B(25,500)\le 0.01\)

---

### Q16. Longer shared prefix at \(y=500\)  
**Difficulty: Easy · theme P1(b) · ledger `P1-B-x2000-y500`**  
**(Evidence: EXTRAPOLATED from Fit B asymptote — not MEASURED.)**

**From the assignment.** Longest measured prefix is \(x=1000\); at \(y=500\), MEASURED \(B(1000,500)<0\).

**This question changes.** Extend shared prefix to \(x=2000\) at \(y=500\) (same `wsd0.0`, seeds).

**(a)** (12 points). Which interval contains \(B(2000,500)\)?

· 1. \((-\infty, -0.12)\) · 2. \([-0.12, -0.01)\) · 3. \([-0.01, 0.01]\) · 4. \((0.01, \infty)\)

**(b)** (8 points). Does averaging work (\(B\le 0.01\))?

· 1. Yes · 2. No · 3. Only if LR retuned · 4. Undefined

---

### Q17. Ensemble–avg gap at extrapolated early fork  
**Difficulty: Hard · theme P1(a–b) · ledger `P1-G-x0-y750`**  
**(Evidence: EXTRAPOLATED from Fit C ∘ Fit A — not MEASURED.)**

**From the assignment.** On failure cells, \(G=L(\theta_{\mathrm{avg}})-L(p_{\mathrm{ens}})\) grows with \(B\).

**This question changes.** At the extrapolated cell \((0,750)\), predict \(G\) using the Batch 1 \(G\)–vs–\(B\) fit composed with Fit A’s \(B(0,750)\).

**(a)** (12 points). Which interval contains \(G(0,750)\)?

· 1. \([0.5, 1.0)\) · 2. \([1.0, 1.5)\) · 3. \([1.5, 2.0)\) · 4. \([2.0, \infty)\)

**(b)** (8 points). Relative to MEASURED \(G(0,500)\approx 0.73\):

· 1. \(G(0,750)\) smaller  
· 2. About equal  
· 3. Clearly larger (\(\Delta G\gtrsim 0.3\))  
· 4. Gap sign flips (\(G<0\))

---

### Q18. Medium branch, zero prefix  
**Difficulty: Easy · theme P1(b) · ledger `P1-B-x0-y250`**

**From the assignment.** Same forked-run grid / barrier definition.

**This question changes.** Restrict to cell \((x,y)=(0,250)\) (shorter branch than Q1’s \((0,500)\)).

**(a)** (10 points). Which interval contains \(B(0,250)\)?

· 1. \((-\infty, 0.01]\) · 2. \((0.01, 0.05)\) · 3. \([0.05, 0.2)\) · 4. \([0.2, \infty)\)

**(b)** (10 points). Ranking vs MEASURED pattern at \(x=0\):

· 1. \(B(0,250) < B(0,100) < B(0,500)\)  
· 2. \(B(0,100) < B(0,250) < B(0,500)\)  
· 3. \(B(0,500) < B(0,250)\)  
· 4. All three \(\le 0.01\)

---

### Q19. Borderline long-branch cell  
**Difficulty: Medium · theme P1(b) · ledger `P1-B-x100-y500`**

**From the assignment.** Failure cells are those with \(B>0.01\).

**This question changes.** Restrict to \((x,y)=(100,500)\).

**(a)** (10 points). Which interval contains \(B(100,500)\)?

· 1. \((-\infty, 0.01]\) · 2. \((0.01, 0.05)\) · 3. \([0.05, 0.2)\) · 4. \([0.2, \infty)\)

**(b)** (10 points). On this cell, median-style \(G=L(\theta_{\mathrm{avg}})-L(p_{\mathrm{ens}})\) is in:

· 1. \([0, 0.05)\) · 2. \([0.05, 0.15)\) · 3. \([0.15, 0.30)\) · 4. \([0.30, \infty)\)

---

### Q20. Grid-wide connectivity fraction  
**Difficulty: Easy · theme P1(a–b) · ledger `P1-grid`**

**From the assignment.** Full \(6\times 5\) forked-run heatmap under `wsd0.0`.

**This question changes.** Report the fraction of the 30 measured cells with \(B\le 0.01\).

**(a)** (12 points). Which interval contains that fraction?

· 1. \([0, 0.2)\) · 2. \([0.2, 0.5)\) · 3. \([0.5, 0.8)\) · 4. \([0.8, 1]\)

**(b)** (8 points). Where do the \(B>0.01\) failures concentrate?

· 1. Large \(x\), small \(y\)  
· 2. Small \(x\), large \(y\)  
· 3. Uniform across the grid  
· 4. Only at \(x=1000\)

---

### Q21. Fit-\(C\) gap from a mid-band barrier  
**Difficulty: Medium · theme P1(a–c) · ledger `P1-G-fitC-mid`**  
**(Evidence: FITTED in-sample + MEASURED check on cell \((50,500)\).)**

**From the assignment.** Ensemble–avg gap \(G\) tracks barrier height when modes separate.

**This question changes.** Take MEASURED cell \((50,500)\) with midpoint \(B\in[0.05,0.2)\). Using the linear \(G\)–vs–\(B\) relationship fitted on Batch 1 (Appendix Q1.1 Fit C), predict \(G\).

**(a)** (12 points). Which interval contains \(G(50,500)\)?

· 1. \([0.10, 0.15)\) · 2. \([0.15, 0.20)\) · 3. \([0.20, 0.25)\) · 4. \([0.25, 0.35)\)

**(b)** (8 points). Closest mechanistic reading of Fit C:

· 1. \(G\approx B\) (gap equals barrier)  
· 2. \(G\approx B+0.09\) (gap ≈ barrier plus nearly constant ensemble win vs midpoint endpoints)  
· 3. \(G\) independent of \(B\)  
· 4. \(G\approx -B\)

---

### Q22. Batch 2 schedule regret bands (jobs 1–4)  
**Difficulty: Medium · theme P2(a) · ledger `B2-P2-wsd0.0-R` … `B2-P2-wsd0.2-R` / `P2-sched-rank`**  
**(Evidence: MEASURED — Batch 2 locked `2026-10-11T00:32:07Z`; finals from `batch2_partial_finals.csv` / ledger Observed.)**

**From the assignment.** P2(a) four schedules at equal tokens; regret vs best raw final.

**This question changes.** Restrict to Batch 2 jobs `a3-b2-p2-wsd0.0`, `…-linear`, `…-cos`, `…-wsd0.2` (no dip). Let \(R_M=L_M-L_{\mathrm{ref}}\) with \(L_{\mathrm{ref}}=\min\) of the four finals.

**(a) Constant schedule regret** (10 points). Which interval contains \(R_{\mathrm{wsd0.0}}\)?

· 1. \([0, 0.02)\) · 2. \([0.02, 0.08)\) · 3. \([0.08, 0.15)\) · 4. \([0.15, \infty)\)

**(b) Ranking** (10 points). Closest final-val ranking:

· 1. \(L_{\mathrm{wsd0.0}} < \min(L_{\mathrm{cos}},L_{\mathrm{wsd0.2}},L_{\mathrm{linear}})\)  
· 2. \(\min(L_{\mathrm{cos}},L_{\mathrm{wsd0.2}},L_{\mathrm{linear}}) < L_{\mathrm{wsd0.0}}\) and \(\max(R_{\mathrm{cos}},R_{\mathrm{wsd0.2}},R_{\mathrm{linear}})<0.02\)  
· 3. All four within \(0.005\)  
· 4. Linear uniquely best by \(\ge 0.05\)

---

### Q23. Batch 2 horizon-free pair (jobs 6–7)  
**Difficulty: Hard · theme P2(d) · ledger `B2-P2-invsqrt-dL` / `B2-P2-sgdr-dL`**  
**(Evidence: MEASURED invsqrt + SGDR \(\Delta L\) vs linear — both band MISS; \|SGDR−invsqrt\| HIT.)**

**From the assignment.** Invsqrt and SGDR with fixed \((w,T)\); no budget-specific cooldown.

**This question changes.** Batch 2 jobs `a3-b2-p2-invsqrt` and `a3-b2-p2-sgdr2000` at the full 614M-token stop. Compare each to the best of `{cos, wsd0.2, linear}` from Q22. Let \(\Delta L_M=L_M-L_{\mathrm{tuned}}\).

**(a) Invsqrt gap** (10 points). Interval for \(\Delta L_{\mathrm{invsqrt}}\):

· 1. \((-\infty, -0.01)\) · 2. \([-0.01, 0.01)\) · 3. \([0.01, 0.08)\) · 4. \([0.08, \infty)\)

**(b) SGDR vs invsqrt** (10 points). \(\lvert L_{\mathrm{sgdr}}-L_{\mathrm{invsqrt}}\rvert\) in:

· 1. \([0, 0.03)\) · 2. \([0.03, 0.08)\) · 3. \([0.08, 0.15)\) · 4. \([0.15, \infty)\)

---

### Q24. Batch 2 sharpness starters (jobs 8–9)  
**Difficulty: Easy · theme P3(a)i + P3(b) · ledger `B2-P3-sgd-lam` / `B2-P3-adamw-pre`**  
**(Evidence: MEASURED — (a) median \(\lambda_{\max}=245.16314737956807\); (b) post-warmup median \(\eta\lambda^{\mathrm{pre}}=30.987715914514\).)**

**From the assignment.** Clean full-batch GD \(N=32\), \(\eta=0.01\), 1500 updates; minibatch AdamW with preconditioned sharpness.

**This question changes.** Only the two Batch 2 starter jobs: `a3-b2-p3-0-sgd` and `a3-b2-p3-1-adamw` (no mid-run \(\eta\) change).

**(a) Full-batch median \(\lambda_{\max}\) (1200–1500)** (10 points):

· 1. \([50, 150)\) · 2. \([150, 250)\) · 3. \([250, 350)\) · 4. \([350, 450)\)

**(b) Minibatch post-warmup median \(\eta\lambda^{\mathrm{pre}}\)** (10 points):

· 1. \([0, 20)\) · 2. \([20, 30)\) · 3. \([30, 50)\) · 4. \([50, \infty)\)

---

### Q25. P5 exploratory: post-separation cooldown vs constant continue  
**Difficulty: Hard · theme P1×P2 · ledger `P5-B-cool-x0` / `P5-dB-cool` / `P5-B-const-ctrl` / `P5-rank-arms` / `P5-emergent-basin`**  
**(Evidence: MEASURED — `P5-B-cool-x0` **MISS**; `P5-dB-cool` **MISS**; `P5-B-const-ctrl` **HIT**; `P5-rank-arms` **MISS**; `P5-G-cool-fitC` **HIT**; `P5-emergent-basin` **MISS** (from `internal/p5-score-dryrun.md` / `media/p5/`))

**From the assignment (reference only).** P1 forked-run grid with `wsd0.0`. MEASURED cell \((x,y)=(0,500)\) has midpoint barrier \(B=0.6237>0.01\) (weight averaging fails). P2 river-valley picture: large LR bounces on valley walls; decaying LR settles toward the river.

**This question changes.** Take the two \((0,500)\) branch endpoints (seeds `{1,2}`). Continue **both** for \(y_{\mathrm{cool}}=100\) more updates **without** resetting model or optimizer state, under one of two arms, then re-measure \(B\):

- Arm C: learning rate decays to ~0 over those 100 updates (cosine or linear cooldown).
- Arm K: learning rate stays at the constant post-warmup value (`wsd0.0`).

Everything else (data distribution, seeds, eval protocol) matches P1. Distinct from Q4 / `P1-d-dB-cos` (cosine *during* the fork branch). Let \(B_R=0.6237\) (MEASURED raw midpoint).

**(a) Barrier after cooldown** (12 points). Which interval contains \(B_C\)? (Implies \(\Delta B=B_R-B_C\) in ledger `P5-dB-cool` **PREDICT `[0.5,0.8)`** if option 1.)

· 1. \((-\infty, 0.01]\) · 2. \((0.01, 0.05)\) · 3. \([0.05, 0.2)\) · 4. \([0.2, \infty)\)

**(b) Barrier after constant continue** (8 points). Which interval contains \(B_K\)?

· 1. \((-\infty, 0.01]\) · 2. \((0.01, 0.05)\) · 3. \([0.05, 0.2)\) · 4. \([0.5, \infty)\)

**(c) Emergent interpretation** (10 points). Which statement do you predict? (Also encodes ranking `P5-rank-arms`: \(B_C < B_R\) and \(B_C < B_K\).)

· 1. Same-valley walls: \(B_C\le 0.01\) and \(B_K>0.01\)  
· 2. True disconnected modes: \(B_C>0.05\) and \(B_K>0.05\)  
· 3. Cooldown irrelevant: \(\lvert B_C-B_R\rvert<0.05\)  
· 4. Constant continue connects: \(B_K\le 0.01\) and \(B_C>0.01\)

---

### Q26. Late-training sharpest Hessian family  
**Difficulty: Easy · theme P4(a) · ledger `P4-a-family-sharpest`**  
**(Evidence: MEASURED @13/13 — late_sharpest=`head` (\(5.641603135007765\)); ledger **MISS** vs PREDICT attention|mlp.)**

**From the assignment.** Measure provided AdamW checkpoints: per-family block \(\lambda_{\max}\) for `embedding`, `attention`, `qk_norm`, `mlp`, `norm`, `head`.

**This question changes.** Report which family is sharpest near the final step (9375 / `FINAL_STEP`), and how init differs.

**(a) Late sharpest family** (12 points). Closest prediction:

· 1. `embedding` uniquely sharpest  
· 2. `head` uniquely sharpest  
· 3. `attention` or `mlp` (tied-ok)  
· 4. `qk_norm` or `norm`

**(b) Init vs late** (8 points). At step 0 vs 614M:

· 1. Same family always sharpest  
· 2. `embedding`/`head` more often in top-2 at init than at late  
· 3. Only `mlp` can be sharp early  
· 4. No family structure at init

---

### Q27. Spectrum / block energy over training  
**Difficulty: Medium · theme P4(a) · ledger `P4-a-density-neg` / `P4-a-block-energy`**  
**(Evidence: dens-neg MEASURED **MISS** — neg_frac \(0.00451\to\ldots\to0.06965\) no shrink 0→9375; `P4-a-block-energy` MEASURED **HIT** — excess_final=0.08727119221832025; fam_above=True; layer_above=True.)**

**From the assignment.** Density at init / 10M / 100M / 614M; split \(\mathrm{tr}(H^2)\) within vs between family blocks vs a random partition of equal sizes.

**This question changes.** Directional claims only (no invented magnitudes as MEASURED).

**(a) Negatives** (10 points). Closest spectrum story:

· 1. No negatives ever  
· 2. Negatives at init; negative mass shrinks by 614M; final has clear positive outliers  
· 3. Negatives grow to dominate at final  
· 4. Density undefined on transformers

**(b) Block energy** (10 points). Within-family Frobenius fraction vs matched random partition at final:

· 1. Family \(\le\) random  
· 2. Family \(>\) random; excess in \([0.05, 0.30)\)  
· 3. Family \(>\) random; excess \(\ge 0.50\)  
· 4. Only layer partition matters; family = random

---

### Q28. Training recipe moves \(\lambda_{\max}\) inside the loss band  
**Difficulty: Hard · theme P4(b) · ledger `P4-b-train-2x`**  
**(Evidence: PREDICTION only — P4 draft; freeze `TrainConfig` before Batch 3 `b-train`.)**

**From the assignment.** Move final \(\lambda_{\max}\) by \(\ge 2\times\) while keeping val in \([2.925,2.931]\).

**This question changes.** One training-knob arm (WD / peak LR / embedding-init — not hand rescale). Baseline mid-band \(\lambda\sim 5.25\).

**(a)** (12 points). Joint outcome you predict:

· 1. \(\lambda\) factor \(\ge 2\times\) **and** val in \([2.925,2.931]\)  
· 2. \(\lambda\) factor \(\ge 2\times\) but val exits the band (still “success”)  
· 3. Val stays in band but \(\lambda\) moves \(<2\times\)  
· 4. Both criteria fail

**(b)** (8 points). If val exits the band:

· 1. Widen the band and count success  
· 2. Count as **failed arm** (do not relabel)  
· 3. Drop \(\lambda\) criterion  
· 4. Use train loss instead of val

---

### Q29. Subspace \(k\) ablation  
**Difficulty: Medium · theme P4(c) · ledger `P4-c-k-ablate`**  
**(Evidence: PREDICTION only — optional Batch 3 follow-on; not measured.)**

**From the assignment.** Default `k=10`, refresh 25, 300 steps from SGDM @ 1526.

**This question changes.** Repeat `top` / `removed` at \(k=3\) and \(k=30\) (same start/refresh/steps).

**(a) Mean grad fraction at \(k=3\)** (10 points):

· 1. \([0.15, 0.35)\) · 2. \([0.35, 0.65)\) · 3. \([0.65, 0.85)\) · 4. \([0.85, 1]\)

**(b) At \(k=30\) vs \(k=10\)** (10 points):

· 1. Fraction lower; removed–full gap larger  
· 2. Fraction higher (\([0.75,0.95)\)); top even closer to full  
· 3. Identical to \(k=10\)  
· 4. Removed becomes best arm

---

## Answer sheet (practice)

Name ________________  SUNet ID ________________

| Q | Part | Your option | Assignment problem cited | Observations (MEASURED / FITTED only) | Rationale |
|---|------|-------------|--------------------------|----------------------------------------|-----------|
| Q1 | a | | P1(b) | | |
| Q1 | b | | P1(b) | | |
| Q2 | a | | P1(a–b) | | |
| Q2 | b | | P1(a–b) | | |
| Q3 | a | | P1(c) | | |
| Q3 | b | | P1(c) | | |
| Q4 | a | | P1(d) | | |
| Q4 | b | | P1(d) | | |
| Q5 | a | | P2(b) | | |
| Q5 | b | | P2(b) | | |
| Q6 | a | | P2(c) | | |
| Q6 | b | | P2(c) | | |
| Q7 | a | | P3(a)ii | | |
| Q7 | b | | P3(a)ii | | |
| Q8 | a | | P3(b) | | |
| Q8 | b | | P3(b) | | |
| Q9 | a | | P4(b) | | |
| Q9 | b | | P4(b) | | |
| Q10 | a | | P4(c) | | |
| Q10 | b | | P4(c) | | |
| Q11 | a | | P2(d) | | |
| Q11 | b | | P2(d) | | |
| Q12 | a | | P3(a)i | | |
| Q12 | b | | P3(a)i | | |
| Q13 | a | | P1(a) | | |
| Q13 | b | | P1(a–b) | | |
| Q14 | a | | P1(b) | | |
| Q14 | b | | P1(b) | | |
| Q15 | a | | P1(b) | | |
| Q15 | b | | P1(b) | | |
| Q16 | a | | P1(b) | | |
| Q16 | b | | P1(b) | | |
| Q17 | a | | P1(a–b) | | |
| Q17 | b | | P1(a–b) | | |
| Q18 | a | | P1(b) | | |
| Q18 | b | | P1(b) | | |
| Q19 | a | | P1(b) | | |
| Q19 | b | | P1(a–b) | | |
| Q20 | a | | P1(a–b) | | |
| Q20 | b | | P1(b) | | |
| Q21 | a | | P1(a–c) | | |
| Q21 | b | | P1(a–c) | | |
| Q22 | a | | P2(a) B2 | | |
| Q22 | b | | P2(a) B2 | | |
| Q23 | a | | P2(d) B2 | | |
| Q23 | b | | P2(d) B2 | | |
| Q24 | a | | P3(a)i B2 | | |
| Q24 | b | | P3(b) B2 | | |
| Q25 | a | | P5 expl. | | |
| Q25 | b | | P5 expl. | | |
| Q25 | c | | P5 expl. | | |
| Q26 | a | | P4(a) | | |
| Q26 | b | | P4(a) | | |
| Q27 | a | | P4(a) | | |
| Q27 | b | | P4(a) | | |
| Q28 | a | | P4(b) | | |
| Q28 | b | | P4(b) | | |
| Q29 | a | | P4(c) | | |
| Q29 | b | | P4(c) | | |

---

## Answer key

**How to read.** Keep the original **PREDICTION** option. Fill **Resolved** only when evidence exists (**MEASURED**, **FITTED**, or **EXTRAPOLATED**). Never erase a wrong PREDICTION — the miss is the learning signal. Leave Resolved = **—** with tag **PENDING** when the metric is missing (mid-dip, EMA, P3 η-change, mid-budget HF, P4 optional train/k-ablate, `P5-L-cool-dir`). **P5** otherwise MEASURED 6/7 (see Q25 / rule 8). P4 stage-a **13/13** ledger apply: Q26 / Q27a from ledger HIT/MISS; Q27b Resolved **HIT** (MEASURED excess_final=0.08727119221832025). Do **not** leave SGDR / P3 clean-edge / AdamW precond blank — those are MEASURED @ Batch 2 9/9.

Sources for MEASURED cells: `media/p1/barrier-table.csv`, `media/p1/barrier-summary.json`, `media/p1/a3-mode-connectivity.json`, `media/p2/batch2_partial_finals.csv` (jobs 0–6), `media/p2/p3_sgd_sharpness_summary.json`, `media/p2/p3_adamw_sharpness_summary.json`, P4 13/13: `media/p4/stage-a/a-adamw-*.json` (\(\lambda@9375=5.967310256347902\); ledger `docs/a3-prediction-ledger.md`). Fits: intuition guide Appendix Q1.1. Batch 2 PREDICTIONs locked `2026-10-11T00:32:07Z` — mid-run W&B not used.

| Q | Part | PREDICTION (option) | Resolved | Tag | Mechanism / evidence |
|---|------|---------------------|----------|-----|----------------------|
| Q1 | a | **3** `[0.05, 0.2)` | **4** `[0.2, ∞)` · \(B=0.6237\) | MEASURED | Early fork + long branch → large barrier (magnitude miss) |
| Q1 | b | **1** `(−∞, 0.01]` | **1** · \(B=-0.0597\) | MEASURED | Long prefix + short branch → averaging works (hit) |
| Q2 | a | **3** | **3** · all four failure \(G\ge 0.01\) | MEASURED | Output ensemble bypasses parameter-space ridge (hit) |
| Q2 | b | **2** `[0.02, 0.08)` | **4** `[0.2, ∞)` · median \(G\approx 0.279\) | MEASURED | Gaps \(0.1311,0.2309,0.3273,0.7281\) (magnitude miss) |
| Q3 | a | **1** | **1** · \(\alpha=0.5\) worst on \((50,500)\) | MEASURED | Unimodal midpoint barrier when \(B\) large (hit) |
| Q3 | b | **3** `[0.05, 0.2)` | **3** · \(B=0.1299\) | MEASURED | Mid-band cell \((50,500)\) (hit) |
| Q4 | a | **3** `[0.02, 0.1)` | — | PENDING | `P1-d-dB-cos` · no cosine-fork MEASURED |
| Q4 | b | **3** Depends on \(B_W\) | — | PENDING | Note: MEASURED \(B_W(250,250)=-0.0779\) already \(\le 0.01\) under `wsd0.0` |
| Q5 | a | **3** `[0.5, 0.75)` | — | PENDING | B2-locked · `P2-ema-gap-frac` · no EMA gap MEASURED |
| Q5 | b | **3** | — | PENDING | B2-locked · no EMA ranking MEASURED |
| Q6 | a | **2** `[−0.02, 0.02)` | **2** · \(R=0.0012464523315429688\) vs `wsd0.2` | MEASURED | `rms7iyyt`−`soosjcxr`; band HIT |
| Q6 | b | **1** `(−∞, −0.02)` | — | PENDING | mid-dip not in `batch2_partial_finals.csv` |
| Q7 | a | **3** `[350, 450)` | — | PENDING | η-half arm not run · PREDICTION \(2/0.005=400\) |
| Q7 | b | **2** `[80, 120)` | — | PENDING | η-double arm not run · PREDICTION \(2/0.02=100\) |
| Q8 | a | **3** `[30, 50)` | **3** · \(\eta\lambda^{\mathrm{pre}}=30.987715914514\) | MEASURED | `B2-P3-adamw-pre` · HIT near 38 |
| Q8 | b | **2** | **2** · plateau \(\approx 30.15\); not stuck at 2 | MEASURED | noisier band near ~38 vs full-batch \(\eta\lambda\approx 2\) |
| Q9 | a | **1** `[0, 0.002)` | **4** · |ΔL|=2.73853462934494 | MEASURED | `P4-invar-dL` · **MISS** · file-exact from dry-run |
| Q9 | b | **4** or **1** | **1** · ratio=0.4658784038724411 | MEASURED | `P4-invar-dL` · **MISS** · file-exact from dry-run |
| Q10 | a | **3** `[0.6, 0.85)` | **1** · mean=0.09296940652964016 | MEASURED | `P4-top10-frac` · **MISS** · file-exact from dry-run |
| Q10 | b | **1** | **3** · gap=-0.000344693660736084 | MEASURED | `P4-c-loss-rank` · **MISS** · L=(3.582527816295624,5.502330541610718,3.5821831226348877) |
| Q11 | a | **3** `[0.01, 0.08)` | **4** · \(\Delta L=0.09225320816040039\in[0.08,\infty)\) | MEASURED | `1jri34em` vs linear; band MISS (SGDR likewise MISS — see Q23) |
| Q11 | b | **2** | — | PENDING | Mid-budget stop not in Batch 2 required stop |
| Q12 | a | **2** `[150, 250)` | **2** · median \(\lambda_{\max}=245.16314737956807\) | MEASURED | `B2-P3-sgd-lam` · `0dewkykj` / `p3_sgd_sharpness_summary.json` HIT |
| Q12 | b | **2** | **2** · late train ↓ (mid≈2.14@1224 → final 1.2686@1499) | MEASURED | Cohen-style continue-decrease; part of `B2-P3-sgd-lam` |
| Q13 | a | **2** `[0, 0.01)` | **3** · \(G_{\mathrm{ens}}=3.7483-3.6454=0.1029\) | MEASURED | Soft ensemble beats best endpoint by \(\ge 0.01\) (sign/magnitude miss vs PREDICTION) |
| Q13 | b | **1** | **1** · \(L_{\mathrm{ens}}=3.6454\) vs \(L_{\mathrm{avg}}=4.3734\) | MEASURED | High-\(B\) cell; ensemble ≪ weight avg (hit) |
| Q14 | a | **2** `[0.9, 1.4)` | **2** · growth-lin \(\approx 1.03\); quad \(\approx 1.22\) | EXTRAPOLATED | Fit A beyond \(y=500\); not MEASURED |
| Q14 | b | **3** | **3** | EXTRAPOLATED | Growth regime: longer \(y\) raises \(B\) at \(x=0\) |
| Q15 | a | **2** `[0.25, 0.40)` | **2** · exp \(\approx 0.319\); lin-interp \(\approx 0.377\) | EXTRAPOLATED | Fit B + MEASURED \((0,500),(50,500)\) |
| Q15 | b | **2** | **2** | EXTRAPOLATED | Monotone in \(x\) between anchors |
| Q16 | a | **2** `[-0.12, -0.01)` | **2** · asymptote \(\approx -0.066\) | EXTRAPOLATED | Fit B \(x\to\infty\) at \(y=500\) |
| Q16 | b | **1** Yes | **1** | EXTRAPOLATED | Asymptote well below \(0.01\) |
| Q17 | a | **2** `[1.0, 1.5)` | **2** · \(G(B_{\mathrm{lin}})\approx 1.15\); \(G(B_{\mathrm{quad}})\approx 1.35\) | EXTRAPOLATED | Fit C ∘ Fit A |
| Q17 | b | **3** | **3** | EXTRAPOLATED | Larger \(B\) ⇒ larger \(G\) |
| Q18 | a | **4** `[0.2, \infty)` | **4** · \(B=0.2173\) | MEASURED | Failure cell \((0,250)\) |
| Q18 | b | **2** | **2** · \(B(0,100)=-0.0275 < B(0,250)=0.2173 < B(0,500)=0.6237\) | MEASURED | Growth along \(x=0\) |
| Q19 | a | **2** `(0.01, 0.05)` | **2** · \(B=0.0319\) | MEASURED | Mildest failure cell |
| Q19 | b | **2** `[0.05, 0.15)` | **2** · \(G=0.1311\) | MEASURED | Smallest failure gap |
| Q20 | a | **2** `[0.2, 0.5)` | **4** `[0.8, 1]` · \(26/30=0.867\) | MEASURED | Connectivity under-predicted |
| Q20 | b | **2** | **2** · failures \((0,250),(0,500),(50,500),(100,500)\) | MEASURED | Small \(x\), large \(y\) |
| Q21 | a | **3** `[0.20, 0.25)` | **3** · FITTED \(G\approx 0.233\); MEASURED \(G=0.2309\) | FITTED+MEASURED | Fit C on mid-band cell (hit) |
| Q21 | b | **2** | **2** · MEASURED \(\bar L-L_{\mathrm{ens}}\) mean \(\approx 0.089\) | FITTED | \(G\approx B+0.09\) |
| Q22 | a | **2** `[0.02, 0.08)` | **4** `[0.15, \infty)` · \(R=0.16930222511291504\) (band miss; worst-of-four hit) | MEASURED | `B2-P2-wsd0.0-R` · constant worse than band |
| Q22 | b | **2** | **2** · linear≺cos≺wsd0.2≪wsd0.0; \(R_{\mathrm{lin}}=0\) \(R_{\mathrm{cos}}=0.005916595458984375\) \(R_{\mathrm{wsd0.2}}=0.009538650512695312\); \(L\): linear(2.9282028675079346) < cos(2.934119462966919) < wsd0.2(2.93774151802063) < wsd0.0(3.0975050926208496) (rank HIT; wsd0.2 top-2 MISS) | MEASURED | `P2-sched-rank` + `B2-P2-linear-R` / `B2-P2-cos-R` / `B2-P2-wsd0.2-R` |
| Q23 | a | **3** `[0.01, 0.08)` | **4** · \(\Delta L=0.09225320816040039\in[0.08,\infty)\) | MEASURED | `B2-P2-invsqrt-dL` · `1jri34em` vs linear; band MISS |
| Q23 | b | **1** `[0, 0.03)` | **1** · \(\lvert L_{\mathrm{sgdr}}-L_{\mathrm{invsqrt}}\rvert=0.014348745346069336\); \(\Delta L_{\mathrm{sgdr}}=0.10660195350646973\) band MISS vs linear | MEASURED | `B2-P2-sgdr-dL` · job 6 `51b3qq4n` @ 9/9 · \(L_{\mathrm{sgdr}}=3.0348048210144043\) \(L_{\mathrm{linear}}=2.9282028675079346\) |
| Q24 | a | **2** `[150, 250)` | **2** · median \(\lambda_{\max}=245.16314737956807\) | MEASURED | `B2-P3-sgd-lam` · HIT near \(2/\eta=200\) |
| Q24 | b | **3** `[30, 50)` | **3** · \(\eta\lambda^{\mathrm{pre}}=30.987715914514\) | MEASURED | `B2-P3-adamw-pre` · HIT |
| Q25 | a | **1** `(−∞, 0.01]` | **4** · B_C=0.7697049379348755; dB=-0.14600493793487546 (B_R=0.6237 [JSON] − B_C=0.7697049379348755) | MEASURED | `P5-B-cool-x0` · **MISS** · file-exact from dry-run |
| Q25 | b | **4** `[0.5, ∞)` | **4** · B_K=0.7320129871368408 | MEASURED | `P5-B-const-ctrl` · **HIT** · file-exact from dry-run |
| Q25 | c | **1** same-valley walls | **2** · B_C=0.7697049379348755 B_R=0.6237 B_K=0.7320129871368408 | MEASURED | `P5-emergent-basin` · **MISS**; `P5-rank-arms` · **MISS** · file-exact from dry-run |
| Q26 | a | **3** attention or mlp | **2** · late=`head` (\(5.641603135007765\)) | MEASURED | `P4-a-family-sharpest` · **MISS** (not attention\|mlp); ratio \(7.38\notin[1.5,5)\) |
| Q26 | b | **2** | **2** partial · early_top2=`embedding`,`attention`; late=`head` | MEASURED | init has embed (not head); late head-led — row **MISS** on late ranking |
| Q27 | a | **2** | shrink failed · neg_frac \(0.00451\to0.01491\to0.93446\to0.06965\) | MEASURED | `P4-a-density-neg` · shrink_0→9375=false · **MISS** (final still \(\in[0,0.15)\)) |
| Q27 | b | **2** excess `[0.05,0.30)` | **2** · excess_final=0.08727119221832025; fam_above=True; layer_above=True | MEASURED | `P4-a-block-energy` · **HIT** · file-exact from dry-run |
| Q28 | a | **1** | — | PENDING | `P4-b-train-2x` · b not started |
| Q28 | b | **2** failed arm | — | PENDING | `P4-b-train-2x` · b not started |
| Q29 | a | **2** `[0.35, 0.65)` | — | PENDING | `P4-c-k-ablate` · c not started |
| Q29 | b | **2** | — | PENDING | `P4-c-k-ablate` · c not started |

### How to promote PREDICTION → MEASURED / confirm EXTRAPOLATED

1. Run the corresponding starter / ledger ID (or measure an extrapolated cell).  
2. Write the measured option in **Resolved** with tag **MEASURED**.  
3. If MEASURED ≠ PREDICTION, keep both; update intuition guide §A/D and mark ledger row `resolved`.  
4. For EXTRAPOLATED rows: leave Observed empty until a future measure; do not paste fit points into Observed as if measured.  
5. Never erase a wrong PREDICTION without recording the miss — that is the learning signal for the Oct 20 quiz.  
6. For Batch 2 (`B2-*` / Q5–Q6, Q8, Q11–Q12, Q22–Q24): promote only from **final** metrics — never from mid-run W&B snapshots. All nine `B2-*` P2/P3 band rows promoted @ 9/9 (incl. both P3 sharpness HITs); Q5 EMA / Q6b mid-dip still PENDING.  
7. For P4 @ **13/13**: Q26 / Q27a filled from ledger apply (`family` **MISS**, dens-neg **MISS**; sibling `norm-sharpness` **HIT**, `lam-9375-earlyfit` **MISS**). Q27b Resolved **HIT** from ledger apply (excess_final=0.08727119221832025). b/c scored: `P4-invar-dL` **MISS**, `P4-top10-frac` **MISS**, `P4-c-loss-rank` **MISS**. Q28–Q29 stay PENDING without train/k-ablate. Status: `internal/p4-bc-apply-readme.md` · dry-runs under `internal/`.
8. For P5 exploratory (Q25 / `P5-*`): scored MEASURED: `P5-B-cool-x0` **MISS**, `P5-dB-cool` **MISS**, `P5-B-const-ctrl` **HIT**, `P5-rank-arms` **MISS**, `P5-G-cool-fitC` **HIT**, `P5-emergent-basin` **MISS**; `P5-L-cool-dir` **INSUFFICIENT** (`dL=null` / \(L_R,L_C\) missing — no HIT/MISS). Keep PREDICTION options if miss; see `internal/p5-fill-on-complete.md`.
