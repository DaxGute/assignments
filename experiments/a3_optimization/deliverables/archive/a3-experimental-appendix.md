# A3 Experimental Appendix

Configs, seeds, plot refs, and raw measurements for the official report.  
**Markdown only** (report support) — not a separate study PDF.

## Environment

| Field | Value |
|-------|-------|
| Repo branch | `cursor/a3-from-desktop` (merged `a3-optimization2-81fe` @ `958f46c`) |
| Starter SHA | course `a3_release` `57c60bb` |
| Modal env | `cs312-daxton` (Mac / course auth) |
| W&B | `daxgute-stanford-university` / `assignments` (**MEASURED** from Batch 1 prefix) |
| Study PDF policy | report + intuition guide (+ optional quiz); no appendix PDF |

---

## P1 Mode Connectivity

### Config

| Key | Value |
|-----|-------|
| `lr_schedule` | `wsd0.0` |
| PREFIX_STEPS $x$ | `{0, 50, 100, 250, 500, 1000}` |
| BRANCH_STEPS $y$ | `{25, 50, 100, 250, 500}` |
| `batch_order_seed` | `{1, 2}` |
| EXPERIMENT_KEY | `a3-mode-connectivity` |
| peak LR | `0.003` |
| seed (model/data) | `model_seed=42`, `data_seed=42` |
| Checkout | `~/Desktop/assignments` @ `cursor/a3-from-desktop` `958f46c` |

### Runs

| Stage | Run name / ID | Status | Notes |
|-------|---------------|--------|-------|
| prefix | `model-d8-lr0.003-tok614M-wsd0.0-stop1000` · W&B `rq1k0bz1` · Modal `ap-L6RKrCcPCbhhgBvr3DLJfU` | **DONE** | end `val_loss=3.48832` |
| branches | 12 runs · Modal `ap-57b3QORBZcKneFVkuuq7F3` | **DONE** | all keep-ckpts on volume |
| measure | Modal `ap-ZU7fsZIl4Mq2Wi5veNzXCb` | **DONE** | JSON below |

### Artifacts

| Artifact | Path |
|----------|------|
| Raw measure JSON (repo) | `experiments/a3_optimization/results/a3-mode-connectivity.json` (20091 B, 2026-10-10 17:21 PDT) |
| Store copy | `media/p1/a3-mode-connectivity.json` |
| Barrier table CSV | `media/p1/barrier-table.csv` |
| Summary JSON | `media/p1/barrier-summary.json` |

### Plots

| Figure | Content | Path | Label |
|--------|---------|------|-------|
| P1-b heatmap | Annotated \(B(x,y)\) | `media/p1/plots/B_heatmap.png` | **MEASURED** |
| P1 Fit A | \(B\) vs \(y\) (+ FITTED lines) | `media/p1/plots/B_vs_y_curves.png` | MEASURED points + **FITTED** |
| P1 Fit B | \(B\) vs \(x\) @ \(y=500\) | `media/p1/plots/B_vs_x_y500.png` | MEASURED points + **FITTED** |
| P1-a / Fit C | \(L_{\mathrm{ens}}\) vs \(L_{\mathrm{avg}}\); \(G\) vs \(B\) | `media/p1/plots/ensemble_vs_avg.png` | MEASURED points + **FITTED** |

Barrier numbers: `media/p1/barrier-table.csv` · `media/p1/barrier-summary.json`.  
FITTED equations (Fits A/B/C): `internal/p1-scaling-fit-status.md` · intuition-guide Appendix Q1.1. **Never label FITTED as MEASURED.**

### Barrier summary (**MEASURED**)

$B = L(\theta_{\mathrm{avg}}) - \tfrac12(L_1+L_2)$ at $\alpha=0.5$.  
Source: `media/p1/barrier-summary.json` (derived from `a3-mode-connectivity.json`).

| Stat | Value |
|------|-------|
| Cells | 30 |
| $B_{\min}$ | −0.1161 at $(x,y)=(250,100)$ |
| $B_{\max}$ | 0.6237 at $(0,500)$ |
| $B_{\mathrm{mean}}$ | −0.0334 |
| $B\le 0.01$ | **26 / 30** |
| Failures ($B>0.01$) | $(0,250)$, $(0,500)$, $(50,500)$, $(100,500)$ |

### $B$ heatmap (**MEASURED**; ✓ = $B\le 0.01$)

| $x$ \\ $y$ | 25 | 50 | 100 | 250 | 500 |
|------------|------|------|------|------|------|
| 0 | −0.0288 ✓ | −0.0416 ✓ | −0.0275 ✓ | 0.2173 | 0.6237 |
| 50 | −0.1009 ✓ | −0.1053 ✓ | −0.0910 ✓ | 0.0055 ✓ | 0.1299 |
| 100 | −0.0932 ✓ | −0.1019 ✓ | −0.0972 ✓ | −0.0512 ✓ | 0.0319 |
| 250 | −0.0963 ✓ | −0.1106 ✓ | −0.1161 ✓ | −0.0779 ✓ | −0.0522 ✓ |
| 500 | −0.0747 ✓ | −0.0915 ✓ | −0.0953 ✓ | −0.0910 ✓ | −0.0763 ✓ |
| 1000 | −0.0597 ✓ | −0.0769 ✓ | −0.0855 ✓ | −0.0855 ✓ | −0.0825 ✓ |

### Measured cells — endpoints, ensemble, midpoint, $B$ (**MEASURED**; ✓ means $B\le 0.01$)

| x | y | L(θ₁) | L(θ₂) | L_ens | L(θ_avg) | B |
|---|---|-------|-------|-------|----------|---|
| 0 | 25 | 6.4380 | 6.4653 | 6.4077 | 6.4228 | -0.0288 ✓ |
| 0 | 50 | 5.6329 | 5.6864 | 5.5798 | 5.6180 | -0.0416 ✓ |
| 0 | 100 | 5.0282 | 5.0396 | 4.9238 | 5.0064 | -0.0275 ✓ |
| 0 | 250 | 4.3202 | 4.3092 | 4.2046 | 4.5319 | 0.2173 |
| 0 | 500 | 3.7511 | 3.7483 | 3.6454 | 4.3734 | 0.6237 |
| 50 | 25 | 5.2708 | 5.2877 | 5.1962 | 5.1784 | -0.1009 ✓ |
| 50 | 50 | 5.0453 | 5.0449 | 4.9441 | 4.9398 | -0.1053 ✓ |
| 50 | 100 | 4.6943 | 4.6940 | 4.5872 | 4.6031 | -0.0910 ✓ |
| 50 | 250 | 4.1278 | 4.1245 | 4.0231 | 4.1316 | 0.0055 ✓ |
| 50 | 500 | 3.7009 | 3.6898 | 3.5943 | 3.8253 | 0.1299 |
| 100 | 25 | 4.8268 | 4.8219 | 4.7374 | 4.7311 | -0.0932 ✓ |
| 100 | 50 | 4.6870 | 4.6719 | 4.5790 | 4.5776 | -0.1019 ✓ |
| 100 | 100 | 4.4887 | 4.4439 | 4.3591 | 4.3691 | -0.0972 ✓ |
| 100 | 250 | 3.9961 | 3.9747 | 3.8808 | 3.9342 | -0.0512 ✓ |
| 100 | 500 | 3.6670 | 3.6582 | 3.5634 | 3.6945 | 0.0319 |
| 250 | 25 | 4.2400 | 4.2449 | 4.1581 | 4.1461 | -0.0963 ✓ |
| 250 | 50 | 4.1383 | 4.1451 | 4.0505 | 4.0311 | -0.1106 ✓ |
| 250 | 100 | 3.9972 | 3.9767 | 3.8869 | 3.8709 | -0.1161 ✓ |
| 250 | 250 | 3.7429 | 3.7357 | 3.6447 | 3.6615 | -0.0779 ✓ |
| 250 | 500 | 3.5740 | 3.5712 | 3.4844 | 3.5204 | -0.0522 ✓ |
| 500 | 25 | 3.7266 | 3.7154 | 3.6522 | 3.6463 | -0.0747 ✓ |
| 500 | 50 | 3.7105 | 3.6971 | 3.6226 | 3.6123 | -0.0915 ✓ |
| 500 | 100 | 3.6654 | 3.6611 | 3.5798 | 3.5680 | -0.0953 ✓ |
| 500 | 250 | 3.5767 | 3.5748 | 3.4918 | 3.4847 | -0.0910 ✓ |
| 500 | 500 | 3.4862 | 3.4857 | 3.4037 | 3.4097 | -0.0763 ✓ |
| 1000 | 25 | 3.4786 | 3.4796 | 3.4221 | 3.4194 | -0.0597 ✓ |
| 1000 | 50 | 3.4730 | 3.4757 | 3.4051 | 3.3974 | -0.0769 ✓ |
| 1000 | 100 | 3.4619 | 3.4611 | 3.3867 | 3.3760 | -0.0855 ✓ |
| 1000 | 250 | 3.4241 | 3.4225 | 3.3449 | 3.3378 | -0.0855 ✓ |
| 1000 | 500 | 3.3812 | 3.3810 | 3.3017 | 3.2986 | -0.0825 ✓ |

---

## P2 River-Valley

### Config

| Key | Value |
|-----|-------|
| schedules | `wsd0.0`, `linear`, `cos`, `wsd0.2` |
| `ema_decays` | `(0.99, 0.999)` on schedule arms |
| `lr_dip` | on one `wsd0.2` arm (`dip0.4-0.45-0.5x0.1`) |
| horizon-free | `invsqrt`, `sgdr2000` |
| Batch | Batch 2 · Modal `ap-Izg7CMIJgyf29zQokn4SYv` |

### Batch 2 jobs (**MEASURED** 9/9)

Finals from `media/p2/batch2_finals.csv` (`state=finished`, `progress=1.0`). P3 sharpness from `media/p2/p3_sgd_sharpness_summary.json` / `media/p2/p3_adamw_sharpness_summary.json` only — blanks mean not in those sources (do not invent).

| # | label | run_name | W&B | state | Label | train_loss | val_loss | optimizer_step | med. \(\lambda_{\max}\) | med. \(\eta\lambda\) / \(\eta\lambda^{\mathrm{pre}}\) | sharpness window |
|---|-------|----------|-----|-------|-------|------------|----------|----------------|-------------------------|-----------------------------------------------------|------------------|
| 0 | `a3-b2-p2-wsd0.0` | `model-d8-lr0.003-tok614M-wsd0.0-ema0.99-0.999` | [`widt420l`](https://wandb.ai/daxgute-stanford-university/assignments/runs/widt420l) | finished | **MEASURED** | 3.0593721866607666 | 3.0975050926208496 | 9374 | | | |
| 1 | `a3-b2-p2-linear` | `model-d8-lr0.003-tok614M-ema0.99-0.999` | [`r0294p93`](https://wandb.ai/daxgute-stanford-university/assignments/runs/r0294p93) | finished | **MEASURED** | 2.886288642883301 | 2.9282028675079346 | 9374 | | | |
| 2 | `a3-b2-p2-cos` | `model-d8-lr0.003-tok614M-cos-ema0.99-0.999` | [`sktw67q8`](https://wandb.ai/daxgute-stanford-university/assignments/runs/sktw67q8) | finished | **MEASURED** | 2.8933908939361572 | 2.934119462966919 | 9374 | | | |
| 3 | `a3-b2-p2-wsd0.2` | `model-d8-lr0.003-tok614M-wsd0.2-ema0.99-0.999` | [`soosjcxr`](https://wandb.ai/daxgute-stanford-university/assignments/runs/soosjcxr) | finished | **MEASURED** | 2.8900909423828125 | 2.93774151802063 | 9374 | | | |
| 4 | `a3-b2-p2-wsd0.2-dip` | `model-d8-lr0.003-tok614M-wsd0.2-dip0.4-0.45-0.5x0.1` | [`rms7iyyt`](https://wandb.ai/daxgute-stanford-university/assignments/runs/rms7iyyt) | finished | **MEASURED** | 2.8953793048858643 | 2.938987970352173 | 9374 | | | |
| 5 | `a3-b2-p2-invsqrt` | `model-d8-lr0.003-tok614M-invsqrt-warmupsteps94` | [`1jri34em`](https://wandb.ai/daxgute-stanford-university/assignments/runs/1jri34em) | finished | **MEASURED** | 2.983588457107544 | 3.020456075668335 | 9374 | | | |
| 6 | `a3-b2-p2-sgdr2000` | `model-d8-lr0.003-tok614M-sgdr2000-warmupsteps94-minlr0.1` | [`51b3qq4n`](https://wandb.ai/daxgute-stanford-university/assignments/runs/51b3qq4n) | finished | **MEASURED** | 2.996020793914795 | 3.0348048210144043 | 9374 | | | |
| 7 | `a3-b2-p3-0-sgd` | `model-d8-lr0.01-bs32-epochs1500.0-tok32.8K-wd0.0-emb16-nogradclip-b10.0-sgd-constant` | [`0dewkykj`](https://wandb.ai/daxgute-stanford-university/assignments/runs/0dewkykj) | finished | **MEASURED** | 1.268615484237671 | 8.053705215454102 | 1499 | 245.16314737956807 | 2.451631473795681 | updates 1200–1500 (n=12); raw \(\lambda_{\max}\), \(\eta\lambda\) |
| 8 | `a3-b2-p3-1-adamw` | `model-d8-lr0.003-epochs0.32-tok614M-warmup0.03125-wsd0.0` | [`foew12y0`](https://wandb.ai/daxgute-stanford-university/assignments/runs/foew12y0) | finished | **MEASURED** | 3.2420499324798584 | 3.2397050857543945 | 2999 | 10329.238638171333 | 30.987715914514 | post-warmup (n=25); \(\lambda_{\max}^{\mathrm{pre}}\), \(\eta\lambda^{\mathrm{pre}}\) |

Artifacts: `media/p2/batch2_finals.csv`, `media/p2/batch2_final_metrics.json`, `media/p2/p3_sgd_sharpness_summary.json`, `media/p2/p3_adamw_sharpness_summary.json`, `media/p2/p3_sharpness_medians.csv`


## P3 Edge of Stability

### Config

| Key | Value |
|-----|-------|
| Starter core | full-batch SGD + minibatch AdamW sharpness (2 jobs in Batch 2) |
| Batch | Batch 2 · Modal `ap-Izg7CMIJgyf29zQokn4SYv` |

### Runs (**MEASURED** finals + sharpness)

Jobs 7–8 from the Batch 2 table above. Sharpness columns only:

| # | label | W&B | Label | train_loss | val_loss | step | med. \(\lambda_{\max}\) | med. \(\eta\lambda\) / \(\eta\lambda^{\mathrm{pre}}\) | Source field |
|---|-------|-----|-------|------------|----------|------|-------------------------|-----------------------------------------------------|--------------|
| 7 | `a3-b2-p3-0-sgd` | [`0dewkykj`](https://wandb.ai/daxgute-stanford-university/assignments/runs/0dewkykj) | **MEASURED** | 1.268615484237671 | 8.053705215454102 | 1499 | 245.16314737956807 | 2.451631473795681 | `lambda_max_median_1200_1500` / `eta_lambda_median_1200_1500` |
| 8 | `a3-b2-p3-1-adamw` | [`foew12y0`](https://wandb.ai/daxgute-stanford-university/assignments/runs/foew12y0) | **MEASURED** | 3.2420499324798584 | 3.2397050857543945 | 2999 | 10329.238638171333 | 30.987715914514 | `post_warmup_median_lambda_max_pre` / `post_warmup_median_eta_lambda_pre` |

Artifacts: `media/p2/batch2_finals.csv`, `media/p2/p3_*_sharpness_summary.json` (twins under `media/p3/`)


## P4 Inside the Hessian

**Label rule:** fill P4 value cells only from store JSON (`media/p4/…` / `hessian_jobs.load_result`). Stage a is **13/13 COMPLETE** — all `a-adamw-*` steps are **MEASURED**. Do **not** invent \(\lambda\) / density / subspace numbers. Handout bands \(\lambda_{\max}\in[4.5,6.0]\), val \(\in[2.925,2.931]\) are assignment-given, not our MEASURED. Ledger `P4-a-*` HIT/MISS from `internal/p4-9375-hit-miss-summary.md` (operator copy of dry-run DRAFT_*); `P4-a-block-energy` **HIT** (MEASURED excess_final=0.08727119221832025; fam_above=True; layer_above=True from `a-random-energy-1526/9375`). Stages b/c scored: `P4-invar-dL` / `P4-top10-frac` / `P4-c-loss-rank` **MISS** (see dry-runs).

### Config

| Key | Value |
|-----|-------|
| Batch | Batch 3 · Modal stage `a` `ap-DAUDZsFRech705uMDN7q3x` (**13/13** volume; `a-adamw-{0,3,15,61,153,305,763,1526,3052,4578,6104,7630,9375}` in store) |
| Probe rows | 64 |
| Precision | FP32, TF32 off |
| Provided ckpts | shared volume `hard-dl-dclm-v1` → `/root/shared_data/a3_hessian/runs/` |
| AdamW measure steps (`STEPS`) | `0, 3, 15, 61, 153, 305, 763, 1526, 3052, 4578, 6104, 7630, 9375` |
| Density at | `(0, 153, 1526, 9375)` |
| Families | `embedding`, `attention`, `qk_norm`, `mlp`, `norm`, `head` |
| (c) continue | SGDM @ 1526; \(k=10\); 300 steps; refresh 25; arms `full` / `top` / `removed` |

### Stage status

| Stage | Job pattern | Count (starter) | Status | Result path |
|-------|-------------|-----------------|--------|-------------|
| `a` | `a-adamw-{step}` | 13 | **DONE** 13/13 | `media/p4/stage-a/a-adamw-{0,3,15,61,153,305,763,1526,3052,4578,6104,7630,9375}.json` (+ summaries) |
| `b-rescale` | `b-rescale-reference` + `b-rescale-head-x0.25.json` | ≥2 | **DONE** — `P4-invar-dL` **MISS** | `media/p4/` + `/cursor/stores/bc-01947f99-cdd8-4c04-97a8-01a1028bf8af/media/p4/b-rescale-head-x0.25.json` |
| `b-train` | `TrainConfig` arms (freeze before launch) | TBD | **PENDING** | |
| `b-measure` | `b-final-{run}` | TBD | **PENDING** | |
| `c` | `c-full`, `c-top`, `c-removed` | 3 | **DONE** | `media/p4/stage-c/` |
| `fetch` | downloads finished JSON | — | **DONE** — stage a + b-rescale heads (x0.25/x0.5/x2) + c 3/3 in `media/p4/` | `results/` → store `media/p4/` |

### P4(a) — AdamW checkpoint measure (**PARTIAL** 13/13)

MEASURED rows from `media/p4/stage-a/` JSON/summary only. Ledger `P4-a-*` stays unresolved (**INSUFFICIENT**) until mid/final stage-a jobs land.

| Job | step | Label | \(\lambda_{\max}\) | \(\lambda_{\min}\) | density? | Notes |
|-----|------|-------|--------------------|--------------------|----------|-------|
| `a-adamw-0` | 0 | **MEASURED** | 1431.5030691763686 | -1410.061273996107 | yes | tokens=0; probe_loss=8.795355558395386 |
| `a-adamw-3` | 3 | **MEASURED** | 1571.7420262270407 | -601.6596057825041 |  | tokens=196608; probe_loss=8.3955197930336 |
| `a-adamw-15` | 15 | **MEASURED** | 53.36426571742344 | -17.452665724452103 |  | tokens=983040; probe_loss=6.921800225973129 |
| `a-adamw-61` | 61 | **MEASURED** | 18.054717868295125 | -1.8957655982714006 |  | tokens=3997696; probe_loss=5.400916934013367 |
| `a-adamw-153` | 153 | **MEASURED** | 6.502395300121604 | -0.2562527330248557 | yes | tokens=10027008; probe_loss=4.60715526342392 |
| `a-adamw-305` | 305 | **MEASURED** | 4.32926630998023 | -0.06854534676741643 |  | tokens=19988480; probe_loss=4.108570218086243 |
| `a-adamw-763` | 763 | **MEASURED** | 3.02398152405625 | -0.08332850048247277 |  | tokens=50003968; probe_loss=3.5742459148168564 |
| `a-adamw-1526` | 1526 | **MEASURED** | 2.1954286466134665 | -0.0637674257861686 | yes | tokens=100007936; probe_loss=3.3788233399391174 |
| `a-adamw-3052` | 3052 | **MEASURED** | 3.1637529312879873 | -0.06470202230389413 |  | tokens=200015872; probe_loss=3.2200800478458405 |
| `a-adamw-4578` | 4578 | **MEASURED** | 3.7423705126158477 | -0.04897616951626389 |  | tokens=300023808; probe_loss=3.1347245424985886 |
| `a-adamw-6104` | 6104 | **MEASURED** | 4.155164815834522 | -0.03642192924304736 |  | tokens=400031744; probe_loss=3.0628973245620728 |
| `a-adamw-7630` | 7630 | **MEASURED** | 5.281516516770698 | -0.049697684311825786 |  | tokens=500039680; probe_loss=3.006814658641815 |
| `a-adamw-9375` | 9375 | **MEASURED** | 5.967310256347902 | -0.08023598879266304 | yes | tokens=614400000; probe_loss=2.9474442899227142; final |

### P4(a) — Family \(\lambda_{\max}\) by step (**PARTIAL**; steps 0, 3, 15, 61, 153, 305, 763, 1526, 3052, 4578, 6104, 7630, 9375 MEASURED)

| step | embedding | attention | qk_norm | mlp | norm | head | Label |
|------|-----------|-----------|---------|-----|------|------|-------|
| 0 | 1414.2983976747726 | 9.878542601155342 | 0.006507140133876505 | 7.0111895050073985 | 0.019894242489013118 | 0.10186688249987938 | **MEASURED** |
| 3 | 1553.9202300399772 | 11.031923073793246 | 0.007607915139353549 | 10.39684728301151 | 0.022457015802515027 | 1.3887433073010218 | **MEASURED** |
| 15 | 45.81142761722067 | 3.5631538708123083 | 0.0052545383095276 | 4.881036613766535 | 0.04774272770527761 | 17.366603125537793 | **MEASURED** |
| 61 | 16.490874755734055 | 0.8209037679944203 | 0.01704810366362142 | 0.9108818181867759 | 0.027675075171023196 | 10.538219555088416 | **MEASURED** |
| 153 | 2.4572441518559858 | 0.3160409788135334 | 0.018457310532168113 | 0.31233264994292786 | 0.03425765627050702 | 6.13913154756097 | **MEASURED** |
| 305 | 0.43233615820632054 | 0.29590823510701564 | 0.021688786309741786 | 0.2235756892072057 | 0.06393059117419594 | 4.213051793309759 | **MEASURED** |
| 763 | 0.060253284132475486 | 0.42656153617737463 | 0.01967095423000681 | 0.12688991611937916 | 0.19119487444544706 | 2.910567118127918 | **MEASURED** |
| 1526 | 0.02298266769621421 | 0.35228002015156656 | 0.018872888134703898 | 0.11421332384774996 | 0.419747262683154 | 2.066653951751226 | **MEASURED** |
| 3052 | 0.015344048212154827 | 0.4595499107430324 | 0.019440740722556856 | 0.18380350452107147 | 0.7062101443702926 | 2.9795749676673937 | **MEASURED** |
| 4578 | 0.015767613208961386 | 0.5082573113267944 | 0.024574157359865772 | 0.24068523075468984 | 0.6015410920209205 | 3.558488252437332 | **MEASURED** |
| 6104 | 0.016467160202990005 | 0.6584555395717259 | 0.030617703609270873 | 0.3553282596782511 | 0.522832764614011 | 3.9631347788971394 | **MEASURED** |
| 7630 | 0.030232045836065136 | 0.9592068563649059 | 0.044463133646784565 | 0.5427710855233483 | 0.47101060402527123 | 5.020768145757741 | **MEASURED** |
| 9375 | 0.03250318241691776 | 1.2078980133022608 | 0.0592233945185601 | 0.7639929133496728 | 0.47457653612243983 | 5.641603135007765 | **MEASURED** |

(Other `STEPS` rows may be appended after fetch; do not invent. Families from full `media/p4/stage-a/a-adamw-*.json` — summaries may omit families.)

### P4(b) — Rescale / train measure (**MEASURED** — `P4-invar-dL` **MISS**)

Baseline + after-arm from dry-run (`internal/p4-b-score-dryrun.md`). `P4-invar-dL` **MISS** — |ΔL|=2.73853462934494; λ ratio=0.4658784038724411; λ 5.967310256347902→2.780040977639008; probe_loss 2.9474442899227142→5.685978919267654.

| Job | Arm | Label | \(\lambda_{\max}\) | probe_loss | \(\lvert\Delta L_{\mathrm{val}}\rvert\) | \(\lambda\) ratio vs ref | Notes |
|-----|-----|-------|--------------------|------------|----------------------------------------|--------------------------|-------|
| `b-rescale-reference` | reference @ 9375 | **MEASURED** | 5.967310256347902 | 2.9474442899227142 | — | — (baseline) | `scale={}`; identity baseline; not a scored invar arm |
| `b-rescale-head-x0.25` | hand rescale (after-arm) | **MEASURED** | 2.780040977639008 | 5.685978919267654 | 2.73853462934494 | 0.4658784038724411 | `P4-invar-dL` **MISS**; /cursor/stores/bc-01947f99-cdd8-4c04-97a8-01a1028bf8af/media/p4/b-rescale-head-x0.25.json |
| `b-rescale-head-x2` | hand rescale (optional) | **MEASURED** | 7.302746614488089 | 3.9412150979042053 | — | — | `scale={'lm_head.weight': 2.0}`; optional; no rescore (`P4-invar-dL` already **MISS** from x0.25); `media/p4/b-rescale-head-x2.json` |
| | train arm (TBD) | **PENDING** | | | | | |

### P4(c) — SGDM top subspace (**MEASURED**)

| Job | transform | Label | end train/val | mean \(\|P_k g\|/\|g\|\) | Notes |
|-----|-----------|-------|---------------|--------------------------|-------|
| `c-full` | full | **MEASURED** | 3.582527816295624 | 0.09296940652964016 | mean frac **MISS** |
| `c-top` | top | **MEASURED** | 5.502330541610718 | 0.16483314082647363 |  |
| `c-removed` | removed | **MEASURED** | 3.5821831226348877 | 0.027522441710655886 | rank **MISS** gap=-0.000344693660736084 |

### Plots

| Figure | Description | Path | Status |
|--------|-------------|------|--------|
| P4-a early \(\lambda_{\max}\) | Step vs \(\lambda_{\max}\) at all MEASURED STEPS 0…9375 (sorted) | `media/p4/p4_lambda_max_early.png` | **MEASURED** (13/13) |
| P4-a family \(\lambda_{\max}\) early | Step vs family \(\lambda_{\max}\) (embedding/attention/mlp/head) at all MEASURED STEPS 0…9375 | `media/p4/p4_family_lambda_early.png` | **MEASURED** (13/13) |
| P4-a density | Eigenvalue density | | **PENDING** |
| P4-c subspace | Loss / grad frac / \(\lambda_{\max}\) | | **PENDING** |

---

## P5 Optional

| Field | Value |
|-------|-------|
| Question | Post-separation cooldown: same-valley walls vs true modes (design `docs/a3-p5-exploratory-design.md`) |
| Code diff path | `experiments/a3_optimization/p5_cooldown_barrier.py` |
| Resolving command | `uv run python -m experiments.a3_optimization.p5_cooldown_barrier measure` |
| Outcome | **MEASURED** — B_R=0.6237, B_C=0.7697049379348755, B_K=0.7320129871368408, G_C=0.8391239460323332, dB=-0.14600493793487546. Scores: `P5-B-cool-x0` **MISS** (B_C=0.7697049379348755); `P5-dB-cool` **MISS** (dB=-0.14600493793487546 (B_R=0.6237 [JSON] − B_C=0.7697049379348755)); `P5-B-const-ctrl` **HIT** (B_K=0.7320129871368408); `P5-rank-arms` **MISS** (B_C=0.7697049379348755 B_R=0.6237 B_K=0.7320129871368408); `P5-G-cool-fitC` **HIT** (G_C=0.8391239460323332 B_C=0.7697049379348755 |G-(B+0.09)|=0.020580991902542256); `P5-emergent-basin` **MISS** (B_C=0.7697049379348755 B_K=0.7320129871368408 → not-same-valley (JSON emergent='true-modes')); `P5-L-cool-dir` **INSUFFICIENT** (dL / (L_R, L_C) missing). Source: `media/p5/` / `internal/p5-score-dryrun.md` |

Canonical Batch 2 9/9 table (finals + P3 sharpness) lives under **P2 River-Valley → Batch 2 jobs** above. CSV: `media/p2/batch2_fi