# A6 Practice Quiz Solutions

Answers use four labels. **MEASURED** is a finished run. **FITTED** is a curve inside the points that produced it. **THEORETICAL** is a consequence of the definitions. **PREDICTION** is a configuration that has not been trained. Seed noise for the default recipe is about 0.002. Treating the 1,000 validation chunks as independent, the standard error of their mean NLL is 0.0032 on the five no-dropout models. Homology makes that an underestimate of a new draw. A paired difference on the same chunks, such as the ensemble gain, can be sharper than this marginal error.

Anchors, all measured unless noted: d8, 32 epochs, learning rate 0.003, weight decay 3.2, NLL 1.262714 (`0baf962e`). Five no-dropout seeds span 1.260806–1.262947. d6 at 32 epochs is 1.262783 (`125c21fd`); d9 at 32 epochs and weight decay 1.0 is 1.259360 (`1843c594`); d10 at 16 epochs and weight decay 1.0 is 1.261446 (`5e13ec06`). The parameter fit on those seven selected recipes is \(L = 1.25899 + 0.02591 N^{-0.666}\) for \(N\) in millions, RMSE 0.00093. Dropping d10 moves the offset to 1.2545 and the exponent to 0.381. Probability-ensemble means are 1.261927, 1.257089, 1.255512, 1.254726, 1.254255 for \(K=1\) through \(K=5\). The \(K=1\) to \(K=5\) paired gain is 0.00768, 90% interval 0.00729–0.00808. The descriptive fit \(L(K)=1.25246+0.009465 K^{-1.032}\) is not a measurement past \(K=5\).

---

**Q1. (B). PREDICTION.**

The released ladder is width = 64 × depth, not a depth sweep at width 512. Anchor: d8 1.2627 at 30.4M, d9 1.2594 at 43.3M, d6 1.2628 at 12.8M. Going from d6 to d8 added 17.6M parameters and changed the loss by 0.00007, inside seed noise. The same parameter formula that reproduces the released counts gives depth 12, width 768, vocabulary 19, as 102.7M parameters. The fit there is \(1.25899 + 0.02591 \times 102.7^{-0.666} \approx 1.2602\), which is not below d9. The offset itself moved from 1.2590 to 1.2545 when d10 was left out, so the third digit of an extrapolation is not identified. Depth 12 at learning rate 0.003 stays inside the family that already trained through depth 10, so divergence is not the supported failure mode. The supported outcome is a small move around 1.259–1.265, including a real chance of losing to d9 because the larger d10 recipe already lost while using fewer epochs. Confidence: medium. Assumption: no new learning-rate or weight-decay search. IDs: `125c21fd`, `0baf962e`, `1843c594`, `5e13ec06`.

**Q2. (B). PREDICTION.**

d10's 1.2614 is not a pure parameter effect. It was trained for 16 epochs, while d9 was trained for 32, and the two used the same selected learning rate 0.003. Matched learning rate 0.003 and weight decay 3.2 move only from 1.2648 at 16 epochs to 1.2627 at 32 epochs. The 32-to-64 change at that recipe went the other way, 1.2627 to 1.2691. A 32-epoch d10 at weight decay 1.0 should move toward d9 and can still lose. It does not inherit the leave-one-out offset 1.2545, which is a fit artifact. Confidence: medium. Assumption: weight decay 1.0 remains stable for 32 passes on the larger model. IDs: `5e13ec06`, `1843c594`, `cc5e09349d`, `0baf962e`, `e644f28d`.

**Q3. (A). PREDICTION.**

The four surrounding cells are 1.2794, 1.2632, 1.2621, and 1.2732. In log learning rate and log weight decay, the query sits inside that rectangle, nearer the three good corners than the 1.2794 corner. A log-bilinear reading of those corners is about 1.269. The fitted surface predicts 1.265 at this point. The product 0.003 is a local description of the valley, not a law: Spearman of product versus loss on all 18 positive cells is −0.06, and the quadratic residual at (0.001, 0.3) is +0.024. The supported band is 1.26–1.27. The low-learning-rate, low-decay corner near 1.342 is a different part of the grid. Confidence: medium. IDs: `2ff5e803`, `8cc78664`, `lr_wd_runs`.

**Q4. (A). PREDICTION.**

Along weight decay 0.1 the measured losses are still falling at the edge of the grid: 1.2886 at learning rate 0.003 and 1.2697 at 0.01. The quadratic continues that decline only a short distance. Its minimum is at learning rate 0.015 and weight decay 0.179, predicted loss 1.2634, and at learning rate 0.03 and weight decay 0.1 it predicts 1.2649. That is not a new optimum relative to the measured best cell, 1.2621. The surface RMSE is 0.0086, and the same surface is already 0.007 optimistic at learning rate 0.01 and weight decay 1. The product \(0.03\times 0.1=0.003\) matches the best cells and is not a guarantee: product 0.01 already gives 1.2660 and 1.2732 inside the grid. A result near 1.22 would require the bowl to keep falling, and a result near 1.34 is the opposite corner. This run has not been trained. Confidence: medium on the band, higher on there being no new optimum. IDs: `59b769c9`, `2ff5e803`.

**Q5. (C). PREDICTION.**

At 32 epochs the weight-decay 0.1 run climbs to 1.4057 while weight decay 3.2 holds 1.2627. At 16 epochs, learning rate 0.003, the same comparison is weight decay 0.3 at 1.2794, weight decay 1 at 1.2632, and weight decay 3.2 at 1.2648. Weight decay 0.3 is already 0.016 behind the good 16-epoch cells, and the failure mode at long horizon is too little decay. Doubling the horizon should push 0.3 toward the bad run, not into a tie with 3.2. It should not fully reach 1.41, because 0.3 is still three times 0.1 and the 16-epoch gap from 0.1 to 0.3 at this learning rate was only 0.009. The interval 1.32–1.38 is a judgment, not a fit. Confidence: low-medium. IDs: `bd649eb7`, `0baf962e`, `lr_wd_runs`.

**Q6. (B). THEORETICAL.**

Averaging a probability vector with itself returns the same vector, so the NLL is the member NLL, 1.260806. The measured \(K=5\) gain requires disagreement among seeds 42–46. The fitted offset is not a property of a single checkpoint. Identical copies do not interfere. Anchor: seed 45 loss 1.260806 in `ensemble_bank`. Confidence: high.

**Q7. (C). FITTED, then a PREDICTION about its use.**

The fit's RMSE of \(3\times 10^{-6}\) is the error on five means that were already smooth. Refitting on \(K=1\) through \(K=4\) leaves the exponent and the offset almost unchanged, which is what a three-parameter curve does on a monotone sample of this size. The measured marginal gain from \(K=4\) to \(K=5\) is 0.00047, not 0.001. The paired \(K=1\) to \(K=5\) interval says the five-member gain is real. It does not pin \(K=20\). The offset 1.2525 sits 0.0018 below the last measured point, so calling it a floor overstates the fit. Confidence: high that the four-digit claim is unjustified.

**Q8. (B). PREDICTION.**

The measured gains come from members whose individual losses already sit inside 1.2608–1.2629. Nothing in the \(K^{-1}\) fit says a worse model helps. Averaging probabilities with a member at 1.280 mixes in a worse predictive distribution. Uncorrelated errors are not enough if the new model is also less accurate. The ensemble is not the max of the member losses, so it does not equal 1.280 either. Confidence: high on the direction. A quantitative split would need the sixth model's logits.

**Q9. (B). PREDICTION.**

The constant trajectory that reached 1.2604 used a learning rate nine times smaller. The linear runs that kept learning rate 0.01 out to 128 and 256 epochs finished at 1.2658 and 1.2670, worse than the small constant learning rate. Holding 0.003 fixed for 128 epochs never anneals, so it should be worse than 1.2604 by more than seed noise. It should not fall apart to \(\log 19\), because the same learning rate is stable for 32 epochs with decay and the weights already reach 1.26 under milder schedules. Confidence: medium. IDs: `e195f8d6`, `9c8e654e`, `0baf962e`.

**Q10. (B). PREDICTION.**

Longer exposure is not free performance. The matched 32-to-64 comparison at learning rate 0.003 and weight decay 3.2 went from 1.2627 to 1.2691. The constant trajectory, at a much smaller learning rate, improved only from 1.2627 at 32 epochs to 1.2604 at 128. A linear schedule at 0.003 for 128 epochs spends its step size early and is small at the end, so it should land near the 32-epoch result or a little worse, roughly 1.262–1.272. It should not reproduce the weight-decay 0.1 blow-up, and it should not lose to the learning-rate 0.01 linear run by 0.05. Confidence: medium. IDs: `e644f28d`, `e195f8d6`, `9c8e654e`.

**Q11. (B). PREDICTION.**

The measured batch curve, under square-root scaling, is flat from 16 to 64 (1.2624 versus 1.2627) and then worse at 256 (1.2694), 1024 (1.2959), and 4096 (1.3947). Batch 128 is halfway in log batch size between 64 and 256. The supported interpolation is 1.263–1.268. A new optimum at 128 would contradict the flat-then-worse shape. The 4096 result is a different regime, with 256 updates instead of thousands. Confidence: medium-high. IDs: `8f3947f2`, `0baf962e`, `f45fac8b`.

**Q12. (B). PREDICTION.**

The released batch-4096 run changes three things at once: batch size, learning rate, weight decay, and the number of updates (256 versus 15,008). Putting the learning rate and weight decay back to 0.003 and 3.2 removes two of those. It does not restore the update count. The loss gap of 0.132 is much larger than the learning-rate effects inside the 16-epoch grid, which span about 0.08, so a large share of the damage is the short optimization. The run can still be better or worse than 1.3947. The claim that it returns to 1.2627 is not supported. Confidence: medium on the remaining gap, low on its exact size. ID: `5b60be2e`.

**Q13. (B). MEASURED.**

Batch 256 is the flatter probe, sharpness 0.102 against 0.120, and the worse validation loss, 1.2694 against 1.2624. The probe is a directional finite difference on four training chunks at radius 0.05, not a Hessian eigenvalue. These two runs are enough to reject "lower sharpness implies lower NLL" for this probe. They are not enough to explain the whole batch curve: batch 4096 is both sharper (0.282) and worse (1.395). The same probe on the released batch-64 checkpoint, in fp32, is 0.100, with validation NLL 1.2627. That point is as flat as batch 256 and tied with batch 16, so it does not restore the flat-minima ranking. The fp32 rerun of the batch-16 checkpoint reads 0.1208 against the released 0.1203. IDs: `f45fac8b`, `8f3947f2`, `0baf962e`.

**Q14. (B). PREDICTION.**

On the full 30k pool, at learning rate 0.003 and weight decay 3.2, 16, 32, and 64 passes finish at 1.2648, 1.2627, and 1.2691. Repeated exposures are not a substitute for new chunks. The 30k × 16 run has twice as many unique sequences as the 15k × 32 run. It should win. The two runs are not the 64-epoch setting, and matching token exposures does not force matching losses. Confidence: medium-high on the direction, low on the magnitude, because no 15k run has been trained. IDs: `cc5e09349d`, `0baf962e`, `e644f28d`.

**Q15. (B). PREDICTION.**

The 16-epoch winner is not a 32-epoch recipe. Linear runs selected at learning rate 0.01 for 64, 128, and 256 epochs finished at 1.2683, 1.2658, and 1.2670. Weight decay 0.3 is also light relative to the 32-epoch selection of 3.2, and weight decay 0.1 at 32 epochs is the run that climbs to 1.4057. Combining the high learning rate with the light decay should land worse than 1.2627 by more than 0.002. It should not reproduce the entire 0.14 failure of weight decay 0.1, because 0.3 still decays and the learning rate does anneal. Confidence: medium. IDs: `2ff5e803`, `95dc6bce`, `bd649eb7`, `0baf962e`.

**Q16. (B). PREDICTION.**

The 64-epoch regression is real, 1.2691 versus 1.2627. That run's own logged minimum is 1.2647 at epoch 56, already worse than the 32-epoch endpoint, so stopping inside the longer run does not recover the shorter one. The 16-epoch grid does not say that weight decay 10 is a fix at learning rate 0.003. That cell is 1.2783, already worse than weight decay 3.2 at the same learning rate (1.2648). A stronger decay can change the 64-epoch result. Predicting a return to 1.26 treats the 16-epoch low-learning-rate cells as portable, and those cells were learning rate 0.001, not 0.003. Confidence: medium that a full recovery is the wrong bet. IDs: `e644f28d`, `lr_wd_runs`.

**Q17. (C). THEORETICAL, with the magnitude unmeasured.**

The probability average and the softmax of the logit average differ whenever members disagree. Jensen's inequality does not rank the two NLLs in general. A one-token counterexample with correct-class logits 5 and 0 gives probability-average NLL 0.292 and logit-average NLL 0.079, so the probability average is the worse loss on that token. If all five logit tensors were identical, the two procedures would tie at the member loss. They are not identical: the member losses differ, and the probability average gains 0.0077. The logit average is not defined to be the worst member. No logit-averaged NLL has been computed for this bank, so a claim that it is lower by 0.01 is a precision the files do not support. Confidence: high on the distinction, none on a third digit.

**Q18. (B). PREDICTION.**

The released checkpoint was trained and evaluated with dropout off. Turning attention dropout on at evaluation multiplies attention weights by a random mask the optimizer never saw. That raises the expected NLL of a single forward pass. Dropout during training is a different experiment. One evaluation mask is not the five-seed ensemble. Confidence: high on the direction. The size depends on the mask and is not estimated here.

**Q19. (B). PREDICTION.**

Before any target update the related checkpoint is at 1.2576 and scratch is at 3.3739, a gap of 2.116. One tuned pass closes that to 0.069: related finishes at 1.2291 and scratch at 1.2981 (`a6-ft-related-n30000-e1-lr0.0003-wd3.2-s42`, `a6-ft-scratch-n30000-e1-lr0.003-wd3.2-s42`). The closure is scratch learning the token map. The question starts later, after both have already had 16 passes and related DNA still leads. From 32 to 64 scratch passes the loss went from 1.2627 to 1.2691. From 128 to 256 passes the constant trajectory went from 1.2604 to 1.2608. Extra passes over a fixed 30k pool stop helping. A model that is already ahead after 16 target passes has less room to gain from the next 48 than a story of compounding pretraining requires. The gap should shrink or stall. A reversal larger than 0.05, and divergence of both runs, are stronger claims than the epoch sweep supports. Confidence: medium. Assumption: by 16 passes the related model is on the flat part of the target-data curve. The measured 16-pass comparison does not have related DNA in the lead: it finishes at 1.264776 and the supplied scratch run at the same learning rate finishes at 1.263681. A second data order repeats that order, related 1.265161 and scratch 1.263249, with DCLM still ahead at 1.261844. This question conditions on a lead that did not occur, and then asks about 64 passes. That extension is not measured.

**Q20. (A). PREDICTION.**

At learning rate 0.001, weight decay 3.2, and 16 passes, full fine-tuning of related DNA finishes at 1.264776. Freezing the body, so only the embedding and the output head train, finishes at 1.243429 (`a6-ft-related-n30000-e16-lr0.001-wd3.2-frozen_body-s42`). That is 0.021 better, and the endpoint is the minimum. Freezing the output head instead finishes at 1.265271, a tie with full fine-tuning. The body was already useful at step 0, loss 1.2576. Updating it for sixteen passes made the endpoint worse than leaving it fixed. Freezing the embedding alone was not run, and the learning rate was not searched again for each freeze. The measured pair puts the frozen body ahead of full fine-tuning, which is why the old ranking is rejected. Where the frozen embedding sits between them is still a prediction. Confidence: high on frozen body versus full fine-tuning at this learning rate, low on the frozen embedding.

**Q21. (B). PREDICTION.**

The six-pass loop stores 11,426,688 parameters, exactly a dense depth-3 model at width 512. At learning rate 0.003, the learning rate this question uses, it finished at 1.264782, which is 0.0021 above dense depth 8 and inside the dense seed span (`a6-loop-lr0.003-s42`). The selected learning rate, 0.001, finished at 1.264715 on seed 42 and 1.263934 on seed 43, also above dense depth 8. Two core applications keep that same prelude, core, and coda, so the stored count does not change. Executed depth falls from 8 to 4. The six-pass loop is charged the dense depth-8 cost, \(2.289\times 10^{17}\). Two passes are four of those eight blocks, about \(1.14\times 10^{17}\), which is more than the dense depth-3 cost of \(8.587\times 10^{16}\) and less than depth 8. Parameter sharing is the capacity constraint. Removing applications does not relax it, and it reduces compute. The six-pass model already failed to beat dense depth 8, so there is no measured benefit for the two-pass model to keep, and it is not itself a dense depth-8 model. The loss gap may be small. Confidence: medium-high on the direction. The two-pass loss is not measured.

**Q22. (B). PREDICTION.**

d6 and d8 differ by 17.6M stored parameters and by 0.00007 nats, inside seed noise, at the same 30k pool. The four-expert model at this same learning rate, weight decay, and epoch count finished at 1.268434 on seed 42 and 1.265620 on seed 43 (`a6-moe-lr0.003-s42`, `a6-moe-lr0.003-s43`). The gaps versus dense depth 8 are 0.0057 and 0.0029. It stores 96,514,560 parameters and activates 30,454,272. An eight-expert model increases the stored pool again and cuts the tokens each expert is trained on. Both four-expert finishes went the wrong way for the "more stored experts help" story, by a margin larger than the dense seed span. The recipe was selected on seed 42. Seed 43 is the replication, not a second search. Splitting the same tokens across eight experts removes data per expert on top of that. Confidence: medium. The eight-expert loss is not measured.

**Q23. (B). PREDICTION.**

Weight decay 0.1 fails because weights can grow while the same chunks are repeated. Hyperball replaces that penalty with a projection onto the initial Frobenius sphere of each linear weight. The tuned 32-epoch run, learning rate 0.03, finished at 1.271385 on seed 42 and 1.271424 on seed 43 (`a6-hyperball-lr0.03-s42`, `a6-hyperball-lr0.03-s43`). Both are best at epoch 28, at 1.2658 and 1.266788, and then rise to those endpoints while the learning rate is already near zero. The seeds differ by 0.00004. Resuming from that checkpoint does not turn the sphere constraint off, and it also does not restore a large learning rate, so the run is not the weight-decay 0.1 trajectory and does not have to end at 1.41. The large late rises in this experiment were at learning rates 0.003 and 0.01, during the scheduled training, not after the schedule had decayed. The continuation also does not inherit the constant-learning-rate curve, which is an AdamW trajectory. A further drop to 1.20 is not supported: the best movement past 32 epochs on the constant trajectory is about 0.002, and this question resumes a run whose step size has already vanished. Confidence: medium on the mechanism, low on the exact 128-epoch number. That continuation is not measured.

**Q24. (B). PREDICTION.**

The batch-4096 AdamW loss is 1.3947 against 1.2627 at batch 64, a gap of 0.132 after 256 updates. SAM adds a perturbation and a second gradient at the same small update count. The batch-16 versus batch-256 pair already shows that a flatter probe can have the worse loss. SAM can move the 1.3947 number. Closing a 0.13 gap, or matching batch 16, would mean the second gradient replaced thousands of missing updates. That is not the supported reading. Confidence: medium-high on the surviving gap, low on SAM's sign at this batch. ID: `5b60be2e`.

**Q25. (B). PREDICTION.**

The full-pool 16-epoch and 32-epoch results, 1.2632 and 1.2627, show that adding passes on 30k chunks barely moves the loss. The 15k run changes the sample and the exposure together. It is not the 8-epoch full-pool run: update count depends on \(\lceil n/64\rceil\times\) passes, so 15k for 32 passes has about half the updates of 30k for 32 passes and about the same updates as 30k for 16 passes, but only half the unique chunks. Matching updates with the 16-epoch full-pool run is not the same experiment. A smaller pool is not an easier generalization problem. The 15k run should be worse than 1.2627 by more than 0.002. Confidence: medium-high on the direction. No subset run has been trained, so there is no magnitude to report.

---

Unmeasured configurations: Q1–Q5, Q8–Q12, Q14–Q16, Q18, Q19, Q21–Q25. That is 21 questions. Q20 includes a measured pair, frozen body versus full fine-tuning at learning rate 0.001, and an unmeasured arm, freezing the embedding. Q6 and Q17 are consequences of the ensemble definition applied to a checkpoint that was not copied or logit-averaged. Q7 refuses an extrapolation of a fitted curve. Q13 is the one measured sharpness pair.
