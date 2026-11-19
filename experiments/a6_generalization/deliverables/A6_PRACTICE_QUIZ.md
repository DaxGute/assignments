# A6 Practice Quiz

Twenty-five questions. Each stem names a measured anchor, then changes the experiment. Unless an option says the run finished, the right answer is a **prediction**. Seed noise for the default dense recipe is about 0.002: five seeds at learning rate 0.003 and weight decay 3.2 span 1.2608–1.2629. A gap smaller than that is a tie.

Constants, unless a question changes them: 30,000 DNA chunks, context 1,024, dense depth 8, width 512, 30.44M parameters, AdamW, linear decay, batch 64, 32 epochs, learning rate 0.003, weight decay 3.2, seed 42. That run finishes at validation NLL **1.2627**. One epoch is one pass over the same 30,000 chunks, 30.72M tokens, not 30.72M new chunks. The seven-point parameter fit, \(N\) in millions, is \(L = 1.2590 + 0.02591 N^{-0.666}\), and it is only a description of those seven selected recipes.

---

**Q1.** The selected ladder uses width = 64 × depth. d9, width 576, 43.3M parameters, 32 epochs, weight decay 1.0, finished at 1.2594. Continue that ladder to depth 12, width 768, 102.7M parameters, for 32 epochs at learning rate 0.003 and weight decay 3.2. No new search. Closest outcome?

- (A) Near 1.20, because the fit keeps falling like a power law
- (B) Near 1.259–1.265, a small move relative to d9, with a real chance of doing worse
- (C) Near 1.10, because parameters roughly triple from d8 and the fit offset is not binding
- (D) Diverges, because learning rate 0.003 cannot train depth 12

**Q2.** Selected d10, 59.4M parameters, finished at 1.2614 after 16 epochs and weight decay 1.0. Repeat it at 32 epochs, same learning rate 0.003 and weight decay 1.0. Versus the d9 result of 1.2594?

- (A) d10 improves by more than 0.02 and clearly wins
- (B) d10 moves closer to d9 and the two are within about 0.005, or d10 still loses
- (C) d10 gets worse than 1.27, because 32 epochs overfit every large model
- (D) d10 matches the fit offset 1.2545 obtained by dropping d10 from the original fit

**Q3.** On the 16-epoch grid, the best cell is learning rate 0.01, weight decay 0.3, loss 1.2621. The corners around learning rate 0.006 and weight decay 0.5 are measured: (0.003, 0.3) is 1.2794, (0.003, 1) is 1.2632, (0.01, 0.3) is 1.2621, and (0.01, 1) is 1.2732. Predict a new 16-epoch run at learning rate 0.006 and weight decay 0.5, product 0.003.

- (A) About 1.26–1.27
- (B) About 1.34, like the low-learning-rate, low-decay corner
- (C) About 1.20, because the product law is exact
- (D) Worse than 1.40

**Q4.** Same 16-epoch setup. Learning rate 0.03, weight decay 0.1. The highest measured learning rate is 0.01. At that learning rate, weight decay 0.1 scores 1.2697 and weight decay 0.3 scores 1.2621. The fitted surface is minimized at learning rate 0.015 and weight decay 0.179, just outside the grid, with predicted loss 1.2634.

- (A) About 1.26–1.28. This point is just past that minimum, so it should not beat the best cell by more than the surface residual
- (B) About 1.34, like the low-learning-rate, low-decay corner
- (C) About 1.22, because 0.03 is only one step past 0.01
- (D) The product 0.003 guarantees a result within 0.002 of 1.262

**Q5.** At 32 epochs, weight decay 0.1 finishes at 1.4057 and weight decay 3.2 finishes at 1.2627, both at learning rate 0.003. Predict weight decay 0.3 at that same learning rate and horizon.

- (A) Near 1.26, tied with weight decay 3.2
- (B) Near 1.41, tied with weight decay 0.1
- (C) Between them and closer to the bad run: roughly 1.32–1.38
- (D) Below 1.25, because 0.3 was optimal at 16 epochs

**Q6.** Take the best single no-dropout model, seed 45, validation NLL 1.2608. Build a five-member ensemble by copying that checkpoint five times and averaging probabilities.

- (A) About 1.254, the measured \(K=5\) loss
- (B) Exactly the member loss, 1.2608
- (C) About 1.252, the fitted offset
- (D) Worse than 1.27, because identical copies interfere

**Q7.** The fit \(L(K)=1.2525+0.00947 K^{-1.03}\) matches \(K=1\) through \(K=5\) with RMSE about \(3\times 10^{-6}\). What is justified at \(K=20\)?

- (A) The loss will be 1.2529, to four digits
- (B) The loss will keep falling by about 0.001 per extra member, as it did from \(K=4\) to \(K=5\)
- (C) A further drop is plausible and small; 1.2525 is not a measured floor, and the fit cannot certify the third digit
- (D) The ensemble will return to the \(K=1\) loss, because five members already saturated the seed noise

**Q8.** The five supplied members average to 1.2543 at \(K=5\). Add a sixth model whose own validation NLL is 1.280, with errors that are not copies of the other five. What happens to the six-model probability average relative to 1.2543?

- (A) It must improve, because \(K\) increased
- (B) It can get worse. A weak member pulls the average even if its errors are not identical
- (C) It lands on the fitted offset 1.2525
- (D) It equals 1.280, because the worst member dominates an average

**Q9.** The supplied constant-LR trajectory uses learning rate \(3.33\times 10^{-4}\) and reaches 1.2604 at 128 epochs. Run a different constant schedule: learning rate 0.003, weight decay 3.2, for 128 epochs. Versus 1.2604?

- (A) Clearly better, by more than 0.02, because 0.003 is the 32-epoch default
- (B) Worse, likely by well more than seed noise. A constant learning rate nine times larger keeps moving after the data have been memorized
- (C) Tied within 0.002, because constant schedules converge to the same loss
- (D) Diverges to chance, about \(\log 19 \approx 2.94\)

**Q10.** The supplied 128-epoch linear run, learning rate 0.01, finished at 1.2658. Replace only the learning rate with 0.003, still linear decay, 128 epochs, weight decay 3.2. The 32-epoch linear run at learning rate 0.003 finished at 1.2627, and the 64-epoch run at learning rate 0.003 and weight decay 3.2 finished at 1.2691.

- (A) Near 1.255 or better, because a longer schedule with the good short-run learning rate must win
- (B) Near the 32-epoch result or a bit worse, roughly 1.262–1.272, not a large gain over 1.260–1.266
- (C) Near 1.41, the 32-epoch weight-decay 0.1 failure
- (D) Worse than the learning-rate 0.01 linear run by more than 0.05

**Q11.** Batch 16 with square-root scaling finished at 1.2624. Batch 64 finished at 1.2627. Batch 256 finished at 1.2694. Predict batch 128 under the same square-root rule, 32 epochs.

- (A) About 1.24, because noise is monotone and 128 is the sweet spot
- (B) Between the batch-64 and batch-256 results, roughly 1.263–1.268
- (C) About 1.39, like batch 4096
- (D) Worse than batch 16 by more than 0.03

**Q12.** Batch 4096 with scaled learning rate 0.024 and weight decay 25.6 finished at 1.3947, after only 256 updates. Repeat batch 4096 for 32 epochs but keep learning rate 0.003 and weight decay 3.2. What is the main change in the comparison?

- (A) The run should match batch 64, because the learning rate is now the good one and batch size does not matter
- (B) It removes the scaled-step confound but leaves a 256-update optimization problem, so a large gap versus 1.2627 can remain
- (C) It should beat 1.2627, because the unscaled learning rate is smaller and therefore flatter
- (D) It diverges, because weight decay 3.2 is illegal at batch 4096

**Q13.** Batch 256 has directional sharpness 0.102 and validation NLL 1.2694. Batch 16 has sharpness 0.120 and NLL 1.2624. Which claim do these two runs support?

- (A) Lower sharpness implies lower validation NLL on this probe
- (B) The flat-minima story fails for this pair: the sharper probe has the better loss
- (C) Sharpness and loss are the same measurement
- (D) Batch 256 is the generalization optimum because it is flatter

**Q14.** Compare 15,000 chunks for 32 passes with 30,000 chunks for 16 passes. Token exposures match at about 492M. Everything else stays at the default recipe.

- (A) The two runs must match, because exposures match
- (B) The 30,000-chunk run should be better. Unique sequences, not repeated exposures, are what moved loss in the epoch sweep
- (C) The 15,000-chunk run should be better, because 32 passes is the default horizon
- (D) Both should land on the 64-epoch loss 1.269, because each sees “too many epochs”

**Q15.** The 16-epoch optimum was learning rate 0.01 and weight decay 0.3, loss 1.2621. Train that same pair for 32 epochs, with no retune. Versus the 32-epoch default, learning rate 0.003 and weight decay 3.2, loss 1.2627?

- (A) Better by about 0.02, because it won at 16 epochs
- (B) Worse than 1.2627, by more than seed noise. The long-horizon runs that kept learning rate 0.01 were already worse, and weight decay 0.3 is light once the pool is repeated for 32 passes
- (C) Tied within 0.002, because every tuned recipe reaches 1.263
- (D) As bad as weight decay 0.1 at 32 epochs, about 1.41

**Q16.** The measured 64-epoch run at learning rate 0.003 and weight decay 3.2 finished at 1.2691, worse than the 32-epoch result 1.2627. Repeat 64 epochs at weight decay 10, same learning rate.

- (A) Weight decay 10 restores the loss to about 1.26 or better, because the 16-epoch grid liked strong decay at learning rate 0.001
- (B) It can move the loss, but the 16-epoch grid already punished weight decay 10 at learning rate 0.003 (loss 1.278), so a full recovery below 1.263 is not the conservative prediction
- (C) It must match 1.2691, because epoch count fixes the loss once the learning rate is fixed
- (D) It diverges, because weight decay 10 is above the largest stable value

**Q17.** At \(K=5\) the probability average of the five supplied models reaches 1.2543. Average logits instead, then take the softmax and the NLL, on the same predictions.

- (A) The logit average is lower by about 0.01
- (B) The two averages are algebraically identical
- (C) The two procedures differ when members disagree, and neither loss is guaranteed to be lower. Copies tie. No logit-averaged NLL has been computed for this bank
- (D) The logit average equals the worst member

**Q18.** The released no-dropout d8 checkpoint is evaluated with attention dropout 0.15 left on. Training did not use dropout.

- (A) Validation NLL falls, because dropout is regularizing the evaluation
- (B) Validation NLL rises. Dropout at evaluation adds noise the weights were not trained to cancel
- (C) No change, because dropout is disabled by the loss function
- (D) The loss goes to the \(K=5\) ensemble value, because one dropout mask is an ensemble

**Q19.** Suppose a related-DNA model has already been fine-tuned for 16 passes and still leads a scratch model trained for those same 16 passes. Extend both to 64 passes on the same 30k pool, retuning learning rate only within a small grid. What happens to the gap?

- (A) It keeps growing, because pretraining compounds with every epoch
- (B) It shrinks or stalls. Extra passes repeat the same target chunks, and the epoch sweep shows that going from 32 to 64 passes made the scratch recipe worse, not better
- (C) The scratch model jumps ahead by more than 0.05, because 64 passes erase initialization
- (D) Both diverge, because 64 passes exceed the stable horizon for every initialization

**Q20.** Related-DNA initialization, 30k chunks, 16 passes. Three adaptations, learning rate tuned separately: full fine-tuning; freeze only the token embedding; freeze the transformer body and train the embedding and head. Rank the likely validation losses, best first.

- (A) Frozen body, then frozen embedding, then full fine-tuning
- (B) Full fine-tuning, then frozen embedding, then frozen body
- (C) Frozen embedding best, because the embedding is the only DNA-specific tensor
- (D) All three tie within seed noise, because the checkpoint is already trained

**Q21.** The six-pass loop stores one prelude, one shared core, and one coda, and applies the core six times. Train the same block with two core applications instead of six, learning rate 0.003, 32 epochs, weight decay 3.2. Stored parameter count is unchanged.

- (A) The two-pass model matches the six-pass model, because stored parameters match
- (B) The two-pass model is worse. It throws away active depth and keeps the same capacity constraint
- (C) The two-pass model is better, because fewer applications reduce overfitting
- (D) The two-pass model matches dense d8, because the prelude and coda are dense blocks

**Q22.** The four-expert model stores about 96.5M parameters and uses one expert per token. Train an eight-expert top-1 model at the same width, so each expert sees about half as many tokens as in the four-expert model, for 32 epochs at learning rate 0.003 and weight decay 3.2.

- (A) It should improve on the four-expert model by about the d8-to-d9 gap, because stored parameters increase again
- (B) It is more likely to lose or tie. The d6 and d8 dense runs already show that extra stored parameters, at this data size, bought about 0.0001 nats, while splitting tokens across more experts reduces per-expert data
- (C) It matches dense d8 exactly, because active MLP width is unchanged
- (D) It reaches the fitted offset 1.259, because the parameter law depends only on stored count

**Q23.** Hyperball keeps each linear weight on its initial Frobenius sphere and uses no weight decay toward zero. The 32-epoch AdamW gap is 1.4057 at weight decay 0.1 versus 1.2627 at weight decay 3.2. Predict a Hyperball run that is continued from a tuned 32-epoch checkpoint out to 128 epochs, without putting weight decay back.

- (A) It collapses to about 1.41, because any run without weight decay eventually follows the weight-decay 0.1 curve
- (B) It need not follow that curve. The sphere constraint is still on, so the 128-epoch outcome is a small drift around the 32-epoch Hyperball loss, not a return to unconstrained growth
- (C) It reaches 1.20, because extra epochs and a norm constraint stack
- (D) It matches the constant learning-rate trajectory at \(3.33\times 10^{-4}\) exactly

**Q24.** SAM at batch 64 is a different optimizer from the supplied batch sweep. Predict SAM with perturbation radius 0.05 at batch 4096, using that batch's scaled AdamW learning rate 0.024 and weight decay 25.6, for 32 epochs. The AdamW run at those settings finished at 1.3947.

- (A) SAM removes the large-batch penalty and returns the loss near 1.263
- (B) SAM can change sharpness and still leave most of the 0.13 gap. The batch-4096 run took 256 updates; a flatter step does not restore the missing optimization
- (C) SAM must be worse than 1.50, because large batches and sharpness penalties compound
- (D) SAM matches batch 16, because the sharpness probe is the only cause of the batch effect

**Q25.** Default recipe, but the training pool is a random 15,000-chunk subset, still for 32 passes. Exposure is half of the default run, and unique data is half. The 32-epoch full-pool result is 1.2627. The 16-epoch full-pool result at learning rate 0.003 and weight decay 1.0 is 1.2632. Predict the 15k, 32-pass run.

- (A) It matches 1.2627, because 32 passes are what matter
- (B) It is worse than the full-pool 32-pass run by more than seed noise. Halving unique chunks is not the same intervention as cutting passes on the full pool
- (C) It matches the 8-epoch full-pool loss exactly, because both have the same number of updates
- (D) It is better than 1.2627, because a smaller pool is easier to optimize
