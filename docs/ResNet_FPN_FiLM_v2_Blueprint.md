# ResNet-FPN-FiLM v2 Blueprint

This document summarizes the planned upgrades for the ResNet-FPN-FiLM architecture (v2). The goal is to evolve the model in lock-step with the HR-FiLM-Net v2 design.

## A. Goals for the "Path-oracle" v2

| KPI | v1 symptom | Target v2 |
|-----|------------|-----------|
| Soft-Dice (val) (on thin corridor) | 0.88–0.90 | ≥ 0.93 |
| Corridor sharpness | Wide ribbon, side branches | Single ~7 px corridor, no extra blobs |
| False-positive rate (background > 0.3) | ~18 % of map | ≤ 8 % |
| Latency (T4) | 22 ms | ≤ 25 ms |

## B. Architecture upgrades

| Block | Change | Effect | Extra cost |
|-------|--------|--------|------------|
| FiLM depth | Add FiLM-3 on P3 (50²) map. | Robot params modulate medium-scale context where branches split. | +0.1 M params |
| Attention gating in FPN fusion | After each lateral 1×1, multiply features by SE-like gate derived from P4 global context. | Suppresses irrelevant open-room activations. | +0.2 M |
| Head sharpening | Replace DW 3×3 with DW 3×3 + Point-wise 1×1 + ReLU (depthwise-separable conv). | Adds non-linearity to refine edges. | +0.05 M |
| Dropout | Keep 0.1 in FPN lateral & head; enable MC-Dropout at inference. | Provides epistemic uncertainty for ensemble fuse. | none |
| CoordConv injection | Prepend a 2-channel (x,y) coordinate grid to the original 4-chan input via 1×1 conv. | Helps locate absolute positions; improves skeleton alignment. | negligible |

Total params v2 ≈ 7 M; latency +2–3 ms on T4 (still < 25 ms).

## C. Loss function & hyper-parameters

| Item | v1 | v2 setting | Rationale |
|------|----|-----------|-----------|
| Loss mix | Dice 0.6 / Focal 0.4, γ=2 | Dice 0.5 + Focal 0.4 (γ=3) + EdgeDice 0.1 | Sharpen positive class and penalise thick borders. |
| Learning rate | 1e-4 | 2e-4 (AdamW) | Slight boost; transformer-like LR not needed. |
| Warm-up | 1 epoch | 1 epoch linear 0→2e-4 | |
| Scheduler | cosine to 1e-5 | cosine to 2e-5 | |
| Weight decay | 0.02 | 0.01 | |
| Grad clip | — | clip_grad_norm_=3.0 | |
| Label smoothing on GT | none | subtract 0.05 then clamp ≥ 0 | discourages saturation at 1.0 |

## D. Augmentation tweaks (path-oracle-centric)

| Aug | Prob | Note |
|-----|------|------|
| Random corridor thinning | 0.3 | Erode GT by 1-px to teach sharper line. |
| Perspective stretch (<5 %) | 0.2 | Introduces skewed geometry; FPN robust. |
| Cutout noise patches | 0.3 | Mask 10×10 random squares in C-space → forces global reasoning. |

Apply in addition to flips, translate, mix-up.

## E. Training schedule

| Phase | Epochs | LR | Comment |
|-------|-------|----|--------|
| Warm-up | 1 | 0→2e-4 | linear |
| Main | 14 | cosine 2e-4 → 2e-5 | checkpoint every 2 |
| Fine-sharpen | 5 | freeze ResNet stem, lr = 1e-5 | focus on head edges |
| Early stop | patience = 4 | monitor val EdgeDice + main Dice | |

Batch = 64 (32 on T4 with grad-ckpt). Mixed precision on.

## F. Calibration & inference

- **MC-Dropout ensemble (default):** run 5 forward passes with dropout on, average logits → produces smoother but thinner corridor.
- **Temperature scaling:** tune T in [1.0, 2.0] on val set to align corridor width to PRM node budget.

