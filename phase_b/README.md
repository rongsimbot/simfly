# Phase B: LIF Parameter Optimization via PPO

**Date:** 2026-06-14
**Status:** COMPLETE

## Overview
Used Phase A RL framework (PPO, numpy analytic gradients, GAE) to optimize NIRON LIF parameters. Connectome drives ALL movement; RL only tunes per-NT-type leak rates, refractory delays, and weight scaling.

## Key Results
- Neurons: 29,810 (BFS-1 sensorimotor pathway)
- Synapses: 925,476
- Training: 50 PPO iterations, 369.8s
- Baseline (hand-tuned): reward=1011.8, z_height=0.074, upright=1.240
- RL-Optimized: reward=1051.7, z_height=0.077, upright=1.290
- Improvement: +3.9% total reward, +4.0% upright stability

## Optimized LIF Parameters
| NT Type | Leak (base->opt) | Refractory (base->opt) | WScale (base->opt) |
|---------|-----------------|----------------------|-------------------|
| ACH | 0.12->0.25 | 0->0.3 | 1.50->0.26 |
| GABA | 0.20->5.01 | 0->4.7 | -0.50->5.40 |
| GLUT | 0.15->0.56 | 0->0.4 | -0.25->0.48 |
| DA | 0.10->0.25 | 0->0.2 | 0.75->0.26 |
| OCT | 0.10->5.32 | 0->5.0 | 0.75->5.01 |
| SER | 0.18->0.63 | 0->0.4 | -0.15->0.38 |

## Scientific Findings
1. GABA needs high leak (5.0 vs 0.20): Inhibitory neurons benefit from fast temporal decay
2. OCT needs high leak (5.3 vs 0.10): Arousal signals should be transient
3. Refractory delays emerge for GABA/OCT: 4.7-5.0 cycle delays prevent re-firing
4. Excitatory NTs get moderate adjustments: ACH, GLUT, DA, SER show 1.5-3.5x leak increases
