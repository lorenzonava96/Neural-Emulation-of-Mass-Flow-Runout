# Neural Emulation of Mass Flow Runout

This repository provides code, models, and tools for learning neural emulators of mass flow runout processes from physics-based simulations.

The current implementation focuses on predicting:
- runout extent (binary mask)
- final deposit thickness (h)

For further details, please consult the preprint "Neural emulation of gravity-driven geohazard runout" by [Nava et al (2025)](https://arxiv.org/abs/2512.16221).
Model weights can be downloaded from the Zenodo repository [Nava et al (2026)](https://zenodo.org/records/19853721).
The training dataset can be downloaded from the Zenodo repository [Nava et al (2026)](https://zenodo.org/records/19831785).



***Workflow***
![Workflow](https://github.com/lorenzonava96/Neural-Emulation-of-Mass-Flow-Runout/blob/main/figures/scheme.png)

---

## Overview

Numerical simulation of gravitational mass flows (e.g. landslides, debris flows, avalanches, volcanic flows) is computationally expensive.

This project explores the use of deep learning models as fast emulators that approximate the outcome of such simulations, enabling:
- rapid scenario exploration
- ensemble-based uncertainty analysis
- probabilistic hazard mapping

---

## Repository Structure

```text
.
├── src/        # training, preprocessing, inference, neural network architectures
├── weights/         # model weights
├── notebooks/      # visualization and analysis
├── dataset/       # link to training dataset
├── examples/       # examples data for predicting the Maoxian landslide
└── README.md
