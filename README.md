# Neural Emulation of Mass Flow Runout

This repository provides code, models, and tools for learning neural emulators of mass flow runout processes from physics-based simulations.

The current implementation focuses on predicting:
- runout extent (binary mask)
- final deposit thickness (h)

Although initially developed for landslide-like processes, the framework is designed to generalize to a broader class of mass flows.

---

## Overview

Numerical simulation of mass flows (e.g. landslides, debris flows, volcanic flows) is computationally expensive.

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
