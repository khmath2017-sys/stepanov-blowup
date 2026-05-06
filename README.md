# stepanov-blowup
# Stepanov Blow-Up Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

Companion code for the paper:

> **"Robust Detection of Finite-Time Blow-Up in Nonlinear PDEs Using Weighted Stepanov Spaces: A Theoretical and Numerical Investigation"**
>
> *Mokhammad Khaleel Abunavas, Voronezh State University*

## Overview

This repository reproduces the numerical experiments from the paper, demonstrating that the weighted Stepanov norm \(S_w^2\) (with a Gaussian weight) is dramatically more robust to noise than classical \(L^2\) and \(H^1\) norms in detecting blow-up.

### Key Results
- **Generic sharp spike test:** Noise sensitivity reduction of **98.3%** vs \(H^1\).
- **NLS blow-up profile test:** Noise sensitivity reduction of **99.96%** vs \(H^1\).

## Quick Start

### Requirements
```bash
pip install numpy matplotlib scipy
