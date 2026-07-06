# Support-Vector Networks Reproduction

This project reproduces the methodology of Cortes and Vapnik, "Support-Vector
Networks" (1995). The implementation trains SVMs from the dual without using
`sklearn.svm.SVC` as the primary estimator.

## Contents

```text
svm_reproduction/
├── data/
├── notebooks/
├── src/
│   ├── kernels.py
│   ├── optimizer.py
│   ├── svm.py
│   ├── metrics.py
│   ├── visualization.py
│   └── utils.py
├── experiments/
│   ├── synthetic.py
│   ├── usps.py
│   └── mnist.py
├── results/
├── figures/
├── requirements.txt
└── README.md
```

The mathematical summary is in `../paper_analysis.md`; the reproduction report is
in `../reproduction_report.md`.

## Setup

```bash
git clone https://github.com/ryan.martin060708-code/support-vector-networks-reproduction.git
cd svm_reproduction
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

On Linux/macOS, use `source .venv/bin/activate` instead of the Windows activate
script.

## Core Implementation

- `src/kernels.py`: linear, polynomial, and Gaussian/RBF kernels.
- `src/optimizer.py`: dual objective, KKT diagnostics, exact QP wrapper, SMO,
  and a paper-style chunking optimizer.
- `src/svm.py`: binary hard-margin and soft-margin SVMs plus one-vs-rest
  multiclass classification.
- `src/utils.py`: USPS/MNIST loading, OCR preprocessing, and reproducibility
  helpers.

## Run Synthetic Figures

```bash
python experiments/synthetic.py
```

Outputs:

- `figures/synthetic/*_degree2.png`
- `figures/figure2_margin_example.png`
- `figures/figure3_feature_space_network.png`
- `figures/figure4_kernel_network.png`
- `figures/figure9_benchmark_recreation.png`
- `results/synthetic_results.json`

## Run USPS Experiment

Quick smoke run:

```bash
python experiments/usps.py --download --degrees 1 2 --max-train-per-class 50 --max-test-per-class 50
```

Full Table 2 attempt:

```bash
python experiments/usps.py --download --degrees 1 2 3 4 5 6 7
```

The paper does not report `C`; the script defaults to `C=10.0`. Use `--C` to
tune this value. The USPS preprocessing pipeline applies centering, deslanting,
and Gaussian smoothing with `sigma=0.75`.

## Run MNIST/NIST Experiment

Quick smoke run:

```bash
python experiments/mnist.py --download --max-train-per-class 50 --max-test-per-class 50
```

Full MNIST substitute for the NIST experiment:

```bash
python experiments/mnist.py --download --no-store-kernel
```

The original NIST split is not bundled in modern public mirrors, so this script
uses MNIST as the closest readily available 60,000/10,000 handwritten digit
substitute. The MNIST script defaults to `--gamma 0.01` for numerical stability
with normalized `[0, 1]` pixels; pass `--gamma 1.0` for the literal paper kernel
`(u^T v + 1)^4`.

## Reproducibility Notes

- All random generators are seeded from command-line flags.
- The SMO solver is implemented from scratch and optimizes the standard dual.
- The exact QP solver is available for small problems if SciPy is installed.
- Large polynomial SVMs are expensive; the historical USPS/NIST experiments can
  take substantial CPU time in pure Python.
- `sklearn.svm.SVC` is intentionally not used.
