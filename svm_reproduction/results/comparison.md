# Results Comparison

This file records the paper's reported results and the fields populated by the
reproduction scripts. Full USPS and MNIST/NIST runs require downloading the
datasets and can be computationally expensive in a pure-Python SMO
implementation.

## Synthetic 2D Experiments

The paper provides qualitative Figure 5 examples but does not report exact data
coordinates or numeric error rates. The reproduction creates analogous 2D
datasets and stores numeric results in `results/synthetic_results.json`.

| Experiment | Original paper result | Reproduced result | Absolute difference | Percentage difference |
| --- | --- | --- | --- | --- |
| Figure 5 quadratic-style split | Qualitative plot only | 0.0% training error, 6 support vectors | Not defined | Not defined |
| Figure 5 XOR-style split | Qualitative plot only | 13.3333% training error, 25 support vectors | Not defined | Not defined |
| Figure 5 ellipse-style split | Qualitative plot only | 0.0% training error, 6 support vectors | Not defined | Not defined |
| Figure 5 overlap-style split | Qualitative plot only | 6.6667% training error, 15 support vectors | Not defined | Not defined |

## USPS Table 2

The table below currently includes the 20-examples-per-class smoke run executed
in this environment. It verifies the end-to-end pipeline but is not the full
paper-scale result; run `experiments/usps.py --download --degrees 1 2 3 4 5 6 7`
for the full comparison.

| Degree | Original raw error % | Reproduced raw error % | Absolute difference | Percentage difference | Original mean support vectors | Reproduced mean support vectors |
| ---: | ---: | --- | --- | --- | ---: | --- |
| 1 | 12.0 | 20.0 smoke | 8.0 | 66.6667% | 200 | 23.2 smoke |
| 2 | 4.7 | 18.5 smoke | 13.8 | 293.617% | 127 | 28.8 smoke |
| 3 | 4.4 | Pending full run | Pending | Pending | 148 | Pending full run |
| 4 | 4.3 | Pending full run | Pending | Pending | 165 | Pending full run |
| 5 | 4.3 | Pending full run | Pending | Pending | 175 | Pending full run |
| 6 | 4.2 | Pending full run | Pending | Pending | 185 | Pending full run |
| 7 | 4.3 | Pending full run | Pending | Pending | 190 | Pending full run |

After running `python experiments/usps.py --download`, the script writes
`results/usps_results.md`, `results/usps_results.csv`, and
`results/usps_results.json` with reproduced raw error, training accuracy, test
accuracy, and support-vector counts.

## NIST / MNIST Degree-4 Experiment

The reproduced value below is a MNIST smoke run with 20 examples per class and
`gamma=0.01`. It tests the full one-vs-rest degree-4 path but is intentionally
not comparable to the full 60,000/10,000 NIST benchmark.

| Metric | Original paper result | Reproduced result | Absolute difference | Percentage difference |
| --- | ---: | --- | --- | --- |
| Combined test error % | 1.1 | 18.0 smoke | 16.9 | 1536.36% |
| Classifier 0 support vectors | 1379 | 51 smoke | 1328 | 96.3017% |
| Classifier 1 support vectors | 989 | 50 smoke | 939 | 94.9444% |
| Classifier 2 support vectors | 1958 | 77 smoke | 1881 | 96.0674% |
| Classifier 3 support vectors | 1900 | 61 smoke | 1839 | 96.7895% |
| Classifier 4 support vectors | 1224 | 73 smoke | 1151 | 94.0359% |
| Classifier 5 support vectors | 2024 | 80 smoke | 1944 | 96.0474% |
| Classifier 6 support vectors | 1527 | 73 smoke | 1454 | 95.2194% |
| Classifier 7 support vectors | 2064 | 78 smoke | 1986 | 96.2209% |
| Classifier 8 support vectors | 2332 | 71 smoke | 2261 | 96.9554% |
| Classifier 9 support vectors | 2765 | 71 smoke | 2694 | 97.4322% |

The reproduction uses MNIST when the original NIST split is unavailable, so
differences combine algorithmic, preprocessing, dataset, and hardware/runtime
effects.
