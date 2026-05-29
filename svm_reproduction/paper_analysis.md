# Paper Analysis: "Support-Vector Networks" (Cortes and Vapnik, 1995)

This analysis is based on the supplied PDF `BF00994018.pdf` and records the
mathematical and experimental details used in the reproduction code under
`svm_reproduction/`.

## Main Contribution

Cortes and Vapnik introduce the support-vector network for two-group
classification. The method combines three ideas:

1. The optimal separating hyperplane: choose the separating hyperplane with the
   largest margin, not an arbitrary separator.
2. Expansion on support vectors: the solution vector can be written as a linear
   combination of a small subset of training patterns.
3. Kernel convolution of dot products: replace feature-space dot products with
   a valid kernel, enabling very high-dimensional polynomial decision surfaces.

The paper extends earlier optimal-margin work from separable data to
non-separable data by introducing soft margins.

## Training Data and Notation

The training set is

```text
(y_1, x_1), ..., (y_l, x_l),    y_i in {-1, +1}.
```

The separating function in input or feature space is

```text
f(x) = w^T x + b,    prediction = sign(f(x)).
```

For a kernelized classifier, `x` is implicitly replaced by `phi(x)`.

## Margin Maximization

With the normalization used in the paper, separability is expressed as

```text
y_i (w^T x_i + b) >= 1    for i = 1, ..., l.
```

The distance between the two supporting hyperplanes

```text
w^T x + b = +1
w^T x + b = -1
```

is

```text
rho = 2 / ||w||.
```

Maximizing the margin is therefore equivalent to minimizing `||w||^2`.

## Hard-Margin Primal Problem

The separable optimal hyperplane solves

```text
minimize    1/2 ||w||^2
subject to  y_i (w^T x_i + b) >= 1,    i = 1, ..., l.
```

The factor `1/2` is conventional and does not change the minimizer; it makes the
dual derivative cleaner.

## Hard-Margin Dual Derivation

The Lagrangian is

```text
L(w, b, alpha)
  = 1/2 w^T w - sum_i alpha_i [ y_i (w^T x_i + b) - 1 ],
alpha_i >= 0.
```

Stationarity with respect to `w` and `b` gives

```text
dL/dw = 0  =>  w = sum_i alpha_i y_i x_i
dL/db = 0  =>  sum_i alpha_i y_i = 0.
```

Substitution yields the dual objective

```text
maximize    W(alpha)
          = sum_i alpha_i
            - 1/2 sum_i sum_j alpha_i alpha_j y_i y_j (x_i^T x_j)

subject to  alpha_i >= 0
            sum_i alpha_i y_i = 0.
```

In matrix form, with `D_ij = y_i y_j x_i^T x_j`,

```text
W(alpha) = alpha^T 1 - 1/2 alpha^T D alpha.
```

The paper states the relationship between the optimal dual value and the margin:

```text
W(alpha*) = 1/2 ||w*||^2 = 2 / rho^2.
```

## KKT Conditions

For the hard-margin problem:

```text
alpha_i >= 0
y_i (w^T x_i + b) - 1 >= 0
alpha_i [ y_i (w^T x_i + b) - 1 ] = 0
w = sum_i alpha_i y_i x_i
sum_i alpha_i y_i = 0.
```

Consequences:

- If `alpha_i = 0`, the point does not contribute to `w`.
- If `alpha_i > 0`, the point lies on a margin hyperplane and is a support
  vector: `y_i f(x_i) = 1`.
- The decision function is unique, though the support-vector expansion need not
  be unique in degenerate cases.

## Soft-Margin Formulation

The paper first motivates non-separable classification by adding slack variables
`xi_i >= 0`:

```text
y_i (w^T x_i + b) >= 1 - xi_i.
```

The reproduction implements the requested L1 soft-margin objective:

```text
minimize    1/2 ||w||^2 + C sum_i xi_i
subject to  y_i (w^T x_i + b) >= 1 - xi_i
            xi_i >= 0.
```

The appendix derives the same standard box-constrained dual for `F(u)=u`:

```text
maximize    sum_i alpha_i
            - 1/2 sum_i sum_j alpha_i alpha_j y_i y_j K(x_i, x_j)

subject to  0 <= alpha_i <= C
            sum_i alpha_i y_i = 0.
```

Soft-margin KKT conditions:

```text
alpha_i >= 0,  mu_i >= 0
y_i f(x_i) >= 1 - xi_i
xi_i >= 0
alpha_i [ y_i f(x_i) - 1 + xi_i ] = 0
mu_i xi_i = 0
w = sum_i alpha_i y_i phi(x_i)
sum_i alpha_i y_i = 0
C - alpha_i - mu_i = 0.
```

Thus `0 <= alpha_i <= C`. Interpretations:

- `alpha_i = 0`: correctly classified outside the margin.
- `0 < alpha_i < C`: exactly on the margin, `y_i f(x_i)=1`.
- `alpha_i = C`: inside the margin or misclassified.

The paper also discusses more general convex penalties and a squared-error
variant. Those are documented but not the primary estimator because the requested
objective is the L1 soft margin above.

## Kernel Trick

The feature-space classifier is

```text
f(x) = w^T phi(x) + b
     = sum_i alpha_i y_i phi(x_i)^T phi(x) + b.
```

If a symmetric function `K(u, v)` satisfies Mercer's condition, it can be used as
a dot product in a Hilbert feature space:

```text
K(u, v) = phi(u)^T phi(v).
```

The kernelized dual replaces `x_i^T x_j` by `K(x_i, x_j)`:

```text
D_ij = y_i y_j K(x_i, x_j).
```

Prediction becomes

```text
predict(x) = sign( sum_{i in SV} alpha_i y_i K(x_i, x) + b ).
```

Only support vectors with nonzero `alpha_i` are needed at prediction time.

## Polynomial Kernel Derivation

The paper's main experimental kernel is

```text
K(u, v) = (u^T v + 1)^d.
```

For `d=2`, the expansion contains constant, linear, squared, and pairwise
interaction terms. In the paper's example, the feature coordinates include

```text
x_1, ..., x_n,
x_1^2, ..., x_n^2,
x_1 x_2, ..., x_n x_{n-1}.
```

More generally, `(u^T v + 1)^d` corresponds to all monomials up to total degree
`d`, with scaling constants. This is why high-degree polynomial classifiers can
be trained without explicitly constructing the huge feature space.

## Gaussian / RBF Kernel

The paper mentions radial basis support-vector networks through kernels of the
form

```text
K(u, v) = exp( -||u-v||^2 / sigma_related_scale ).
```

The reproduction uses the modern parameterization

```text
K(u, v) = exp( -||u-v||^2 / (2 sigma^2) ).
```

## Training Algorithm

The paper describes solving the dual quadratic optimization problem and gives an
incremental chunking scheme:

1. Split training data into portions.
2. Solve the QP on the first portion.
3. Keep support vectors from the current portion.
4. Add examples from the next portion that violate the current constraints.
5. Repeat until all portions are processed or separability fails.

The reproduction implements:

- `DualQPSolver`: exact small-problem QP using SciPy's constrained optimizer.
- `SMOOptimizer`: from-scratch sequential minimal optimization for the dual.
- `ChunkingOptimizer`: an educational active-set/chunking wrapper inspired by
  the paper.

## Prediction Algorithm

For a fitted binary model:

```text
score(x) = sum_{i in SV} alpha_i y_i K(x_i, x) + b
label(x) = +1 if score(x) >= 0 else -1.
```

For digit recognition, the paper trains ten one-vs-rest classifiers and predicts
the class with largest raw output:

```text
label(x) = argmax_c f_c(x).
```

## Experimental Setup Extracted from the Paper

### Synthetic 2D Experiments

- Use 2D artificial pattern sets.
- Kernel: `K(u, v) = (u^T v + 1)^2`.
- Figures show support patterns with double circles and training errors with
  crosses.
- Exact coordinates are not specified, so the reproduction generates comparable
  quadratic, XOR, elliptical, and overlapping 2D datasets.

### USPS Digit Recognition

- Dataset: US Postal Service digits.
- Paper size: about 7,300 training and 2,000 test patterns.
- Resolution: 16 x 16, so input dimensionality is 256.
- Preprocessing: centering, deslanting, and Gaussian smoothing.
- Smoothing: Gaussian standard deviation `sigma = 0.75`.
- Classifier: ten one-vs-rest polynomial SVMs.
- Kernels: degrees 1 through 7, `K(u, v) = (u^T v + 1)^d`.
- Prediction: maximum output over the ten classifiers.

Original Table 2:

| Degree | Raw error % | Mean support vectors | Feature-space dimension |
| --- | ---: | ---: | --- |
| 1 | 12.0 | 200 | 256 |
| 2 | 4.7 | 127 | about 33000 |
| 3 | 4.4 | 148 | about 1e6 |
| 4 | 4.3 | 165 | about 1e9 |
| 5 | 4.3 | 175 | about 1e12 |
| 6 | 4.2 | 185 | about 1e14 |
| 7 | 4.3 | 190 | about 1e16 |

### NIST / MNIST Experiment

- Paper dataset: NIST Special Database 3 mixture.
- Size: 60,000 training and 10,000 test patterns.
- Resolution: 28 x 28, input dimensionality 784.
- Preprocessing: none.
- Classifier: degree-4 polynomial kernel, ten one-vs-rest classifiers.
- Combined test error reported by paper: 1.1%.

Original Table 3 per binary classifier:

| Classifier | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Support patterns | 1379 | 989 | 1958 | 1900 | 1224 | 2024 | 1527 | 2064 | 2332 | 2765 |
| Train errors | 7 | 16 | 8 | 11 | 2 | 4 | 8 | 16 | 4 | 1 |
| Test errors | 19 | 14 | 35 | 35 | 36 | 49 | 32 | 43 | 48 | 63 |

## Assumptions and Ambiguities

- The paper does not provide exact synthetic data coordinates; generated
  datasets are visual analogues, not bitwise reproductions.
- The paper does not report the chosen soft-margin `C`; scripts expose `--C` and
  default to `10.0`.
- USPS mirrors provide 7,291/2,007 examples rather than exactly 7,300/2,000 in
  many modern distributions.
- The exact NIST split is not readily available; the reproduction uses MNIST when
  NIST is unavailable.
- Centering and deslanting are described but not algorithmically specified; the
  code uses intensity moments and horizontal shear, which are standard OCR
  preprocessing choices.
- Figure 9 exact non-SVN benchmark bar heights are not fully legible in the
  supplied OCR; comparison files clearly separate reported numeric values from
  recreated/approximate plotted values.
