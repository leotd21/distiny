# distiny

![awww](this_is_not_meth.jpg)

A tiny distribution prediction engine. Implements univariate statistical modeling with Gaussian Copula dependency modeling. It supports mixed data types: **numerical**, **ordinal**, and **categorical**.

---

## High-Level Intuition

Real-world tabular datasets often contain:
- Continuous numeric features
- Ordered discrete features (e.g., ratings)
- Categorical features (e.g., gender, region)

This generator works in **two main phases**:

1. **Model each column (marginals)** using the best-fitting statistical distribution.
2. **Model inter-column dependencies** using a **Gaussian Copula**.

By decoupling marginals and correlations, we generate realistic yet controllable synthetic data.

---

## How It Works (Step-by-Step)

### Step 1: Column Type Inference
- Automatically infer if a column is:
  - `numeric`: continuous or wide-range integers
  - `ordinal`: integer values with an inherent order (e.g., 1–5)
  - `categorical`: strings or small sets of labels with no order

➡️ Alternatively, you can pass an explicit schema.

---

### Step 2: Fit Marginal Distributions (Per Column)
- For each numeric or ordinal column:
  - Fit candidate distributions (Normal, Uniform, Exponential)
  - Select the best using **Kolmogorov-Smirnov (KS) statistic**
- For categorical:
  - Estimate the empirical distribution (value frequencies)

**Reference**: [Kolmogorov–Smirnov test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)

---

### Step 3: Transform to Uniform [0, 1]
- Apply the **CDF** of the fitted distribution to each numeric/ordinal value.
- This maps data to the **uniform space**, a key step for Copula modeling.

---

### Step 4: Gaussianize (Probit Transform)
- Convert uniform values to standard normal via:

  \[ z = \Phi^{-1}(u) \]

  Where \( \Phi^{-1} \) is the inverse CDF of the standard normal distribution.

**Reference**: [Probit function](https://en.wikipedia.org/wiki/Probit_function)

---

### Step 5: Fit Correlation (Gaussian Copula)
- On the Gaussianized matrix, compute the **correlation matrix**.
- This captures dependencies between features under the Gaussian assumption.

**Reference**: [Copula (statistics)](https://en.wikipedia.org/wiki/Copula_(probability_theory))

---

### Step 6: Sample New Gaussianized Points
- Draw new samples from a multivariate normal distribution:

  \[ Z' \sim \mathcal{N}(0, \Sigma) \]

---

### Step 7: Reverse Transform
- Convert back to uniform space using standard normal CDF
- For each column:
  - Apply the **inverse CDF** of the fitted marginal to get synthetic values
  - For categorical: sample using multinomial distribution

---

## Design Choices Summary

| Column Type | Transformation | Copula | Synthesis Method |
|-------------|----------------|--------|------------------|
| Numeric     | CDF → Probit   | ✅     | Inverse CDF      |
| Ordinal     | Empirical CDF  | ✅     | Inverse CDF (rounded) |
| Categorical | None           | ❌     | Multinomial sample |

---

## Bug Fixes & Stability Improvements

### 🛠 Issue: `SVD did not converge` during sampling
This error was caused by `norm.ppf` encountering values of `0` or `1`, which results in `inf` and breaks the Gaussian Copula correlation matrix.

**✅ Fix:**
- Clip values passed to `norm.ppf` with a small epsilon margin:

```python
cdf_vals = np.clip(cdf_vals, 1e-6, 1 - 1e-6)
```

This prevents extreme values and ensures numerical stability when converting to Gaussian space.

### 🛠 Issue: All synthetic columns inferred as `object` or string
Originally, the synthetic output was returned as an object array, leading to loss of numeric typing.

**✅ Fix:**
- Use NumPy’s `np.array(..., dtype=object)` when columns are mixed type
- For numeric-only datasets, columns retain their float or int types internally

The updated `sample()` method now returns well-typed NumPy arrays for downstream use.

---

## Unit Testing
- Generates synthetic data from a toy dataset with:
  - Numeric: `age`, `salary`
  - Ordinal: `grade` (1–5)
  - Categorical: `gender`
- Ensures the output format and types are preserved
- Validates marginal distribution similarity (KS test or histogram overlap)

Run all tests with:
```bash
pytest tests/
```

---

## References
- [Gaussian Copula Models](https://en.wikipedia.org/wiki/Copula_(probability_theory))
- [Synthetic Data with Gaussian Copula](https://sdv.dev/SDV/user_guides/single_table/gaussian_copula.html)
- [Kolmogorov-Smirnov Test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
- [Probit Model](https://en.wikipedia.org/wiki/Probit_model)

---

## 📜 License
MIT
