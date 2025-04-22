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
### Visual Overview

```text
[Raw Data]
    ↓
[Step 1: Column Type Inference]
    ↓
[Step 2: Fit Column Distributions]
    ↓
[Step 3: Normalize to [0,1]]
    ↓
[Step 4: Gaussianize Data]
    ↓
[Step 5: Learn Feature Correlation]
    ↓
[Step 6: Sample Synthetic Data]
    ↓
[Step 7: Reverse Transform to Raw Format]
    ↓
[Synthetic Data Output]
```

---

## How It Works
### Step 1: Understand the Column Types
- We first figure out what kind of data each column has:
  - Numbers with real values → **numeric**
  - Ordered small integers → **ordinal**
  - Text labels or limited categories → **categorical**

(You can let the system guess or provide your own schema.)

---

### Step 2: Fit Marginal Distributions (Per Column)
- Try out a few common distributions and pick the one that best matches the data. 
- For each numeric or ordinal column:
  - Fit candidate distributions (Normal, Uniform, Exponential)
  - Select the best using **Kolmogorov-Smirnov (KS) statistic**
- For categorical:
  - Estimate the empirical distribution (value frequencies). Just count how often each label appears.

---

### Step 3: Normalize Everything (Turn Data into 0 to 1 Range)
- We apply a transformation that spreads all values into a range between 0 and 1. This makes different types of data easier to compare and mix.
- Apply the **CDF** of the fitted distribution to each numeric/ordinal value.
- This maps data to the **uniform space**, a key step for Copula modeling.
---

### Step 4: Make the Data Bell-Shaped
- We turn those 0-to-1 values into values that follow a bell curve (like height in a population).
- This helps us capture how features depend on each other.
- Convert uniform values to standard normal via [probit function](https://en.wikipedia.org/wiki/Probit)

---

### Step 5: Understand How Features Relate
- We measure how features move together — like whether salary increases with age — using a correlation matrix.
- Fit Correlation (Gaussian Copula): on the Gaussianized matrix, compute the **correlation matrix**.
- This captures dependencies between features under the Gaussian assumption. [Copula (statistics)](https://en.wikipedia.org/wiki/Copula_(probability_theory))

---

### Step 6: Create New Synthetic Data
- We use the learned relationships to create new rows of data that look like the original but are freshly generated.
- Draw new samples from a multivariate normal distribution.

---

### Step 7: Convert Everything Back
- We reverse all the earlier transformations to get the new data back into the original formats (e.g., labels for gender, integers for grade).
- Reverse Transform: Convert back to uniform space using standard normal CDF
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

### Issue: `SVD did not converge` during sampling
This error was caused by `norm.ppf` encountering values of `0` or `1`, which results in `inf` and breaks the Gaussian Copula correlation matrix.

**✅ Fix:**
- Clip values passed to `norm.ppf` with a small epsilon margin:

```python
cdf_vals = np.clip(cdf_vals, 1e-6, 1 - 1e-6)
```

This prevents extreme values and ensures numerical stability when converting to Gaussian space.

### Issue: All synthetic columns inferred as `object` or string
Originally, the synthetic output was returned as an object array, leading to loss of numeric typing.

**✅ Fix:**
- Use NumPy’s `np.array(..., dtype=object)` when columns are mixed type
- For numeric-only datasets, columns retain their float or int types internally

The updated `sample()` method now returns well-typed NumPy arrays for downstream use.

---

## Sanity check
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

## License
MIT
