import numpy as np
from collections import Counter
from scipy.stats import ks_2samp, chisquare
from distiny.engine import HybridSynthesizer, NormalDist

def compare_numeric(real_col, synth_col):
    stat, pval = ks_2samp(real_col.astype(float), synth_col.astype(float))
    return stat, pval

def compare_categorical(real_col, synth_col):
    real_counts = Counter(real_col)
    synth_counts = Counter(synth_col)
    keys = sorted(set(real_counts) | set(synth_counts))
    real_freq = np.array([real_counts.get(k, 0) for k in keys])
    synth_freq = np.array([synth_counts.get(k, 0) for k in keys])
    stat, pval = chisquare(f_obs=synth_freq + 1e-8, f_exp=real_freq + 1e-8)
    return stat, pval

def test_distribution_similarity():
    # Generate real data
    np.random.seed(42)
    real_data = np.array([
        np.random.normal(20, 5, 100) + np.random.normal(60, 5, 100),           # numeric: age
        np.random.uniform(50000, 80000, 100),   # numeric: salary
        np.random.randint(1, 6, 100),           # ordinal
        np.random.choice(["M", "F"], 100)       # categorical
    ], dtype=object).T  # transpose to rows

    column_types = ["numeric", "numeric", "ordinal", "categorical"]

    model = HybridSynthesizer()
    model.fit(real_data, column_types=column_types)
    synthetic = model.sample(n_samples=100)

    for i, col_type in enumerate(column_types):
        real_col = real_data[:, i]
        synth_col = synthetic[:, i]

        if col_type in ["numeric", "ordinal"]:
            _, pval = compare_numeric(real_col, synth_col)
        elif col_type == "categorical":
            _, pval = compare_categorical(real_col, synth_col)
        print(f"p-value: {pval}")
        assert pval > 0.05, f"Distribution mismatch in column {i} ({col_type})"
    print("Yay!!")
