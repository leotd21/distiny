import numpy as np
import pytest
from distiny.engine import HybridSynthesizer

def test_hybrid_synthesizer_generates_data_correctly():
    # Sample input data
    data = np.array([
        [25, 50000, 3, 'M'],
        [30, 60000, 4, 'F'],
        [22, 52000, 2, 'F'],
        [28, 58000, 3, 'M'],
        [35, 75000, 5, 'M'],
    ], dtype=object)

    # Explicit column types: numeric, numeric, ordinal, categorical
    column_types = ["numeric", "numeric", "ordinal", "categorical"]

    model = HybridSynthesizer()
    model.fit(data, column_types=column_types)

    synthetic = model.sample(n_samples=5)
    print(synthetic)

    assert synthetic.shape == data.shape, "Shape mismatch between real and synthetic data"
    
    # Numeric columns should be float
    assert all(isinstance(val, float) for val in synthetic[:, 0]), "Age column not float"
    assert all(isinstance(val, float) for val in synthetic[:, 1]), "Salary column not float"

    # Ordinal column should be integers
    assert all(isinstance(val, (int, np.integer)) for val in synthetic[:, 2])

    # Categorical column should contain valid strings
    assert all(isinstance(val, str) for val in synthetic[:, 3])
