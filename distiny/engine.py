import numpy as np
from scipy.stats import norm
from collections import Counter

# -----------------------------
# Distributions
# -----------------------------
class NormalDist:
    """
    Fit a Gaussian distribution (mean and std) to numeric data using gradient descent
    to minimize negative log-likelihood.
    Can also behave like a distribution object with `pdf`, `cdf`, and `ppf` methods.
    """
    name = "normal"
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-6):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def _nll(self, x, mu, sigma):
        """Negative log-likelihood of normal distribution."""
        var = sigma ** 2
        return 0.5 * np.sum(np.log(2 * np.pi * var) + ((x - mu) ** 2) / var)

    def _grad(self, x, mu, sigma):
        """Gradients of NLL with respect to mu and sigma."""
        n = len(x)
        dmu = -np.sum(x - mu) / (sigma ** 2)
        dsigma = -n / sigma + np.sum(((x - mu) ** 2)) / (sigma ** 3)
        return dmu, dsigma

    def fit(self, x):
        x = np.asarray(x)
        mu = np.mean(x)
        sigma = np.std(x)

        for _ in range(self.max_iter):
            dmu, dsigma = self._grad(x, mu, sigma)
            mu -= self.learning_rate * dmu
            sigma -= self.learning_rate * dsigma
            sigma = max(sigma, 1e-6)

            if np.abs(dmu) < self.tol and np.abs(dsigma) < self.tol:
                break

        self.mu_ = mu
        self.sigma_ = sigma
        return mu, sigma

    def pdf(self, x):
        return norm.pdf(x, loc=self.mu_, scale=self.sigma_)

    def cdf(self, x):
        return norm.cdf(x, loc=self.mu_, scale=self.sigma_)

    def ppf(self, q):
        return norm.ppf(q, loc=self.mu_, scale=self.sigma_)

    def get_params(self):
        return self.mu_, self.sigma_

    def set_params(self, mu, sigma):
        self.mu_ = mu
        self.sigma_ = sigma

class UniformDist:
    name = "uniform"

    @staticmethod
    def fit(data):
        a, b = np.min(data), np.max(data)
        return a, b

    @staticmethod
    def cdf(x, params):
        a, b = params
        return (x - a) / (b - a + 1e-8)

    @staticmethod
    def ppf(u, params):
        a, b = params
        return a + u * (b - a)

class ExponentialDist:
    name = "exponential"

    @staticmethod
    def fit(data):
        lam = 1 / (np.mean(data) + 1e-8)
        return lam,

    @staticmethod
    def cdf(x, params):
        lam, = params
        return 1 - np.exp(-lam * x)

    @staticmethod
    def ppf(u, params):
        lam, = params
        return -np.log(1 - u + 1e-8) / lam

# -----------------------------
# Utilities
# -----------------------------
def infer_column_type(col):
    unique_vals = np.unique(col)
    if col.dtype.kind in {'U', 'S', 'O'}:
        return "categorical"
    elif col.dtype.kind in {'i'} and len(unique_vals) < 15:
        return "ordinal"
    else:
        return "numeric"

def ks_statistic(data, cdf_func, params):
    sorted_data = np.sort(data)
    n = len(data)
    empirical_cdf = np.arange(1, n + 1) / n
    theoretical_cdf = cdf_func(sorted_data, params)
    return np.max(np.abs(empirical_cdf - theoretical_cdf))

def fit_best_distribution(data):
    candidates = [NormalDist, UniformDist, ExponentialDist]
    best_fit = None
    best_params = None
    best_score = float('inf')
    for dist in candidates:
        try:
            params = dist.fit(data)
            score = ks_statistic(data, dist.cdf, params)
            if score < best_score:
                best_fit = dist
                best_params = params
                best_score = score
        except:
            continue
    return best_fit, best_params

# -----------------------------
# Synthesizer Class
# -----------------------------
class HybridSynthesizer:
    def __init__(self):
        self.column_types = []
        self.marginals = []
        self.cov = None

    def fit(self, data, column_types=None):
        if column_types is None:
            self.column_types = [infer_column_type(data[:, i]) for i in range(data.shape[1])]
        else:
            self.column_types = column_types

        transformed = []
        self.marginals = []

        for i in range(data.shape[1]):
            col = data[:, i]
            col_type = self.column_types[i]

            if col_type in ["numeric", "ordinal"]:
                col = col.astype(float)
                dist, params = fit_best_distribution(col)
                cdf_vals = dist.cdf(col, params)
                cdf_vals = np.clip(cdf_vals, 1e-6, 1 - 1e-6)  # Avoid infs in norm.ppf
                z = norm.ppf(cdf_vals)
                self.marginals.append((col_type, dist, params))
                transformed.append(z)
            elif col_type == "categorical":
                counter = Counter(col)
                categories = list(counter.keys())
                probs = [counter[c] / len(col) for c in categories]
                self.marginals.append((col_type, categories, probs))

        z_matrix = np.column_stack(transformed)
        self.cov = np.corrcoef(z_matrix.T)

    def sample(self, n_samples):
        num_cols = len([m for m in self.marginals if m[0] != "categorical"])
        mvn_samples = np.random.multivariate_normal(np.zeros(num_cols), self.cov, n_samples)

        synthetic_data = []
        mvn_index = 0

        for col_type, *info in self.marginals:
            if col_type in ["numeric", "ordinal"]:
                dist, params = info
                u = norm.cdf(mvn_samples[:, mvn_index])
                col = dist.ppf(u, params)
                if col_type == "ordinal":
                    col = np.round(col).astype(int)
                synthetic_data.append(col)
                mvn_index += 1
            else:
                categories, probs = info
                sampled = np.random.choice(categories, size=n_samples, p=probs)
                synthetic_data.append(sampled)

        return np.array(list(zip(*synthetic_data)), dtype=object)
