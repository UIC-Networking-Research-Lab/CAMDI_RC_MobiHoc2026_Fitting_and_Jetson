"""Fit and serialize accuracy regressors from compression measurements."""
import os
import sys
import glob
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

_MODEL_BUILDERS = {
    "linear_monotonic": lambda: LinearRegression(positive=True),
    "poly2":     lambda: Ridge(alpha=0.1),
    "poly3":     lambda: Ridge(alpha=0.1),
    "gbm":       lambda: GradientBoostingRegressor(
                     n_estimators=300, max_depth=8, learning_rate=0.05,
                     random_state=42),
    "rf":        lambda: RandomForestRegressor(
                     n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    "mlp":       lambda: MLPRegressor(
                     hidden_layer_sizes=(128, 64, 32), max_iter=3000,
                     random_state=42, early_stopping=True),
    "mlp_small": lambda: MLPRegressor(
                     hidden_layer_sizes=(64, 32), max_iter=2000,
                     random_state=42, early_stopping=True),
}


def load_csv_data(path_or_dir,
                  k_columns=None,
                  target_column="avg_bleu",
                  compressor_filter="topk",
                  k_values_column="k_values"):
    """
    Load experiment CSV(s) and return feature matrix X and target vector y.

    Parameters
    ----------
    path_or_dir : str
        A single CSV path **or** a directory (all ``*_results_*.csv`` inside
        will be concatenated).
    k_columns : list[int] or list[str] or None
        * ``None`` — auto-parse the ``k_values_column`` string and use ALL
          resulting columns (``k0, k1, ...``).
        * ``list[int]`` — indices into the parsed k-vector to keep, e.g.
          ``[2, 5, 8]`` picks 3 out of 11.
        * ``list[str]`` — explicit column names already present in the CSV,
          e.g. ``["param_a", "param_b"]`` (skips k_values parsing entirely).
    target_column : str
        CSV column for the regression target.
    compressor_filter : str or None
        If not None, keep only rows where ``df["compressor"] == compressor_filter``.
    k_values_column : str
        CSV column that stores a comma-separated string of k-values.
        Ignored when ``k_columns`` is a list of existing column names.

    Returns
    -------
    X : ndarray (n_samples, n_features)
    y : ndarray (n_samples,)
    n_features : int
    feature_names : list[str]
    """
    # Collect files
    if os.path.isdir(path_or_dir):
        files = sorted(glob.glob(os.path.join(path_or_dir, "*_results_*.csv")))
        if not files:
            raise FileNotFoundError(
                f"No *_results_*.csv files found in {path_or_dir}")
    else:
        files = [path_or_dir]

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # Optional compressor filter
    if compressor_filter and "compressor" in df.columns:
        df = df[df["compressor"] == compressor_filter].copy()

    # Resolve feature columns
    if k_columns is not None and all(isinstance(c, str) for c in k_columns):
        # User supplied explicit column names already in the CSV
        feature_names = list(k_columns)
    else:
        # Parse comma-separated k_values string -> k0, k1, ...
        if k_values_column not in df.columns:
            raise KeyError(
                f"Column '{k_values_column}' not in CSV. "
                f"Available: {list(df.columns)}")
        parsed = df[k_values_column].str.split(",", expand=True).astype(float)
        n_total = parsed.shape[1]
        for i in range(n_total):
            df[f"k{i}"] = parsed[i]

        if k_columns is None:
            indices = list(range(n_total))
        else:
            indices = list(k_columns)
        feature_names = [f"k{i}" for i in indices]

    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' not in CSV. "
            f"Available: {list(df.columns)}")

    X = df[feature_names].values.astype(float)
    y = df[target_column].values.astype(float)

    print(f"Loaded {len(y)} samples from {len(files)} file(s), "
          f"{len(feature_names)} features {feature_names}")
    return X, y, len(feature_names), feature_names


class AccuracyEstimator:
    """
    Flexible accuracy estimator — auto-adapts to any number of cut points.

    Parameters
    ----------
    model_type : str
        ``"linear_monotonic"``, ``"poly2"``, ``"poly3"``, ``"gbm"``, ``"rf"``,
        ``"mlp"``, ``"mlp_small"``.
    """

    def __init__(self, model_type="poly2"):
        if model_type not in _MODEL_BUILDERS:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Choose from {list(_MODEL_BUILDERS)}")
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.poly = None           # set for poly2/poly3
        self.n_features = None     # set during train()
        self.feature_names = None
        self.trained = False
        self.metrics = {}

    # ── Training ─────────────────────────────────────────────────────────

    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train from numpy arrays.  ``n_features`` is auto-detected from X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        test_size : float
        random_state : int

        Returns
        -------
        dict — metrics (train/test RMSE, MAE, R²).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self.n_features = X.shape[1]
        if self.feature_names is None:
            self.feature_names = [f"f{i}" for i in range(self.n_features)]

        print(f"Training {self.model_type} estimator  "
              f"({len(y)} samples, {self.n_features} features)")

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

        # Scale
        X_tr_s = self.scaler.fit_transform(X_tr)
        X_te_s = self.scaler.transform(X_te)

        # Polynomial expansion
        if self.model_type.startswith("poly"):
            degree = int(self.model_type[-1])
            self.poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_tr_s = self.poly.fit_transform(X_tr_s)
            X_te_s = self.poly.transform(X_te_s)

        self.model = _MODEL_BUILDERS[self.model_type]()

        t0 = time.time()
        self.model.fit(X_tr_s, y_tr)
        elapsed = time.time() - t0

        yp_tr = np.clip(self.model.predict(X_tr_s), 0, 1)
        yp_te = np.clip(self.model.predict(X_te_s), 0, 1)

        self.metrics = {
            "train_rmse": float(np.sqrt(mean_squared_error(y_tr, yp_tr))),
            "train_r2":   float(r2_score(y_tr, yp_tr)),
            "test_rmse":  float(np.sqrt(mean_squared_error(y_te, yp_te))),
            "test_mae":   float(mean_absolute_error(y_te, yp_te)),
            "test_r2":    float(r2_score(y_te, yp_te)),
            "train_time":  elapsed,
        }
        print(f"  Train RMSE={self.metrics['train_rmse']:.4f}  "
              f"R²={self.metrics['train_r2']:.4f}")
        print(f"  Test  RMSE={self.metrics['test_rmse']:.4f}  "
              f"R²={self.metrics['test_r2']:.4f}  "
              f"({elapsed:.3f}s)")

        # Retrain on full dataset for production
        X_all_s = self.scaler.fit_transform(X)
        if self.poly is not None:
            X_all_s = self.poly.fit_transform(X_all_s)
        self.model.fit(X_all_s, y)
        self.trained = True
        return self.metrics

    def train_from_csv(self, path_or_dir, k_columns=None,
                       target_column="avg_bleu", compressor_filter="topk",
                       **kwargs):
        """
        Convenience: load CSV(s) then train.

        All parameters are forwarded to ``load_csv_data()``.
        """
        X, y, _, names = load_csv_data(
            path_or_dir,
            k_columns=k_columns,
            target_column=target_column,
            compressor_filter=compressor_filter,
            **kwargs,
        )
        self.feature_names = names
        return self.train(X, y)

    # ── Prediction ───────────────────────────────────────────────────────

    def predict(self, k_values):
        """
        Predict accuracy for one sample.

        Parameters
        ----------
        k_values : list[float]   length must equal ``self.n_features``.

        Returns
        -------
        float   clipped to [0, 1].
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() first.")
        k_values = list(k_values)
        if len(k_values) != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} values, got {len(k_values)}")
        X = np.array(k_values, dtype=float).reshape(1, -1)
        X = self.scaler.transform(X)
        if self.poly is not None:
            X = self.poly.transform(X)
        return float(np.clip(self.model.predict(X)[0], 0, 1))

    def predict_batch(self, k_matrix):
        """
        Predict accuracy for multiple samples.

        Parameters
        ----------
        k_matrix : array-like, shape (n, n_features)

        Returns
        -------
        ndarray, shape (n,)
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() first.")
        X = np.asarray(k_matrix, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} columns, got {X.shape[1]}")
        X = self.scaler.transform(X)
        if self.poly is not None:
            X = self.poly.transform(X)
        return np.clip(self.model.predict(X), 0, 1)

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path):
        """Save trained estimator to a pickle file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "poly": self.poly,
                "model_type": self.model_type,
                "n_features": self.n_features,
                "feature_names": self.feature_names,
                "metrics": self.metrics,
                "trained": self.trained,
            }, f)
        print(f"Saved -> {path}")

    @classmethod
    def load(cls, path):
        """Load a previously saved estimator."""
        with open(path, "rb") as f:
            d = pickle.load(f)
        est = cls.__new__(cls)
        est.model = d["model"]
        est.scaler = d["scaler"]
        est.poly = d.get("poly")
        est.model_type = d["model_type"]
        est.n_features = d["n_features"]
        est.feature_names = d.get("feature_names")
        est.metrics = d.get("metrics", {})
        est.trained = d.get("trained", True)
        print(f"Loaded <- {path}  ({est.model_type}, {est.n_features} features)")
        return est

    # ── Fallback ─────────────────────────────────────────────────────────

    @staticmethod
    def geometric_mean(k_values):
        """Zero-config fallback: geometric mean of k-values."""
        v = np.asarray(k_values, dtype=float)
        return float(np.prod(v) ** (1.0 / len(v)))

    # ── Info ─────────────────────────────────────────────────────────────

    def summary(self):
        """Print a one-line summary."""
        status = "trained" if self.trained else "untrained"
        n = self.n_features or "?"
        r2 = self.metrics.get("test_r2", "N/A")
        if isinstance(r2, float):
            r2 = f"{r2:.4f}"
        return f"AccuracyEstimator({self.model_type}, {n} features, {status}, R²={r2})"

    def __repr__(self):
        return self.summary()


def compare_models(X, y, test_size=0.2, random_state=42):
    """
    Train every registered model type, return a comparison DataFrame.

    Returns
    -------
    pd.DataFrame   sorted by RMSE ascending.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Pre-expand polynomial variants
    poly2 = PolynomialFeatures(2, include_bias=False)
    poly3 = PolynomialFeatures(3, include_bias=False)
    Xtr2 = poly2.fit_transform(X_tr_s); Xte2 = poly2.transform(X_te_s)
    Xtr3 = poly3.fit_transform(X_tr_s); Xte3 = poly3.transform(X_te_s)

    specs = {
        "linear_monotonic": (X_tr_s, X_te_s),
        "poly2":     (Xtr2, Xte2),
        "poly3":     (Xtr3, Xte3),
        "gbm":       (X_tr_s, X_te_s),
        "rf":        (X_tr_s, X_te_s),
        "mlp":       (X_tr_s, X_te_s),
        "mlp_small": (X_tr_s, X_te_s),
    }

    rows = []
    for name, (Xtr, Xte) in specs.items():
        mdl = _MODEL_BUILDERS[name]()
        t0 = time.time()
        mdl.fit(Xtr, y_tr)
        train_time = time.time() - t0
        yp = np.clip(mdl.predict(Xte), 0, 1)
        rmse = float(np.sqrt(mean_squared_error(y_te, yp)))
        mae  = float(mean_absolute_error(y_te, yp))
        r2   = float(r2_score(y_te, yp))
        rows.append(dict(model=name, RMSE=rmse, MAE=mae, R2=r2,
                         train_time_s=train_time))
        print(f"  {name:12s}  RMSE={rmse:.4f}  R²={r2:.4f}  ({train_time:.2f}s)")

    return pd.DataFrame(rows).sort_values("RMSE")
