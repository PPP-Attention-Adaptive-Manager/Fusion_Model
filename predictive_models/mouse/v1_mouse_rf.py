"""
Mouse Random Forest baseline.
Sklearn-based. Does NOT subclass BaseModalityModel — cannot plug into fusion.
Purpose: comparison baseline only.
Trained and evaluated via train_mouse_models.py.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
from pathlib import Path


class MouseRandomForest:
    """
    Two separate RF models:
      clf  — RandomForestClassifier → predicts state (0-4)
      reg  — RandomForestRegressor  → predicts 5 NASA-TLX factors
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth:    int = None,
        random_state: int = 42,
    ):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        self.reg = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y_state: np.ndarray, y_factors: np.ndarray):
        """
        Args:
            X:         (N, 512) Tucker slices
            y_state:   (N,)     int  0-4
            y_factors: (N, 5)   float  normalized [0,1]
        """
        self.clf.fit(X, y_state)
        self.reg.fit(X, y_factors)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            states:  (N,)   int predictions
            factors: (N, 5) float predictions
        """
        return self.clf.predict(X), self.reg.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns (N, 5) class probabilities."""
        return self.clf.predict_proba(X)

    def save(self, path: str | Path):
        joblib.dump({"clf": self.clf, "reg": self.reg}, path)

    @classmethod
    def load(cls, path: str | Path) -> "MouseRandomForest":
        obj  = cls.__new__(cls)
        data = joblib.load(path)
        obj.clf = data["clf"]
        obj.reg = data["reg"]
        return obj
