"""XGBoost wrapper used as the second stage of the GNN+XGBoost ensemble.

Implementation-plan settings (page 7)::

    n_estimators=500, max_depth=8, learning_rate=0.05,
    scale_pos_weight=27.6 (= neg/pos ratio in IEEE-CIS),
    tree_method='hist',  n_jobs=6,
    eval_metric='aucpr'

Wrap rather than subclass: keeps a thin, picklable interface and lets us
treat this as a plain sklearn-style estimator inside the ensemble.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from fraud_detection.utils.logging import get_logger

log = get_logger(__name__)

# IEEE-CIS imbalance (3.5% fraud) -> 0.965/0.035 = 27.57 -> rounded to 27.6
_DEFAULT_SCALE_POS_WEIGHT = 27.6


@dataclass
class XGBoostConfig:
    n_estimators: int = 500
    max_depth: int = 8
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    min_child_weight: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    scale_pos_weight: float = _DEFAULT_SCALE_POS_WEIGHT
    tree_method: str = "hist"
    eval_metric: str = "aucpr"
    early_stopping_rounds: int | None = 30
    n_jobs: int = 6
    random_state: int = 42
    verbosity: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_xgb_kwargs(self) -> dict[str, Any]:
        kw = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "scale_pos_weight": self.scale_pos_weight,
            "tree_method": self.tree_method,
            "eval_metric": self.eval_metric,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "verbosity": self.verbosity,
            "objective": "binary:logistic",
            "early_stopping_rounds": self.early_stopping_rounds,
        }
        kw.update(self.extra)
        return kw


class XGBoostFraudModel:
    """Sklearn-style XGBoost classifier with persistence helpers.

    Notes
    -----
    * ``fit(X_train, y_train, X_val=..., y_val=...)`` enables early stopping.
    * ``predict_proba(X)`` returns a 1-D array of fraud probabilities.
    * Pickle-roundtrips via :meth:`save` / :meth:`load`.
    """

    def __init__(self, config: XGBoostConfig | None = None) -> None:
        self.config = config or XGBoostConfig()
        self.model: xgb.XGBClassifier | None = None
        self._feature_names: list[str] | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> XGBoostFraudModel:
        kwargs = self.config.to_xgb_kwargs()
        if X_val is None or y_val is None:
            # Cannot use early stopping without an eval set.
            kwargs.pop("early_stopping_rounds", None)
        self.model = xgb.XGBClassifier(**kwargs)

        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
        self._feature_names = feature_names
        log.info(
            "xgboost_fit_complete",
            n_train=int(X_train.shape[0]),
            n_features=int(X_train.shape[1]),
            best_iter=getattr(self.model, "best_iteration", None),
            best_score=getattr(self.model, "best_score", None),
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            msg = "XGBoostFraudModel has not been fit"
            raise RuntimeError(msg)
        # Sklearn returns shape (n, 2) for binary -- take the positive col.
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int8)

    # ---- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        if self.model is None:
            msg = "Cannot save an unfit model"
            raise RuntimeError(msg)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {"config": self.config, "model": self.model, "features": self._feature_names},
                f,
            )
        log.info("xgboost_saved", path=str(path))

    @classmethod
    def load(cls, path: str | Path) -> XGBoostFraudModel:
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        inst = cls(payload["config"])
        inst.model = payload["model"]
        inst._feature_names = payload.get("features")
        log.info("xgboost_loaded", path=str(path))
        return inst

    # ---- introspection ---------------------------------------------------

    def feature_importance(self, *, kind: str = "gain") -> dict[str, float]:
        """Return ``{feature_name: importance}`` if names are known, else by index."""
        if self.model is None:
            msg = "Model not fit"
            raise RuntimeError(msg)
        booster = self.model.get_booster()
        scores = booster.get_score(importance_type=kind)
        if self._feature_names:
            # XGBoost names features f0..fN by default; map back.
            mapped: dict[str, float] = {}
            for k, v in scores.items():
                idx = int(k[1:]) if k.startswith("f") and k[1:].isdigit() else None
                if idx is not None and idx < len(self._feature_names):
                    mapped[self._feature_names[idx]] = float(v)
                else:
                    mapped[k] = float(v)
            return mapped
        return {k: float(v) for k, v in scores.items()}


__all__ = ["XGBoostConfig", "XGBoostFraudModel"]
