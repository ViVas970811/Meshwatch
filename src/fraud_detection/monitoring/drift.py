"""Data + concept drift detection (Phase 7).

The :class:`DriftDetector` compares a *reference* distribution (typically the
training split) against a *current* distribution (a window of recent
production transactions) and emits a per-feature :class:`FeatureDrift`
together with an overall :class:`DriftReport`.

Two backends are wired in:

* a pure-NumPy / SciPy implementation -- always available, deterministic,
  fast enough to run on every refresh tick (PSI, KS, chi-square, JSD);
* an optional Evidently bridge -- when ``evidently`` is installed we can
  also emit a rich HTML report. The numpy backend remains the source of
  truth so the API is consistent in CI.

The default thresholds follow the standard credit-risk practitioner cut-offs:

==========  ==========================
PSI value   Interpretation
==========  ==========================
< 0.10      no significant change
0.10-0.25   moderate drift -> warn
>= 0.25     significant drift -> alert
==========  ==========================
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np

# Optional SciPy / pandas imports -- both ship with the base install but we
# keep the import guards in case a downstream user strips them.
try:
    from scipy import stats as _stats

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover -- exercised only in stripped envs
    _HAVE_SCIPY = False

try:
    import pandas as pd

    _HAVE_PANDAS = True
except ImportError:  # pragma: no cover
    _HAVE_PANDAS = False


DriftSeverity = Literal["none", "moderate", "severe"]


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FeatureDrift:
    """Drift statistics for a single feature."""

    feature: str
    kind: Literal["numeric", "categorical"]
    psi: float
    ks_statistic: float | None = None
    ks_pvalue: float | None = None
    chi2_statistic: float | None = None
    chi2_pvalue: float | None = None
    js_divergence: float | None = None
    mean_shift: float | None = None
    severity: DriftSeverity = "none"
    n_reference: int = 0
    n_current: int = 0
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass(slots=True)
class DriftReport:
    """Overall drift report -- a list of feature reports + summary aggregates."""

    features: list[FeatureDrift] = field(default_factory=list)
    overall_psi: float = 0.0
    severity: DriftSeverity = "none"
    n_reference: int = 0
    n_current: int = 0
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reference_label: str = "training"
    current_label: str = "production"
    psi_warn: float = 0.10
    psi_alert: float = 0.25

    # ----- aggregated views ---------------------------------------------------

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def n_moderate(self) -> int:
        return sum(1 for f in self.features if f.severity == "moderate")

    @property
    def n_severe(self) -> int:
        return sum(1 for f in self.features if f.severity == "severe")

    def top(self, k: int = 10) -> list[FeatureDrift]:
        return sorted(self.features, key=lambda f: f.psi, reverse=True)[: max(1, k)]

    # ----- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "reference_label": self.reference_label,
            "current_label": self.current_label,
            "n_reference": self.n_reference,
            "n_current": self.n_current,
            "n_features": self.n_features,
            "n_moderate": self.n_moderate,
            "n_severe": self.n_severe,
            "overall_psi": self.overall_psi,
            "severity": self.severity,
            "psi_warn": self.psi_warn,
            "psi_alert": self.psi_alert,
            "features": [f.to_dict() for f in self.features],
        }


# ---------------------------------------------------------------------------
# Core stats helpers (numpy-only; the SciPy calls are optional add-ons).
# ---------------------------------------------------------------------------


def _to_array(values: Iterable[Any]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=object)
    return arr


def _coerce_numeric(values: Iterable[Any]) -> np.ndarray:
    """Coerce an iterable to ``float64``, dropping NaN/None silently."""
    if _HAVE_PANDAS:
        # pandas tolerates mixed types better than np.asarray.
        return pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy()
    out: list[float] = []
    for v in values:
        if v is None:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(x):
            continue
        out.append(x)
    return np.asarray(out, dtype=np.float64)


def _coerce_categorical(values: Iterable[Any]) -> np.ndarray:
    out: list[str] = []
    for v in values:
        if v is None:
            continue
        try:
            if isinstance(v, float) and math.isnan(v):
                continue
        except TypeError:
            pass
        out.append(str(v))
    return np.asarray(out, dtype=object)


def population_stability_index(
    reference: Sequence[float] | np.ndarray,
    current: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Compute PSI between two 1-D numeric distributions.

    Bins are derived from quantiles of the *reference* distribution so the
    metric is sensitive to the same buckets you tracked in training. Each
    bucket gets a small epsilon to avoid log(0) when a class disappears.
    """
    ref = np.asarray(reference, dtype=np.float64).ravel()
    cur = np.asarray(current, dtype=np.float64).ravel()
    if ref.size == 0 or cur.size == 0:
        return 0.0

    # Quantile edges from the reference; guarantee strict monotonicity by
    # bumping repeats by a tiny delta (handles low-cardinality numerics).
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(ref, quantiles)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    ref_p = ref_hist / max(ref.size, 1)
    cur_p = cur_hist / max(cur.size, 1)
    ref_p = np.where(ref_p == 0, eps, ref_p)
    cur_p = np.where(cur_p == 0, eps, cur_p)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def categorical_psi(
    reference: Sequence[Any],
    current: Sequence[Any],
    *,
    eps: float = 1e-6,
) -> float:
    """PSI for categorical distributions using observed categories from reference."""
    ref = np.asarray(reference, dtype=object).ravel()
    cur = np.asarray(current, dtype=object).ravel()
    if ref.size == 0 or cur.size == 0:
        return 0.0
    cats = np.unique(np.concatenate([ref, cur]))
    ref_p = np.array([(ref == c).sum() for c in cats], dtype=np.float64) / max(ref.size, 1)
    cur_p = np.array([(cur == c).sum() for c in cats], dtype=np.float64) / max(cur.size, 1)
    ref_p = np.where(ref_p == 0, eps, ref_p)
    cur_p = np.where(cur_p == 0, eps, cur_p)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def js_divergence(
    reference: Sequence[float] | np.ndarray,
    current: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 30,
) -> float:
    """Jensen-Shannon divergence (base-2) on histogrammed numeric features.

    Symmetric and bounded in [0, 1]; the square root is the metric form.
    Returned in nats here for consistency with PSI (scale only matters for
    relative comparison across features).
    """
    ref = np.asarray(reference, dtype=np.float64).ravel()
    cur = np.asarray(current, dtype=np.float64).ravel()
    if ref.size == 0 or cur.size == 0:
        return 0.0
    lo = float(min(ref.min(), cur.min()))
    hi = float(max(ref.max(), cur.max()))
    if hi == lo:
        return 0.0
    edges = np.linspace(lo, hi, n_bins + 1)
    rh, _ = np.histogram(ref, bins=edges)
    ch, _ = np.histogram(cur, bins=edges)
    rp = rh / max(rh.sum(), 1)
    cp = ch / max(ch.sum(), 1)
    m = 0.5 * (rp + cp)
    eps = 1e-12
    rp = np.where(rp == 0, eps, rp)
    cp = np.where(cp == 0, eps, cp)
    m = np.where(m == 0, eps, m)
    kl_rm = float(np.sum(rp * np.log(rp / m)))
    kl_cm = float(np.sum(cp * np.log(cp / m)))
    return 0.5 * (kl_rm + kl_cm)


def ks_test(
    reference: Sequence[float] | np.ndarray,
    current: Sequence[float] | np.ndarray,
) -> tuple[float, float | None]:
    """Two-sample Kolmogorov-Smirnov test.

    Returns ``(statistic, p_value)``. p-value is ``None`` if SciPy is missing
    -- the statistic is still a useful drift signal on its own.
    """
    ref = np.asarray(reference, dtype=np.float64).ravel()
    cur = np.asarray(current, dtype=np.float64).ravel()
    if ref.size == 0 or cur.size == 0:
        return 0.0, None
    if _HAVE_SCIPY:
        res = _stats.ks_2samp(ref, cur)
        return float(res.statistic), float(res.pvalue)
    # Numpy fallback -- compute KS statistic by hand on the merged ECDF.
    merged = np.sort(np.concatenate([ref, cur]))
    cdf_r = np.searchsorted(np.sort(ref), merged, side="right") / ref.size
    cdf_c = np.searchsorted(np.sort(cur), merged, side="right") / cur.size
    return float(np.max(np.abs(cdf_r - cdf_c))), None


def chi_square(
    reference: Sequence[Any],
    current: Sequence[Any],
) -> tuple[float, float | None]:
    """Pearson chi-square goodness-of-fit on categorical distributions."""
    ref = np.asarray(reference, dtype=object).ravel()
    cur = np.asarray(current, dtype=object).ravel()
    if ref.size == 0 or cur.size == 0:
        return 0.0, None
    cats = np.unique(np.concatenate([ref, cur]))
    observed = np.array([(cur == c).sum() for c in cats], dtype=np.float64)
    expected = np.array([(ref == c).sum() for c in cats], dtype=np.float64)
    # Scale expected to the current sample size.
    if expected.sum() > 0:
        expected = expected * (observed.sum() / expected.sum())
    expected = np.where(expected == 0, 1e-6, expected)
    stat = float(np.sum((observed - expected) ** 2 / expected))
    if _HAVE_SCIPY:
        dof = max(len(cats) - 1, 1)
        p = float(1 - _stats.chi2.cdf(stat, dof))
        return stat, p
    return stat, None


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------


@dataclass
class DriftDetectorConfig:
    """Tunable knobs for :class:`DriftDetector`."""

    psi_warn: float = 0.10
    psi_alert: float = 0.25
    n_bins: int = 10
    js_bins: int = 30
    severity_combine: Literal["max", "mean"] = "max"
    reference_label: str = "training"
    current_label: str = "production"

    def severity(self, psi: float) -> DriftSeverity:
        if psi >= self.psi_alert:
            return "severe"
        if psi >= self.psi_warn:
            return "moderate"
        return "none"


class DriftDetector:
    """Compute :class:`DriftReport` between a reference and a current sample.

    Parameters
    ----------
    reference
        A mapping from feature name -> 1-D iterable of values seen during
        training. Numeric columns are kept as-is; everything else is
        coerced to string.
    numeric_features, categorical_features
        Optional explicit splits. If omitted, the detector infers the kind
        per feature: numerics if every value is castable to float, else
        categorical.
    config
        Threshold + binning configuration. Defaults are the conventional
        PSI cut-offs (0.10 / 0.25).
    """

    def __init__(
        self,
        reference: Mapping[str, Iterable[Any]],
        *,
        numeric_features: Sequence[str] | None = None,
        categorical_features: Sequence[str] | None = None,
        config: DriftDetectorConfig | None = None,
    ) -> None:
        self.config = config or DriftDetectorConfig()
        self._numeric_ref: dict[str, np.ndarray] = {}
        self._categorical_ref: dict[str, np.ndarray] = {}

        explicit_num = set(numeric_features or [])
        explicit_cat = set(categorical_features or [])

        for feature, values in reference.items():
            if feature in explicit_num:
                self._numeric_ref[feature] = _coerce_numeric(values)
                continue
            if feature in explicit_cat:
                self._categorical_ref[feature] = _coerce_categorical(values)
                continue
            num = _coerce_numeric(values)
            raw = _to_array(values)
            # Heuristic: if a non-trivial portion of values coerced to numeric
            # without loss, treat the column as numeric. Otherwise -- categorical.
            if num.size >= max(1, int(0.6 * max(raw.size, 1))):
                self._numeric_ref[feature] = num
            else:
                self._categorical_ref[feature] = _coerce_categorical(values)

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    @property
    def numeric_features(self) -> list[str]:
        return list(self._numeric_ref.keys())

    @property
    def categorical_features(self) -> list[str]:
        return list(self._categorical_ref.keys())

    @property
    def n_reference(self) -> int:
        if self._numeric_ref:
            return int(next(iter(self._numeric_ref.values())).size)
        if self._categorical_ref:
            return int(next(iter(self._categorical_ref.values())).size)
        return 0

    # ------------------------------------------------------------------
    # detection
    # ------------------------------------------------------------------

    def detect(self, current: Mapping[str, Iterable[Any]]) -> DriftReport:
        """Compute drift statistics for every feature present in *both* sides."""
        features: list[FeatureDrift] = []
        n_current_max = 0

        for name, ref in self._numeric_ref.items():
            if name not in current:
                continue
            cur = _coerce_numeric(current[name])
            n_current_max = max(n_current_max, int(cur.size))
            if cur.size == 0:
                features.append(
                    FeatureDrift(
                        feature=name,
                        kind="numeric",
                        psi=0.0,
                        n_reference=int(ref.size),
                        n_current=0,
                        notes="no current values",
                    )
                )
                continue
            psi = population_stability_index(ref, cur, n_bins=self.config.n_bins)
            ks_stat, ks_p = ks_test(ref, cur)
            jsd = js_divergence(ref, cur, n_bins=self.config.js_bins)
            mean_shift = float(cur.mean() - ref.mean()) if ref.size else 0.0
            features.append(
                FeatureDrift(
                    feature=name,
                    kind="numeric",
                    psi=float(psi),
                    ks_statistic=float(ks_stat),
                    ks_pvalue=ks_p,
                    js_divergence=float(jsd),
                    mean_shift=mean_shift,
                    severity=self.config.severity(psi),
                    n_reference=int(ref.size),
                    n_current=int(cur.size),
                )
            )

        for name, ref_cat in self._categorical_ref.items():
            if name not in current:
                continue
            cur_cat = _coerce_categorical(current[name])
            n_current_max = max(n_current_max, int(cur_cat.size))
            if cur_cat.size == 0:
                features.append(
                    FeatureDrift(
                        feature=name,
                        kind="categorical",
                        psi=0.0,
                        n_reference=int(ref_cat.size),
                        n_current=0,
                        notes="no current values",
                    )
                )
                continue
            psi = categorical_psi(ref_cat, cur_cat)
            chi2, chi2_p = chi_square(ref_cat, cur_cat)
            features.append(
                FeatureDrift(
                    feature=name,
                    kind="categorical",
                    psi=float(psi),
                    chi2_statistic=float(chi2),
                    chi2_pvalue=chi2_p,
                    severity=self.config.severity(psi),
                    n_reference=int(ref_cat.size),
                    n_current=int(cur_cat.size),
                )
            )

        report = DriftReport(
            features=features,
            n_reference=self.n_reference,
            n_current=n_current_max,
            reference_label=self.config.reference_label,
            current_label=self.config.current_label,
            psi_warn=self.config.psi_warn,
            psi_alert=self.config.psi_alert,
        )
        report.overall_psi = self._overall(features)
        report.severity = self.config.severity(report.overall_psi)
        return report

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _overall(self, features: list[FeatureDrift]) -> float:
        if not features:
            return 0.0
        vals = [f.psi for f in features]
        if self.config.severity_combine == "mean":
            return float(np.mean(vals))
        return float(np.max(vals))


# ---------------------------------------------------------------------------
# Optional Evidently bridge
# ---------------------------------------------------------------------------


def evidently_html_report(
    reference: Mapping[str, Iterable[Any]],
    current: Mapping[str, Iterable[Any]],
    *,
    output_path: str | None = None,
) -> str | None:
    """Render an Evidently HTML report when the extra is installed.

    Returns the path written (or ``None`` if Evidently is unavailable). We
    keep this strictly optional -- the numpy backend covers every numeric
    contract we expose via the API; Evidently is purely a richer view for
    humans browsing the dashboard.
    """
    try:
        from evidently import Report  # type: ignore[import-not-found]
        from evidently.presets import DataDriftPreset  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover -- only exercised when extra missing
        return None

    if not _HAVE_PANDAS:  # pragma: no cover
        return None

    ref_df = pd.DataFrame({k: list(v) for k, v in reference.items()})
    cur_df = pd.DataFrame({k: list(v) for k, v in current.items()})

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref_df, current_data=cur_df)
    if output_path is not None:
        snapshot.save_html(output_path)
    return output_path


# ---------------------------------------------------------------------------
# Convenience helpers for the two extra drift flavours from the plan:
#   * label drift            -- production fraud rate vs. training fraud rate
#   * prediction drift (PSI) -- production score distribution vs. baseline
# ---------------------------------------------------------------------------


def label_drift(
    production_labels: Iterable[Any],
    *,
    reference_rate: float = 0.035,
) -> dict[str, float]:
    """Absolute label drift = |production fraud rate - reference|.

    Returns a small dict with ``production_rate``, ``reference_rate`` and
    ``absolute_drift`` so the caller can feed all three into the monitoring
    registry in one call.
    """
    labels = [int(bool(int(v))) for v in production_labels if v is not None]
    if not labels:
        return {
            "production_rate": 0.0,
            "reference_rate": float(reference_rate),
            "absolute_drift": 0.0,
            "n": 0,
        }
    production_rate = float(sum(labels)) / len(labels)
    return {
        "production_rate": production_rate,
        "reference_rate": float(reference_rate),
        "absolute_drift": abs(production_rate - float(reference_rate)),
        "n": len(labels),
    }


def prediction_distribution_drift(
    production_scores: Sequence[float] | np.ndarray,
    reference_scores: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 20,
) -> float:
    """PSI between two score distributions.

    Thin wrapper around :func:`population_stability_index` with a default
    bin count tuned for the [0, 1] probability range.
    """
    return population_stability_index(
        reference=reference_scores, current=production_scores, n_bins=n_bins
    )


__all__ = [
    "DriftDetector",
    "DriftDetectorConfig",
    "DriftReport",
    "DriftSeverity",
    "FeatureDrift",
    "categorical_psi",
    "chi_square",
    "evidently_html_report",
    "js_divergence",
    "ks_test",
    "label_drift",
    "population_stability_index",
    "prediction_distribution_drift",
]
