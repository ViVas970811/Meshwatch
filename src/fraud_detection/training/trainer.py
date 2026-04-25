"""Training loop for :class:`fraud_detection.models.FraudHeteroGNN`.

Implementation-plan settings (page 7):

* **Loss:** Focal Loss (alpha=0.75, gamma=2.0)
* **Optimiser:** AdamW (lr=1e-3, weight_decay=1e-4)
* **Schedule:** cosine annealing over the full epoch budget
* **Sampler:** :class:`torch_geometric.loader.NeighborLoader` (15, 10, 5)
* **Batch size:** 4096 transactions
* **Early stopping:** patience 15 on validation AUPRC
* **Gradient clipping:** L2 norm 1.0
* **Tracking:** MLflow (params, metrics per-epoch, final artefacts)

The trainer is a single ``Trainer`` class that owns the model, optimiser,
scheduler, callbacks, loss, and an MLflow run. ``Trainer.fit(...)`` runs
the full loop and returns the trained model and an :class:`EvaluationResult`
on the validation set.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from fraud_detection.models.hetero_gnn import FraudHeteroGNN
from fraud_detection.models.losses import FocalLoss
from fraud_detection.training.callbacks import EarlyStopping, ModelCheckpoint
from fraud_detection.training.evaluator import EvaluationResult, evaluate_predictions
from fraud_detection.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    epochs: int = 100
    batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    neighbor_sampling: tuple[int, ...] = (15, 10, 5)
    num_workers: int = 0  # set to 4 in scripts/train.py for prod -- 0 for tests
    early_stop_patience: int = 15
    early_stop_metric: str = "val_auprc"  # informational; mode is fixed below
    cosine_eta_min: float = 1e-6
    log_every_n_epochs: int = 1
    seed: int = 42
    target_node_type: str = "transaction"
    device: str = "cpu"
    # Sampling mode -- "neighbor" needs pyg-lib/torch-sparse; "full_graph"
    # forwards the entire graph each epoch (fine for <= a few hundred K
    # transactions on CPU). "auto" falls back to "full_graph" if the
    # neighbor sampler's C extensions aren't installed.
    sampling: str = "auto"  # one of: auto | neighbor | full_graph
    # MLflow knobs (auto-skip if mlflow isn't installed at runtime)
    mlflow_enabled: bool = True
    mlflow_experiment: str = "fraud-detection-gnn"
    mlflow_run_name: str | None = None
    mlflow_extra_params: dict[str, Any] = field(default_factory=dict)


def _neighbor_sampler_available() -> bool:
    """Return True iff PyG can build a ``NeighborLoader`` on this install."""
    try:
        import pyg_lib  # noqa: F401

        return True
    except ImportError:
        pass
    try:
        import torch_sparse  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Mask helpers -- if the input HeteroData has no train/val split, build one
# chronologically from a time tensor.
# ---------------------------------------------------------------------------


def ensure_temporal_masks(
    data: HeteroData,
    *,
    time_tensor: torch.Tensor | None = None,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    target_node_type: str = "transaction",
) -> HeteroData:
    """Idempotently set ``train_mask``/``val_mask``/``test_mask`` on the target node.

    If valid masks already exist (no all-True / all-False), leaves them
    alone. Otherwise sorts target nodes by ``time_tensor`` (or by index if
    none given -- assumes the upstream pipeline already sorted by time)
    and slices 60/20/20.
    """
    target = data[target_node_type]
    n = target.num_nodes
    has_valid = (
        hasattr(target, "train_mask")
        and hasattr(target, "val_mask")
        and target.train_mask.sum() > 0
        and target.val_mask.sum() > 0
        and target.train_mask.sum() < n
    )
    if has_valid:
        return data

    log.info("building_temporal_masks", n=n, has_time_tensor=time_tensor is not None)
    if time_tensor is None:
        order = torch.arange(n)
    else:
        if time_tensor.shape[0] != n:
            msg = f"time_tensor length {time_tensor.shape[0]} != n_nodes {n}"
            raise ValueError(msg)
        order = torch.argsort(time_tensor, stable=True)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    target.train_mask = train_mask
    target.val_mask = val_mask
    target.test_mask = test_mask
    return data


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """End-to-end GNN training loop.

    Notes on memory: ``NeighborLoader`` materialises a fresh sub-graph per
    batch. With sampling fanout (15, 10, 5) and batch size 4096 each batch
    pulls roughly 50K transactions worth of neighborhood -- well within
    16GB. For full-graph eval we just call ``model(data)`` once.
    """

    def __init__(
        self,
        model: FraudHeteroGNN,
        config: TrainerConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or TrainerConfig()
        torch.manual_seed(self.config.seed)
        self._device = torch.device(self.config.device)
        self.model.to(self._device)

        # Resolve "auto" sampling: prefer NeighborLoader if available, else full-graph.
        if self.config.sampling == "auto":
            self._sampling = "neighbor" if _neighbor_sampler_available() else "full_graph"
            log.info(
                "sampling_resolved",
                chosen=self._sampling,
                neighbor_available=self._sampling == "neighbor",
            )
        elif self.config.sampling in ("neighbor", "full_graph"):
            self._sampling = self.config.sampling
        else:
            msg = f"Invalid sampling mode '{self.config.sampling}'"
            raise ValueError(msg)

        self.criterion = FocalLoss(
            alpha=self.config.focal_alpha,
            gamma=self.config.focal_gamma,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=self.config.cosine_eta_min,
        )
        self.early_stopping = EarlyStopping(patience=self.config.early_stop_patience, mode="max")
        self.checkpoint = ModelCheckpoint(monitor="val_auprc", mode="max")
        self.history: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _make_loader(
        self,
        data: HeteroData,
        node_indices: torch.Tensor,
        *,
        shuffle: bool,
    ) -> NeighborLoader:
        return NeighborLoader(
            data,
            num_neighbors=list(self.config.neighbor_sampling),
            batch_size=self.config.batch_size,
            input_nodes=(self.config.target_node_type, node_indices),
            shuffle=shuffle,
            num_workers=self.config.num_workers,
        )

    # ------------------------------------------------------------------
    # Training and evaluation primitives
    # ------------------------------------------------------------------

    def _train_one_epoch_neighbor(self, loader: NeighborLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        for batch in loader:
            batch = batch.to(self._device)
            self.optimizer.zero_grad(set_to_none=True)
            # NeighborLoader puts the seed nodes first; ``batch_size`` of them.
            tgt = batch[self.config.target_node_type]
            seed_size = int(getattr(tgt, "batch_size", tgt.num_nodes))
            seed_idx = torch.arange(seed_size, device=self._device)
            logits = self.model(batch, target_indices=seed_idx)
            y = tgt.y[:seed_size].to(self._device).float()
            loss = self.criterion(logits, y)
            loss.backward()
            if self.config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()
            total_loss += loss.item() * seed_size
            total_samples += seed_size
        return total_loss / max(total_samples, 1)

    def _train_one_epoch_full(self, data: HeteroData, train_idx: torch.Tensor) -> float:
        """Full-graph training: forward all transactions, slice loss to the train mask.

        Used as a fallback when NeighborLoader's C extensions aren't
        available (CPU-only installs without ``pyg-lib``/``torch-sparse``).
        For graphs of <~500K transactions on a 16GB CPU this is faster
        than batched neighbor sampling anyway -- one forward per epoch
        instead of N/4096 sub-graph forwards.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits_all = self.model(data)
        y_all = data[self.config.target_node_type].y.float().to(self._device)
        # Slice to train set only -- val/test logits are still computed
        # by the forward but their gradients are masked out.
        train_idx_dev = train_idx.to(self._device)
        loss = self.criterion(logits_all[train_idx_dev], y_all[train_idx_dev])
        loss.backward()
        if self.config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def _evaluate_split(
        self,
        data: HeteroData,
        node_indices: torch.Tensor,
    ) -> tuple[EvaluationResult, np.ndarray, np.ndarray]:
        """Score one split via full-graph forward then slice."""
        self.model.eval()
        data = data.to(self._device)
        logits_all = self.model(data)  # (N_target,)
        probs_all = torch.sigmoid(logits_all).cpu().numpy()
        y_all = data[self.config.target_node_type].y.cpu().numpy().astype(np.int8)
        idx = node_indices.cpu().numpy()
        y = y_all[idx]
        p = probs_all[idx]
        result = evaluate_predictions(y, p)
        return result, y, p

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, data: HeteroData) -> dict[str, Any]:
        """Run the full training loop.

        Returns a dict with keys: ``model``, ``history``, ``val_result``,
        ``best_epoch``, ``best_val_auprc``.
        """
        target = data[self.config.target_node_type]
        if not hasattr(target, "train_mask") or not hasattr(target, "val_mask"):
            msg = (
                "Target node type must have train_mask and val_mask; "
                "call ensure_temporal_masks(data) first."
            )
            raise ValueError(msg)
        train_idx = target.train_mask.nonzero(as_tuple=True)[0]
        val_idx = target.val_mask.nonzero(as_tuple=True)[0]
        if len(train_idx) == 0 or len(val_idx) == 0:
            msg = f"Empty train/val masks (train={len(train_idx)}, val={len(val_idx)})"
            raise ValueError(msg)

        train_loader = (
            self._make_loader(data, train_idx, shuffle=True)
            if self._sampling == "neighbor"
            else None
        )

        # MLflow context (lazy import keeps test envs without mlflow happy)
        mlflow_ctx = self._maybe_start_mlflow()
        log.info(
            "training_start",
            n_train=len(train_idx),
            n_val=len(val_idx),
            n_params=self.model.n_parameters(),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            device=str(self._device),
            sampling=self._sampling,
        )

        best_val: EvaluationResult | None = None
        try:
            for epoch in range(self.config.epochs):
                t0 = time.time()
                if self._sampling == "neighbor":
                    train_loss = self._train_one_epoch_neighbor(train_loader)  # type: ignore[arg-type]
                else:
                    train_loss = self._train_one_epoch_full(data, train_idx)
                val_result, _, _ = self._evaluate_split(data, val_idx)
                self.scheduler.step()
                lr_now = self.optimizer.param_groups[0]["lr"]

                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_auprc": val_result.auprc,
                    "val_auroc": val_result.auroc,
                    "val_log_loss": val_result.log_loss,
                    "lr": lr_now,
                    "epoch_seconds": time.time() - t0,
                }
                self.history.append(row)
                self._mlflow_log_metrics(mlflow_ctx, row, step=epoch)

                if epoch % self.config.log_every_n_epochs == 0 or epoch == self.config.epochs - 1:
                    log.info(
                        "epoch_end",
                        **{k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()},
                    )

                improved = self.checkpoint.step(
                    val_result.auprc,
                    epoch=epoch,
                    model=self.model,
                    extra={"train_loss": train_loss},
                )
                if improved:
                    best_val = val_result
                if self.early_stopping.step(val_result.auprc, epoch=epoch):
                    log.info(
                        "early_stop",
                        epoch=epoch,
                        best_epoch=self.checkpoint.best_epoch,
                        best_val_auprc=self.checkpoint.best_score,
                    )
                    break

            # Restore best weights before returning.
            self.checkpoint.restore(self.model)
            log.info(
                "training_complete",
                best_epoch=self.checkpoint.best_epoch,
                best_val_auprc=self.checkpoint.best_score,
                stopped_early=self.early_stopping.should_stop,
            )
        finally:
            self._maybe_end_mlflow(mlflow_ctx)

        return {
            "model": self.model,
            "history": self.history,
            "val_result": best_val,
            "best_epoch": self.checkpoint.best_epoch,
            "best_val_auprc": self.checkpoint.best_score,
        }

    # ------------------------------------------------------------------
    # MLflow plumbing (best-effort, never breaks training)
    # ------------------------------------------------------------------

    def _maybe_start_mlflow(self):
        if not self.config.mlflow_enabled:
            return None
        try:
            import mlflow

            mlflow.set_experiment(self.config.mlflow_experiment)
            run = mlflow.start_run(run_name=self.config.mlflow_run_name)
            params = {
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "grad_clip_norm": self.config.grad_clip_norm,
                "focal_alpha": self.config.focal_alpha,
                "focal_gamma": self.config.focal_gamma,
                "neighbor_sampling": "_".join(map(str, self.config.neighbor_sampling)),
                "n_params": self.model.n_parameters(),
                "device": str(self._device),
                **self.config.mlflow_extra_params,
            }
            mlflow.log_params(params)
            return run
        except Exception as exc:
            log.warning("mlflow_disabled", error=str(exc))
            return None

    def _mlflow_log_metrics(self, run, metrics: dict[str, float], *, step: int) -> None:
        if run is None:
            return
        try:
            import mlflow

            for k, v in metrics.items():
                if isinstance(v, (int, float)) and k != "epoch":
                    mlflow.log_metric(k, float(v), step=step)
        except Exception as exc:
            log.debug("mlflow_log_failed", error=str(exc))

    def _maybe_end_mlflow(self, run) -> None:
        if run is None:
            return
        try:
            import mlflow

            mlflow.end_run()
        except Exception:
            pass


__all__ = ["Trainer", "TrainerConfig", "ensure_temporal_masks"]
