"""Linear probe pre-fit on a frozen encoder (preprint "probe head").

The preprint initialises the linear instance head of the probehead / ftALL
regimes from a logistic regression fitted on ego-graph embeddings of the
frozen (BGRL-pretrained) encoder: every ego-graph inherits the label of its
region, a ``StandardScaler`` + ``LogisticRegression`` is fitted on the training
regions, and the scaler is folded into the weights so the head acts on raw
embeddings. For a binary task the class-0 row stays zero (the logistic decision
function becomes the class-1 logit); for more classes the multinomial
coefficients are used directly.

Nothing here is specific to the preprint sizes: the number of roots per region,
the seed, the regulariser and the ego-graph settings are all arguments (the CLI
reads them from ``configs/probe.yaml`` and from ``data.sampler``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

__all__ = [
    "ProbeEmbeddings",
    "ProbeFit",
    "attach_datamodule",
    "collect_frozen_embeddings",
    "fit_probe",
    "fold_probe_into_linear_head",
    "probe_sampler_config",
    "region_mean_auroc",
    "save_probe_head",
    "load_probe_head",
]


@dataclass
class ProbeEmbeddings:
    """Ego-graph embeddings with their region ids and inherited region labels."""

    embeddings: np.ndarray  # (n_instances, dim)
    labels: np.ndarray  # (n_instances,) int
    region_ids: np.ndarray  # (n_instances,) str
    label_by_region: Dict[str, int] = field(default_factory=dict)


@dataclass
class ProbeFit:
    weight: torch.Tensor  # (num_classes, dim), scaler folded in
    bias: torch.Tensor  # (num_classes,)
    metrics: Dict[str, float] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)


def attach_datamodule(model: Any, datamodule: Any, trainer: Any) -> None:
    """Bind a datamodule to a model outside ``Trainer.fit`` so ``setup`` can inspect it.

    ``SupervisedModule.setup`` infers the encoder input width and the categorical
    vocabulary from ``trainer.datamodule``; this mirrors what Lightning does at
    the start of ``fit``/``test``.
    """
    trainer._data_connector.attach_data(model, datamodule=datamodule)
    model.trainer = trainer
    datamodule.trainer = trainer


def probe_sampler_config(
    base: Any,
    *,
    roots_per_unit: Optional[int],
    seed: int,
    batch_size: int,
) -> Any:
    """Evaluation sampler for the probe: fixed uniform roots per region, no bagging.

    ``base`` is the datamodule's training ``SamplerConfig``; its ego-graph settings
    (depth, neighbour limit, radius cap, interior seeds) are kept.
    """
    from grass_mil.data.components.samplers import SamplerConfig

    if base is None or getattr(base, "name", "identity") == "identity":
        raise ValueError(
            "The probe needs an ego-graph sampler (data.sampler.name=shadow_custom with "
            "runtime.enabled=true, e.g. data/sampler=ego_graph_mil)."
        )
    runtime = dict(base.runtime or {})
    runtime.update(
        {
            "enabled": True,
            "subsample_fraction": 1.0,
            "subsample_count": int(roots_per_unit) if roots_per_unit else None,
            "subsample_seed": None,
            "fixed_per_unit_seed": int(seed),
            "proportional_root_sampling": False,
            "unit_as_bag": False,
            "merge_units": False,
            "shuffle_subgraphs": False,
            "subgraph_batch_size": int(batch_size),
        }
    )
    return SamplerConfig(name=base.name, kwargs=dict(base.kwargs or {}), runtime=runtime)


def _region_ids_of(batch: Any, n: int) -> List[str]:
    from grass_mil.models.training.bagging import extract_bag_ids

    ids = extract_bag_ids(batch, bag_key="region_id", bag_fallback_key="sample_id")
    if len(ids) == 1 and n > 1:
        ids = ids * n
    if len(ids) != n:
        raise ValueError(f"Batch carries {len(ids)} region ids for {n} ego-graphs.")
    return [str(v) for v in ids]


def _labels_of(batch: Any, n: int, target_columns: Optional[Sequence[str]]) -> np.ndarray:
    from grass_mil.models.training.bagging import select_target_columns

    if not hasattr(batch, "graph_y"):
        raise ValueError("Batch has no graph_y; the probe needs region labels.")
    y = select_target_columns(batch.graph_y.float(), batch, target_columns)
    y = y.view(y.shape[0], -1)[:, 0]
    if y.shape[0] == 1 and n > 1:
        y = y.expand(n)
    return np.rint(y.detach().cpu().numpy()).astype(int)


@torch.no_grad()
def collect_frozen_embeddings(
    model: Any,
    dataset: Any,
    *,
    sampler_config: Any,
    target_columns: Optional[Sequence[str]] = None,
    device: Optional[torch.device | str] = None,
    num_workers: int = 0,
) -> ProbeEmbeddings:
    """Encode ``roots_per_unit`` ego-graphs per region with the frozen encoder."""
    from grass_mil.data.components.samplers import get_sampler_strategy

    strategy = get_sampler_strategy(sampler_config)
    loader = strategy.build_dataset_loader(
        dataset=dataset, batch_size=1, num_workers=num_workers, pin_memory=False, shuffle=False
    )
    device = torch.device(device) if device is not None else next(model.parameters()).device
    was_training = model.training
    model.eval()
    embs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    rids: List[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        _, graph_emb, _ = model._forward_instances(batch)
        n = int(graph_emb.shape[0])
        embs.append(graph_emb.detach().float().cpu().numpy())
        labels.append(_labels_of(batch, n, target_columns))
        rids.append(np.asarray(_region_ids_of(batch, n), dtype=object))
    if was_training:
        model.train()
    if not embs:
        raise ValueError("No ego-graphs were produced; is the dataset split empty?")
    e = np.concatenate(embs)
    y = np.concatenate(labels)
    r = np.concatenate(rids)
    by_region = {str(k): int(v) for k, v in zip(r, y)}
    return ProbeEmbeddings(embeddings=e, labels=y, region_ids=r, label_by_region=by_region)


def region_mean_auroc(prob: np.ndarray, region_ids: np.ndarray, label_by_region: Dict[str, int]):
    """AUROC of the per-region mean probability (how the preprint scores the probe)."""
    from sklearn.metrics import roc_auc_score

    regions = np.unique(region_ids)
    means = np.array([float(prob[region_ids == r].mean()) for r in regions])
    y = np.array([label_by_region[str(r)] for r in regions])
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, means))


def fold_probe_into_linear_head(
    coef: np.ndarray,
    intercept: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    *,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold ``StandardScaler`` + logistic coefficients into ``Linear(dim, num_classes)``.

    Binary (``coef`` has one row): the class-1 row carries the decision function and
    the class-0 row is zero, so ``softmax`` over the two logits equals the logistic
    probability. Multinomial: rows map one-to-one onto classes.
    """
    coef = np.asarray(coef, dtype=np.float64)
    intercept = np.asarray(intercept, dtype=np.float64).reshape(-1)
    w = coef / scaler_scale[None, :]
    b = intercept - (coef * scaler_mean[None, :] / scaler_scale[None, :]).sum(axis=1)
    dim = coef.shape[1]
    W = torch.zeros(num_classes, dim, dtype=torch.float32)
    B = torch.zeros(num_classes, dtype=torch.float32)
    if coef.shape[0] == 1:
        if num_classes < 2:
            raise ValueError("A binary probe needs a 2-logit head (graph_head.output_dim=2).")
        W[1] = torch.from_numpy(w[0].astype(np.float32))
        B[1] = float(b[0])
    else:
        if coef.shape[0] != num_classes:
            raise ValueError(
                f"Probe has {coef.shape[0]} classes but the head has {num_classes} outputs."
            )
        W[:] = torch.from_numpy(w.astype(np.float32))
        B[:] = torch.from_numpy(b.astype(np.float32))
    return W, B


def fit_probe(
    train: ProbeEmbeddings,
    *,
    num_classes: int,
    C: float = 1.0,
    class_weight: Optional[str] = "balanced",
    max_iter: int = 3000,
    eval_sets: Optional[Dict[str, ProbeEmbeddings]] = None,
) -> ProbeFit:
    """Fit the scaler + logistic regression and fold it into head weights."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(train.labels)) < 2:
        raise ValueError("The probe training set has a single class.")
    scaler = StandardScaler().fit(train.embeddings)
    clf = LogisticRegression(max_iter=int(max_iter), C=float(C), class_weight=class_weight)
    clf.fit(scaler.transform(train.embeddings), train.labels)
    W, B = fold_probe_into_linear_head(
        clf.coef_, clf.intercept_, scaler.mean_, scaler.scale_, num_classes=num_classes
    )
    metrics: Dict[str, float] = {"n_train_instances": float(len(train.labels))}
    for name, emb in (eval_sets or {}).items():
        prob = clf.predict_proba(scaler.transform(emb.embeddings))
        if prob.shape[1] == 2:
            p1 = prob[:, 1]
            if len(np.unique(emb.labels)) >= 2:
                metrics[f"{name}_instance_auroc"] = float(roc_auc_score(emb.labels, p1))
            metrics[f"{name}_region_auroc"] = region_mean_auroc(
                p1, emb.region_ids, emb.label_by_region
            )
        metrics[f"n_{name}_instances"] = float(len(emb.labels))
    settings = {"C": float(C), "class_weight": class_weight, "max_iter": int(max_iter)}
    return ProbeFit(weight=W, bias=B, metrics=metrics, settings=settings)


def save_probe_head(
    fit: ProbeFit, path: str | Path, extra: Optional[Dict[str, Any]] = None
) -> Path:
    """Write the folded head as ``graph_head.net.0.{weight,bias}`` (+ metrics)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "graph_head.net.0.weight": fit.weight.clone(),
        "graph_head.net.0.bias": fit.bias.clone(),
        "metrics": dict(fit.metrics),
        "settings": dict(fit.settings),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_probe_head(path: str | Path) -> Dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)
