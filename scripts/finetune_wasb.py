"""Fine-tune WASB HRNet on a repo-side corpus (spec §4.3 step 3).

The vendored WASB trainer is CUDA-locked and Hydra-configured, so this CLI
drives the repo-side harness in ``src.utils.ball_finetune_train`` instead:
builds a training + validation split from the corpus manifest's ``train``
clips (random ``val_frac`` split, seed 0), evaluates hit-rate on the
manifest's ``holdout`` clips each epoch, and checkpoints the best model by
holdout hit-rate (falling back to validation hit-rate when there is no
holdout).

Usage:
    python scripts/finetune_wasb.py \
        --corpus-root output/ball_finetune_corpus \
        --run-dir output/ball_finetune_runs/run1 \
        --epochs 30 --batch 4 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset, random_split  # noqa: E402

from src.utils.ball_finetune_train import (  # noqa: E402
    FinetuneDataset,
    evaluate_hit_rate,
    wbce_loss,
)
from src.utils.wasb_ball_detector import load_wasb_model  # noqa: E402

_DEFAULT_INIT = (
    "third_party/wasb_sbdt/pretrained_weights/wasb_soccer_best.pth.tar"
)


def _pick_device(requested: str) -> str:
    """Resolve ``'auto' | 'cpu' | 'cuda' | 'mps'`` for training.

    Deliberately NOT the detector's conservative cpu-only default:
    training wants to use MPS on macOS when available.
    """
    want = (requested or "auto").strip().lower()
    if want == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return want


def _limit(dataset, limit_samples: int | None):
    if limit_samples is None or limit_samples <= 0 or limit_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit_samples)))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-root", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--init", type=str, default=_DEFAULT_INIT)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--limit-samples", type=int, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    corpus_root: Path = args.corpus_root
    manifest = json.loads((corpus_root / "manifest.json").read_text())
    train_clips = list(manifest.get("train", []))
    holdout_clips = list(manifest.get("holdout", []))

    device = _pick_device(args.device)
    print(f"device={device}", flush=True)

    full_train_ds = FinetuneDataset(corpus_root, train_clips)
    full_train_ds = _limit(full_train_ds, args.limit_samples)

    n_total = len(full_train_ds)
    n_val = int(round(n_total * args.val_frac)) if args.val_frac > 0 else 0
    n_train = n_total - n_val
    if n_val > 0:
        generator = torch.Generator().manual_seed(0)
        train_subset, val_subset = random_split(
            full_train_ds, [n_train, n_val], generator=generator,
        )
    else:
        train_subset, val_subset = full_train_ds, None

    holdout_ds = FinetuneDataset(corpus_root, holdout_clips) if holdout_clips else None

    train_loader = DataLoader(
        train_subset, batch_size=args.batch, shuffle=True,
        num_workers=0, pin_memory=False,
    )

    model, device = load_wasb_model(args.init, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_metric = -1.0
    best_epoch: int | None = None
    best_metric_name = "holdout_hit_rate" if holdout_ds is not None else "val_hit_rate"

    def _write_history() -> None:
        (run_dir / "history.json").write_text(json.dumps({
            "best_metric_name": best_metric_name,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "epochs": history,
        }, indent=2))

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            # HRNet returns {scale: tensor} (out_scales=[0]).
            pred = out[0] if isinstance(out, dict) else out
            loss = wbce_loss(pred, y)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))
        train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0

        val_hit_rate = None
        if val_subset is not None and len(val_subset) > 0:
            val_hit_rate = evaluate_hit_rate(model, val_subset, device)

        holdout_hit_rate = None
        if holdout_ds is not None and len(holdout_ds) > 0:
            holdout_hit_rate = evaluate_hit_rate(model, holdout_ds, device)

        metric = holdout_hit_rate if holdout_hit_rate is not None else val_hit_rate
        wall_time = time.time() - t0

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_hit_rate": val_hit_rate,
            "holdout_hit_rate": holdout_hit_rate,
            "wall_time_s": wall_time,
        }
        history.append(epoch_record)
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"val_hit_rate={val_hit_rate} holdout_hit_rate={holdout_hit_rate} "
            f"wall_time_s={wall_time:.2f}",
            flush=True,
        )

        # Save the last epoch's weights every epoch, so a killed run (or a
        # small noisy holdout that happens to pick an early epoch as
        # "best") never loses the strongest late-training weights.
        torch.save(
            {"model_state_dict": model.state_dict()},
            run_dir / "last.pth.tar",
        )

        if metric is not None and metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            torch.save(
                {"model_state_dict": model.state_dict()},
                run_dir / "best.pth.tar",
            )

        # Write history after EVERY epoch so a killed run keeps its
        # progress instead of losing it all at exit.
        _write_history()

    if not (run_dir / "best.pth.tar").exists():
        # No val/holdout signal available at all — still persist a
        # checkpoint so the CLI's contract (best.pth.tar exists) holds.
        torch.save({"model_state_dict": model.state_dict()}, run_dir / "best.pth.tar")

    _write_history()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
