"""Open-loop prediction metrics, as computed for the paper's SSL figures.

For each episode, ``WorldModel.video_pred`` reconstructs the first ``context``
steps and imagines the rest from the recorded actions. Image error is reported
over all steps ("total"), the context steps and the imagined steps:

* MSE: summed squared error / number of pixels, pooled over episodes;
* SSIM (scikit-image, ``data_range`` = range of the true frame; 1.0 for a
  constant frame), averaged over frames.

``depth_space="legacy"`` scores the images in the space the decoders were trained
in (depth - 0.5), which is how the paper's numbers were produced. MSE does not
depend on the shift, SSIM does (its luminance term is not shift-invariant);
``"raw"`` adds the 0.5 back to both images and scores true depth.
"""

from typing import Dict

import numpy as np

SPLITS = ("total", "context", "imagination")


class ImaginationScores:
    """Accumulates open-loop MSE / SSIM over episodes, split into total, context and
    imagined steps (see the module docstring for the definitions).
    """

    def __init__(self, context: int, depth_space: str = "legacy", ssim: bool = True):
        assert depth_space in ("legacy", "raw"), depth_space
        self.context = context
        self.offset = 0.5 if depth_space == "raw" else 0.0
        self.use_ssim = ssim
        self.mse = {k: [0.0, 0] for k in SPLITS}
        self.ssim = {k: [0.0, 0] for k in SPLITS}

    def add(self, pred: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Score one ``video_pred`` output; returns this episode's metrics and adds it to
        the pooled totals.
        """
        true = pred["i_true"][0].astype(np.float32) + self.offset   # (T, H, W, 1)
        model = pred["i_model"][0].astype(np.float32) + self.offset
        ctx = self.context
        sq = (true - model) ** 2
        episode = {}
        for split, sl in (("total", slice(None)), ("context", slice(None, ctx)), ("imagination", slice(ctx, None))):
            part = sq[sl]
            self.mse[split][0] += float(part.sum())
            self.mse[split][1] += part.size
            episode[f"mse_{split}"] = float(part.mean()) if part.size else float("nan")
        if self.use_ssim:
            from skimage.metrics import structural_similarity
            per_frame = []
            for t in range(true.shape[0]):
                gt, pr = true[t, :, :, 0], model[t, :, :, 0]
                rng = float(gt.max() - gt.min())
                per_frame.append(structural_similarity(gt, pr, data_range=rng) if rng > 1e-6 else 1.0)
            for split, sl in (("total", slice(None)), ("context", slice(None, ctx)), ("imagination", slice(ctx, None))):
                vals = per_frame[sl]
                self.ssim[split][0] += float(sum(vals))
                self.ssim[split][1] += len(vals)
                episode[f"ssim_{split}"] = float(np.mean(vals)) if vals else float("nan")
        return episode

    def summary(self) -> Dict[str, float]:
        """Pooled metrics over every episode added so far (NaN for an empty split)."""
        out = {f"mse_{k}": (s / n if n else float("nan")) for k, (s, n) in self.mse.items()}
        if self.use_ssim:
            out.update({f"ssim_{k}": (s / n if n else float("nan")) for k, (s, n) in self.ssim.items()})
        return out
