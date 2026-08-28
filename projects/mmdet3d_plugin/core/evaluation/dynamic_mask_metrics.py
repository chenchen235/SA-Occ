"""Streaming binary semantic metrics for sparse dynamic BEV masks."""
import numpy as np


class DynamicMaskMetrics:
    """Accumulate calibrated probabilities without retaining per-frame maps."""
    def __init__(self, bins=4096):
        self.bins = bins
        self.pos_hist = np.zeros(bins, dtype=np.int64)
        self.neg_hist = np.zeros(bins, dtype=np.int64)

    def update(self, probability, semantics, mask_camera):
        if hasattr(probability, 'detach'):
            probability = probability.detach().float().cpu().numpy()
        p = np.asarray(probability, dtype=np.float32).squeeze()
        target = ((semantics >= 0) & (semantics <= 10)).any(axis=-1).T
        valid = np.asarray(mask_camera).astype(bool).any(axis=-1).T
        if p.shape != target.shape:
            raise ValueError('dynamic probability {} does not match BEV GT {}'.format(
                p.shape, target.shape))
        p = np.clip(p[valid], 0., 1.)
        y = target[valid]
        index = np.minimum((p * self.bins).astype(np.int64), self.bins - 1)
        self.pos_hist += np.bincount(index[y], minlength=self.bins)
        self.neg_hist += np.bincount(index[~y], minlength=self.bins)

    def _at(self, threshold):
        start = int(np.ceil(threshold * self.bins))
        tp = int(self.pos_hist[start:].sum())
        fp = int(self.neg_hist[start:].sum())
        fn = int(self.pos_hist[:start].sum())
        tn = int(self.neg_hist[:start].sum())
        eps = 1e-12
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                    precision=precision, recall=recall,
                    iou=tp / (tp + fp + fn + eps),
                    dice=2 * tp / (2 * tp + fp + fn + eps))

    def summary(self):
        pos, neg = int(self.pos_hist.sum()), int(self.neg_hist.sum())
        tp = np.cumsum(self.pos_hist[::-1], dtype=np.float64)
        fp = np.cumsum(self.neg_hist[::-1], dtype=np.float64)
        recall = tp / max(pos, 1)
        precision = tp / np.maximum(tp + fp, 1)
        ap = float(np.sum((recall - np.r_[0., recall[:-1]]) * precision))
        tpr = np.r_[0., recall, 1.]
        fpr = np.r_[0., fp / max(neg, 1), 1.]
        auc = float(np.trapz(tpr, fpr))
        at = self._at(0.5)
        return dict(
            dynamic_iou=round(at['iou'], 6),
            dynamic_dice=round(at['dice'], 6),
            dynamic_ap=round(ap, 6),
            dynamic_precision=round(at['precision'], 6),
            dynamic_recall=round(at['recall'], 6),
            dynamic_roc_auc=round(auc, 6),
            threshold=0.5,
            valid_bev_cells=pos + neg,
            dynamic_positive_cells=pos,
        )
