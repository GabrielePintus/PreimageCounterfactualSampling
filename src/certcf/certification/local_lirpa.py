"""Strategy adapter for the existing local LiRPA implementation."""
import numpy as np
import torch

from .backend import BuildRequest, BuildResult
from certcf.geometry.regions import regions_from_legacy


class LocalLiRPABackend:
    """Preserve batching/shrinkage and expose its output as generic regions.

    ``preimage`` is an optional existing session used by the legacy facade.
    Bounds options are those of PreimageApproximation.compute_all_bounds.
    """
    def __init__(self, *, device="cpu", preimage=None, **bounds_options):
        self.device = device
        self.preimage = preimage
        self.bounds_options = bounds_options
        self.legacy_bounds = None

    def build(self, request: BuildRequest) -> BuildResult:
        from .lirpa import PreimageApproximation
        if not request.domains:
            return BuildResult((), request.class_labels, request.input_shape)
        norms, centers, radii, labels = set(), [], [], []
        source_ids = {label: [] for label in request.class_labels}
        for domain in request.domains:
            g, ball = domain.geometry, domain.geometry.ball
            if ball is None or len(g.b) or not np.array_equal(g.lower, ball.center - ball.radius) or not np.array_equal(g.upper, ball.center + ball.radius):
                raise ValueError("LocalLiRPA requires unmodified Lp-ball input domains")
            norms.add(ball.p)
            centers.append(ball.center)
            radii.append(ball.radius)
            labels.append(domain.target_class)
            source_ids[domain.target_class].append(domain.domain_id)
        if len(norms) != 1:
            raise ValueError("LocalLiRPA batching requires a common certification norm")
        norm = norms.pop()
        preimage = self.preimage
        if preimage is None:
            centers_array = np.asarray(centers)
            if not np.array_equal(centers_array, centers_array.astype(np.float32).astype(np.float64)):
                raise ValueError("LocalLiRPA domain centers must be exactly representable in float32")
            if set(labels) != set(request.class_labels):
                raise ValueError("LocalLiRPA currently requires domains for every model class")
            if tuple(sorted(request.class_labels)) != request.class_labels:
                raise ValueError("LocalLiRPA requires sorted labels in model-output order")
            dataset = torch.utils.data.TensorDataset(
                torch.tensor(np.asarray(centers), dtype=torch.float32).reshape(-1, *request.input_shape),
                torch.tensor(labels),
            )
            preimage = PreimageApproximation(request.model, dataset, torch.device(self.device),
                                            cnn=len(request.input_shape) == 3, model_input_shape=request.input_shape)
        else:
            stored_x, stored_y = preimage.dataset.tensors
            if (preimage.model is not request.model
                    or tuple(preimage.class_labels) != request.class_labels
                    or not np.array_equal(stored_x.detach().cpu().numpy().reshape(len(centers), -1), np.asarray(centers))
                    or not np.array_equal(stored_y.detach().cpu().numpy(), np.asarray(labels))):
                raise ValueError("Existing LiRPA session does not match the requested model/domains")
        options = dict(self.bounds_options)
        options.update(norm=norm, eps_array=np.asarray(radii), classification_margin=request.classification_margin)
        bounds = preimage.compute_all_bounds(**options)
        # The inclusion requirement belongs to this strategy, not every atlas.
        for label, values in bounds.items():
            keep = np.asarray(values.get("adaptive_eps_center_certified", np.ones(len(values["X"]), dtype=bool)), dtype=bool)
            count = len(keep)
            values["attempted_anchor_count"] = count
            values["uncertified_anchor_count"] = int((~keep).sum())
            source_ids[label] = [s for s, k in zip(source_ids[label], keep) if k]
            for key, value in list(values.items()):
                if isinstance(value, np.ndarray) and value.ndim and len(value) == count:
                    values[key] = value[keep]
        self.legacy_bounds = bounds
        from dataclasses import replace
        regions = regions_from_legacy(bounds, norm, request.classification_margin)
        counters = dict.fromkeys(request.class_labels, 0)
        mapped = []
        for r in regions:
            idx = counters[r.target_class]
            mapped.append(replace(r, source_domain_id=source_ids[r.target_class][idx]))
            counters[r.target_class] += 1
        return BuildResult(tuple(mapped), request.class_labels, request.input_shape,
                           diagnostics={"backend": "local_lirpa"})
