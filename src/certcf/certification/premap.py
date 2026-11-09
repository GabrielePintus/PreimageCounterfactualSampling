"""Optional, process-isolated adapter for the pinned original PREMAP backend.

The external worker calls PREMAP's input-split algorithm, not a replacement
implementation. Only numeric data cross the process boundary.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile

import numpy as np
import torch
from torch import nn

from .backend import BuildRequest, BuildResult
from certcf.geometry.regions import CertifiedRegion, RegionGeometry


PREMAP_REVISION = "0bbbdb99d5813549d7d0656a05a99073eb4a48ed"
SAMPLING_PROTOCOL = "input-coverage-v2"


@dataclass(frozen=True)
class PremapConfig:
    python: str
    repository: str
    timeout_s: float = 120.0  # whole request, including worker startup
    max_regions: int = 128   # per input domain; reported separately from anchors
    samples: int | None = None  # compatibility alias for coverage_samples
    coverage_threshold: float = 0.9
    seed: int = 0
    device: str = "cpu"
    optimize_bounds: bool = True  # False enables a plain-CROWN matched-domain control.
    coverage_samples: int | None = None

    def __post_init__(self):
        for count in (self.samples, self.coverage_samples):
            if count is not None and (type(count) is not int or count < 2):
                raise ValueError("coverage_samples (or samples alias) must be an integer >= 2")
        if self.samples is not None and self.coverage_samples is not None and self.samples != self.coverage_samples:
            raise ValueError("samples and coverage_samples disagree; use coverage_samples")
        count = self.coverage_samples if self.coverage_samples is not None else self.samples
        count = 2000 if count is None else count
        object.__setattr__(self, "samples", count)
        object.__setattr__(self, "coverage_samples", count)
        if not np.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        if type(self.max_regions) is not int or self.max_regions < 1:
            raise ValueError("max_regions must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not 0 < self.coverage_threshold <= 1:
            raise ValueError("coverage_threshold must be in (0, 1]")
        if type(self.optimize_bounds) is not bool:
            raise ValueError("optimize_bounds must be boolean")
        if not self.optimize_bounds and self.max_regions != 1:
            raise ValueError("The plain-CROWN control requires max_regions=1")

    def sampling_settings(self):
        """Effective sampling contract; optimizer settings are fixed upstream.

        Do not expose fake optimizer knobs: changing them would require an
        explicitly different algorithm/adapter, not just a configuration value.
        """
        return {"protocol": SAMPLING_PROTOCOL,
                "coverage_samples": self.coverage_samples,
                "coverage_samples_per_half": self.coverage_samples // 2,
                "coverage_seed_base": self.seed,
                "coverage_seed_policy": "seed + domain index; reset at each coverage call",
                "optimization_samples": 1000, "optimization_seed": 1,
                "optimization_policy": "fixed by pinned upstream; only used for optimized ReLU bounds"}


def export_sequential(model, input_dim, output_dim):
    """Explicit sequential graph only: never guess an arbitrary module's forward."""
    if type(model) is not nn.Sequential:
        raise TypeError("PREMAP v1 requires an explicit nn.Sequential; pass the actual Linear/ReLU inference graph")
    arrays, layers = {}, []
    width = input_dim
    for layer in model:
        if type(layer) is nn.Linear:
            if layer.in_features != width:
                raise ValueError("Sequential model dimensions do not match request")
            prefix = f"layer{len(layers)}"
            arrays[prefix + "_weight"] = layer.weight.detach().cpu().numpy().copy()
            arrays[prefix + "_bias"] = (layer.bias.detach().cpu().numpy().copy()
                                        if layer.bias is not None else np.zeros(layer.out_features, dtype=np.float32))
            if any(a.dtype != np.float32 or not np.isfinite(a).all() for k, a in arrays.items() if k.startswith(prefix)):
                raise ValueError("PREMAP v1 exports finite float32 weights only")
            layers.append({"type": "linear", "prefix": prefix})
            width = layer.out_features
        elif type(layer) is nn.ReLU:
            layers.append({"type": "relu"})
        elif type(layer) in (nn.Identity, nn.Dropout):
            if isinstance(layer, nn.Dropout) and layer.training:
                raise ValueError("Export the inference model in eval mode")
        else:
            raise TypeError(f"Unsupported PREMAP model layer: {type(layer).__name__}")
    if width != output_dim or not layers or layers[-1]["type"] != "linear":
        raise ValueError("PREMAP requires a final linear logit layer with the specified output dimension")
    return layers, arrays


class PremapBackend:
    def __init__(self, config: PremapConfig):
        self.config = config

    def build(self, request: BuildRequest) -> BuildResult:
        if len(request.input_shape) != 1 or len(request.class_labels) != 2:
            raise ValueError("PREMAP v1 supports binary, continuous tabular requests only")
        if request.classification_margin <= 0:
            raise ValueError("PREMAP requires a positive classification margin to exclude ties")
        for domain in request.domains:
            g = domain.geometry
            if g.ball is not None or len(g.b) or np.any(g.lower >= g.upper):
                raise ValueError("PREMAP input splitting requires explicit full-dimensional boxes, not balls or clipped domains")
            if not np.array_equal(g.lower, g.lower.astype(np.float32).astype(np.float64)) or not np.array_equal(g.upper, g.upper.astype(np.float32).astype(np.float64)):
                raise ValueError("PREMAP box endpoints must be exactly representable in float32; choose them explicitly")
        layers, arrays = export_sequential(request.model, request.input_shape[0], len(request.class_labels))
        if not request.domains:
            return BuildResult((), request.class_labels, request.input_shape,
                               diagnostics={"backend": "premap", "sampling": self.config.sampling_settings()})
        repository = Path(self.config.repository).resolve()
        revision = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        if revision != PREMAP_REVISION:
            raise ValueError(f"PREMAP adapter requires revision {PREMAP_REVISION}, found {revision}")
        dirty = subprocess.run(["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout
        if dirty.strip():
            raise ValueError("PREMAP tracked source must be unmodified for reproducible certification")
        with tempfile.TemporaryDirectory(prefix="certcf-premap-") as temporary:
            root = Path(temporary)
            domains = []
            for i, domain in enumerate(request.domains):
                arrays[f"lower{i}"] = domain.geometry.lower
                arrays[f"upper{i}"] = domain.geometry.upper
                domains.append({"domain_id": domain.domain_id, "target_index": request.class_labels.index(domain.target_class)})
            np.savez(root / "input.npz", **arrays)
            payload = {"layers": layers, "domains": domains, "margin": request.classification_margin,
                       "repository": str(repository), "revision": revision,
                       "class_labels": list(request.class_labels),
                       "sampling": self.config.sampling_settings(),
                       "input_sha256": hashlib.sha256((root / "input.npz").read_bytes()).hexdigest(),
                       "config": {k: getattr(self.config, k) for k in ("max_regions", "coverage_samples", "coverage_threshold", "seed", "device", "optimize_bounds")}}
            fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
            payload["request_sha256"] = fingerprint
            (root / "request.json").write_text(json.dumps(payload, allow_nan=False))
            worker = Path(__file__).with_name("premap_worker.py")
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            environment["PYTHONNOUSERSITE"] = "1"
            timed_out = False
            with (root / "worker.log").open("w") as log:
                process = subprocess.Popen([self.config.python, "-I", str(worker), str(root)],
                                           stdout=log, stderr=subprocess.STDOUT, env=environment,
                                           cwd=root, start_new_session=True)
                try:
                    process.wait(timeout=self.config.timeout_s)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                except BaseException:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise
            log_tail = (root / "worker.log").read_text(errors="replace")[-6000:]
            if not timed_out and process.returncode != 0:
                raise RuntimeError(f"PREMAP worker failed ({process.returncode}):\n{log_tail}")
            regions, domain_stats = [], []
            # Worker checkpoints each complete set of leaves atomically. Never
            # consume half-written iterations or turn sampling failure into proof.
            for i, domain in enumerate(request.domains):
                manifest_path = root / f"domain{i}.json"
                if not manifest_path.exists():
                    continue
                manifest = json.loads(manifest_path.read_text())
                if manifest["request_sha256"] != fingerprint:
                    raise ValueError("PREMAP checkpoint does not belong to this request")
                filename = manifest["file"]
                if Path(filename).name != filename:
                    raise ValueError("Invalid PREMAP output path")
                if hashlib.sha256((root / filename).read_bytes()).hexdigest() != manifest["sha256"]:
                    raise ValueError("PREMAP checkpoint checksum mismatch")
                domain_stats.append(manifest["diagnostics"])
                with np.load(root / filename, allow_pickle=False) as output:
                    for j in range(manifest["count"]):
                        g = RegionGeometry(*(output[f"r{j}_{key}"] for key in ("A", "b", "lower", "upper")))
                        if np.any(g.lower < domain.geometry.lower) or np.any(g.upper > domain.geometry.upper):
                            raise ValueError("Exported PREMAP region escapes its input domain")
                        regions.append(CertifiedRegion(f"premap:{i}:{j}", domain.target_class, g, domain.domain_id,
                            certificate={"backend": "premap", "revision": revision, "classification_margin": request.classification_margin,
                                         "request_sha256": fingerprint, "splitting": "input",
                                         "sampling_protocol": SAMPLING_PROTOCOL}))
            partial = timed_out or len(domain_stats) < len(request.domains) or any(not s["complete"] for s in domain_stats)
            return BuildResult(tuple(regions), request.class_labels, request.input_shape,
                               "budget_exhausted" if partial else "complete",
                               {"backend": "premap", "revision": revision, "timed_out": timed_out,
                                "sampling": self.config.sampling_settings(),
                                "domain_stats": domain_stats, "log_tail": log_tail})
