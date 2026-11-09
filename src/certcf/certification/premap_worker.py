"""Standalone worker for original PREMAP input splitting (external environment).

No certcf imports: the worker must not import the application's auto_LiRPA.
Hooks adapt benchmark-specific sampling and capture full leaf geometry. Bound
optimization, feature selection, splitting, and queue management are upstream.
Only the pinned revision in premap.py is supported; ReLU splitting is excluded.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

import numpy as np
import torch
from torch import nn


class BoxCoverage:
    """Generic binary-box version of upstream's benchmark-specific samplers.

    Keep upstream's reset-per-call RNG policy. Volumes are divided by the
    source-box volume, a common factor that does not change split priorities.
    This class is independent of bound optimization and can be tested directly
    against the original sampling functions with identical inputs.
    """

    def __init__(self, model, target, lower, upper, samples, seed):
        self.model, self.target = model, target
        self.lower, self.upper = lower, upper
        self.samples, self.seed = samples, seed

    def estimate(self, A, b, lo, hi, count):
        lo, hi = lo.reshape(-1), hi.reshape(-1)
        x = lo + torch.rand((count, lo.numel()), device=lo.device) * (hi - lo)
        with torch.no_grad():
            # Upstream uses argmax, including its deterministic tie rule.
            is_target = self.model(x).argmax(1) == self.target
            a = torch.as_tensor(np.asarray(A).reshape(-1, lo.numel()), dtype=x.dtype, device=x.device)
            bias = torch.as_tensor(np.asarray(b).reshape(-1), dtype=x.dtype, device=x.device)
            scores = x @ a.T + bias
            inside = (scores >= 0).all(1)
            n_target = int(is_target.sum().item())
            ratio = float((inside & is_target).sum().item() / n_target) if n_target else 0.0
            volume = float(torch.prod(((hi - lo) / (self.upper - self.lower)).double()).item())
            reward = float(torch.sigmoid(scores[is_target].min(1).values).mean().item()) if n_target else -2.0
        return volume * n_target / count, ratio, reward

    def initial(self, bounds):
        torch.manual_seed(self.seed)
        return self.estimate(bounds["lA"], bounds["lbias"], self.lower, self.upper, self.samples)[:2]

    def pairwise(self, bounds, lows, highs, spec_dim):
        torch.manual_seed(self.seed)
        if spec_dim != 1:
            raise ValueError("The adapter only supports binary specifications")
        pairs = []
        for j in range(0, len(lows), 2):
            pair = [self.estimate(bounds["lA"][k], bounds["lbias"][k], lows[k], highs[k], self.samples // 2)
                    for k in (j, j + 1)]
            volume = sum(v[0] for v in pair)
            ratio = sum(v[0] * v[1] for v in pair) / volume if volume else 0.0
            pairs.append(pair + [(volume, ratio, sum(v[2] for v in pair if v[0] > 0))])
        return pairs


def read_model(payload, arrays):
    layers = []
    for item in payload["layers"]:
        if item["type"] == "relu":
            layers.append(nn.ReLU())
        else:
            weight, bias = arrays[item["prefix"] + "_weight"], arrays[item["prefix"] + "_bias"]
            layer = nn.Linear(weight.shape[1], weight.shape[0])
            with torch.no_grad():
                layer.weight.copy_(torch.from_numpy(weight))
                layer.bias.copy_(torch.from_numpy(bias))
            layers.append(layer)
    return nn.Sequential(*layers).eval()


def checkpoint(root, payload, index, leaves, diagnostics):
    arrays = {}
    for j, (A, b, lower, upper) in enumerate(leaves):
        for name, value in zip(("A", "b", "lower", "upper"), (A, b, lower, upper)):
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            arrays[f"r{j}_{name}"] = np.asarray(value, dtype=np.float64)
    filename = f"domain{index}-{uuid.uuid4().hex}.npz"
    np.savez(root / filename, **arrays)
    record = {"file": filename, "count": len(leaves), "diagnostics": diagnostics,
              "sha256": hashlib.sha256((root / filename).read_bytes()).hexdigest(),
              "request_sha256": payload["request_sha256"]}
    temporary = root / f".domain{index}.json"
    temporary.write_text(json.dumps(record, allow_nan=False))
    os.replace(temporary, root / f"domain{index}.json")


def run(root):
    payload = json.loads((root / "request.json").read_text())
    specification = {k: v for k, v in payload.items() if k != "request_sha256"}
    if hashlib.sha256(json.dumps(specification, sort_keys=True, allow_nan=False).encode()).hexdigest() != payload["request_sha256"]:
        raise ValueError("Specification checksum mismatch")
    if hashlib.sha256((root / "input.npz").read_bytes()).hexdigest() != payload["input_sha256"]:
        raise ValueError("Request checksum mismatch")
    repository = Path(payload["repository"])
    sys.path[:0] = [str(repository / "src"), str(repository)]
    import arguments
    import preimage_batch_approx_input_split as splitting
    from preimage_beta_crown_solver_relu_split import LiRPAConvNet
    from auto_LiRPA import BoundedTensor, PerturbationLpNorm

    defaults = vars(arguments.Config.defaults_parser.parse_args([]))
    arguments.Config.construct_config_dict(defaults)
    cfg, options = arguments.Config, payload["config"]
    cfg["general"]["device"] = options["device"]
    cfg["general"]["seed"] = options["seed"]
    cfg["data"]["dataset"] = "certcf_external"
    cfg["data"]["num_outputs"] = 2
    cfg["bab"]["initial_max_domains"] = 1  # one binary output specification
    cfg["bab"]["branching"]["input_split"]["enable"] = True
    cfg["preimage"].update(under_approx=True, over_approx=False, quant=False,
        save_process=False, save_cov=False, threshold=options["coverage_threshold"],
        sample_num=options["coverage_samples"], branch_budget=2 * options["max_regions"], smooth_val=True)
    cfg["specification"]["norm"] = np.inf
    cfg["solver"]["bound_prop_method"] = "alpha-crown"
    cfg["solver"]["batch_size"] = 1

    original_check = splitting.initial_check_preimage_approx
    original_batch = splitting.batch_verification_input_split
    original_info = splitting.get_preimage_info

    class RegionBudget(Exception):
        pass

    with np.load(root / "input.npz", allow_pickle=False) as arrays:
        for index, domain in enumerate(payload["domains"]):
            torch.manual_seed(options["seed"] + index)
            model = read_model(payload, arrays).to(options["device"])
            target = domain["target_index"]
            # Encode gamma as an output specification without modifying the
            # original weights: the appended layer subtracts it from target.
            shift = nn.Linear(2, 2).to(options["device"])
            with torch.no_grad():
                shift.weight.copy_(torch.eye(2, device=options["device"]))
                shift.bias.zero_()
                shift.bias[target] = -payload["margin"]
            model = nn.Sequential(*list(model.children()), shift).eval()
            # Affine models have no relaxation parameters to optimize.
            cfg["solver"]["bound_prop_method"] = "alpha-crown" if options.get("optimize_bounds", True) and any(isinstance(layer, nn.ReLU) for layer in model) else "crown"
            lower = torch.tensor(arrays[f"lower{index}"], dtype=torch.float32, device=options["device"])
            upper = torch.tensor(arrays[f"upper{index}"], dtype=torch.float32, device=options["device"])
            d = lower.numel()
            c = torch.zeros((1, 1, 2), device=options["device"])
            c[0, 0, target], c[0, 0, 1 - target] = 1, -1
            cfg["preimage"]["label"] = target
            leaves = []
            latest_coverage = None

            coverage = BoxCoverage(model, target, lower, upper,
                                   options["coverage_samples"], options["seed"] + index)

            def persist(complete=False, ratio=None):
                nonlocal latest_coverage
                if ratio is not None:
                    latest_coverage = ratio
                checkpoint(root, payload, index, leaves,
                           {"domain_id": domain["domain_id"], "complete": complete,
                            "estimated_coverage": latest_coverage, "region_count": len(leaves),
                            "coverage_seed": coverage.seed,
                            "bound_method": cfg["solver"]["bound_prop_method"],
                            "coverage_is_sampling_estimate": True})

            def checked_initial(*args):
                nonlocal leaves
                result = original_check(*args)
                bounds = result[3]
                leaves = [(np.asarray(bounds["lA"]).reshape(-1, d), np.asarray(bounds["lbias"]).reshape(-1),
                           lower.cpu().numpy(), upper.cpu().numpy())]
                persist(bool(result[0]), float(result[1]))
                return result

            def captured_info(domains, *args):
                nonlocal leaves
                result = original_info(domains, *args)
                leaves = [(np.asarray(A).reshape(-1, d), np.asarray(b).reshape(-1),
                           np.asarray(lo).reshape(-1), np.asarray(hi).reshape(-1))
                          for (A, b), (lo, hi) in zip(result[1], result[2])]
                persist()
                return result

            def bounded_batch(domains, *args, **kwargs):
                if len(domains) >= options["max_regions"]:
                    captured_info(domains, True, False)
                    raise RegionBudget()
                ratio = original_batch(domains, *args, **kwargs)
                captured_info(domains, True, False)
                persist(ratio >= options["coverage_threshold"], float(ratio))
                return ratio

            # These hooks exist only in this isolated worker. Upstream files
            # remain untouched. Generic sampling replaces benchmark data loaders.
            splitting.calc_input_coverage_initial_input_under = coverage.initial
            splitting.calc_Hrep_coverage_multi_spec_pairwise_under = coverage.pairwise
            splitting.initial_check_preimage_approx = checked_initial
            splitting.get_preimage_info = captured_info
            splitting.batch_verification_input_split = bounded_batch
            midpoint = ((lower + upper) / 2).unsqueeze(0)
            net = LiRPAConvNet(model, in_size=midpoint.shape, c=c)
            x = BoundedTensor(midpoint, PerturbationLpNorm(norm=np.inf, x_L=lower.unsqueeze(0), x_U=upper.unsqueeze(0)))
            try:
                result = splitting.input_bab_approx_parallel_multi(
                    net, torch.stack((lower, upper), dim=-1), x,
                    model_ori=model, rhs=torch.zeros((1, 1), device=options["device"]),
                    branching_method="preimg")
                # Upstream reports a sentinel 1 when no target samples occur;
                # retain the measured initial value rather than claiming 100%.
                persist(bool(result[0]), float(result[4][-1]) if result[0] or result[5] > 1 else None)
            except RegionBudget:
                persist()


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve())
