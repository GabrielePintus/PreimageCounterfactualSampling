"""External-environment oracle; never import this into the main test process.

The build path calls original PREMAP with its ORIGINAL coverage functions.
Only dataset/model loaders and singleton-axis layout are adapted; get_preimage_info is observed to
capture boxes (the upstream return value omits them). No CertCF worker code is
used to compute reference bounds, coverage, splits, queue order, or stopping.
"""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn


def main(repository, root, worker_path):
    sys.path[:0] = [str(repository / "src"), str(repository)]
    import arguments
    import preimage_batch_approx_input_split as splitting
    import test_polyhedron_util as sampling
    from auto_LiRPA import BoundedTensor, PerturbationLpNorm
    from preimage_beta_crown_solver_relu_split import LiRPAConvNet

    arguments.Config.construct_config_dict(vars(arguments.Config.defaults_parser.parse_args([])))
    cfg = arguments.Config
    payload = json.loads((root / "reference.json").read_text())
    cfg["general"].update(device="cpu", seed=payload["seed"])
    cfg["data"].update(dataset="certcf_reference_fixture", num_outputs=2)
    cfg["bab"]["initial_max_domains"] = 1
    cfg["bab"]["branching"]["input_split"]["enable"] = True
    cfg["preimage"].update(under_approx=True, over_approx=False, quant=False,
                           save_process=False, save_cov=False, label=payload["target"],
                           threshold=payload["threshold"], sample_num=payload["samples"],
                           smooth_val=True,
                           # Upstream tests Visited > budget before a split and
                           # adds two visits per split; N leaves need N-1 splits.
                           branch_budget=2 * (payload["max_regions"] - 2))
    cfg["specification"]["norm"] = np.inf
    cfg["solver"].update(bound_prop_method="alpha-crown", batch_size=1)
    torch.manual_seed(payload["seed"])
    layers = []
    with np.load(root / "reference.npz", allow_pickle=False) as data:
        for item in payload["layers"]:
            if item["type"] == "relu":
                layers.append(nn.ReLU())
            else:
                w, b = data[item["prefix"] + "_weight"], data[item["prefix"] + "_bias"]
                layer = nn.Linear(w.shape[1], w.shape[0])
                with torch.no_grad():
                    layer.weight.copy_(torch.from_numpy(w))
                    layer.bias.copy_(torch.from_numpy(b))
                layers.append(layer)
    shift = nn.Linear(2, 2)
    with torch.no_grad():
        shift.weight.copy_(torch.eye(2))
        shift.bias.zero_()
        shift.bias[payload["target"]] = -payload["margin"]
    model = nn.Sequential(*layers, shift).eval()
    lower, upper = (torch.tensor(payload[k], dtype=torch.float32) for k in ("lower", "upper"))
    midpoint = ((lower + upper) / 2).unsqueeze(0)
    sampling.load_model = lambda **kwargs: model
    sampling.load_input_bounds = lambda *args, **kwargs: (
        midpoint, None, upper.unsqueeze(0), lower.unsqueeze(0), None)
    original_initial = sampling.calc_input_coverage_initial_input_under
    def initial_with_benchmark_layout(bounds):
        # Original non-cartpole loaders use an extra singleton specification
        # axis. Change layout only, not values or coverage/termination logic.
        shaped = dict(bounds, lA=np.asarray(bounds["lA"]).reshape(1, 1, 2),
                      lbias=np.asarray(bounds["lbias"]).reshape(1, 1))
        return original_initial(shaped)
    splitting.calc_input_coverage_initial_input_under = initial_with_benchmark_layout

    if payload["mode"] == "sampling":
        # Compare the generic sampler against unchanged upstream functions.
        spec = importlib.util.spec_from_file_location("certcf_worker_for_test", worker_path)
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        coverage = worker.BoxCoverage(model, payload["target"], lower, upper,
                                      payload["samples"], payload["seed"])
        bounds = {"lA": np.array([[[.7, -.2]]], dtype=np.float32),
                  "lbias": np.array([[-.1]], dtype=np.float32)}
        ref = sampling.calc_input_coverage_initial_input_under(bounds)
        volume = float(torch.prod(upper - lower))
        # Deliberately disturb global RNG: initial() must reset it like upstream.
        torch.manual_seed(999)
        actual = coverage.initial(bounds)
        np.testing.assert_allclose(actual, [float(ref[0]) / volume, ref[1]], atol=1e-7)
        lows = torch.stack((lower, torch.tensor([0., -1.]), lower, torch.tensor([-1., 0.])))
        highs = torch.stack((torch.tensor([0., 1.]), upper, torch.tensor([1., 0.]), upper))
        bounds = {"lA": np.repeat(bounds["lA"], 4, axis=0),
                  "lbias": np.repeat(bounds["lbias"], 4, axis=0)[..., None]}
        ref = np.asarray(sampling.calc_Hrep_coverage_multi_spec_pairwise_under(bounds, lows, highs, 1), dtype=float)
        ref[:, :, 0] /= volume
        np.testing.assert_allclose(coverage.pairwise(bounds, lows, highs, 1), ref, atol=2e-7, rtol=1e-6)
        # Audit the optimizer's actual draws, not just declared config values.
        draws = []
        original_sample = torch.distributions.Uniform.sample
        def observed_sample(distribution, sample_shape=torch.Size()):
            draws.append(list(sample_shape))
            return original_sample(distribution, sample_shape)
        torch.distributions.Uniform.sample = observed_sample
        perturbation = PerturbationLpNorm(norm=np.inf, x_L=torch.zeros(1, 2), x_U=torch.ones(1, 2))
        values = []
        for seed in (42, 123):
            torch.manual_seed(seed)
            values.append(perturbation.concretize_matrix_poly_LSE(
                torch.zeros(1, 2), torch.tensor([[[1., -1.]]]), torch.zeros(1, 1), -1, None))
        assert draws == [[1000], [1000]], draws
        torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
        (root / "reference-output.json").write_text(json.dumps({"sampling_equivalent": True, "optimizer_draws": draws}))
        return

    captured = []
    original_info = splitting.get_preimage_info
    def observe_info(*args, **kwargs):
        value = original_info(*args, **kwargs)
        captured[:] = [value]
        return value
    splitting.get_preimage_info = observe_info
    c = torch.zeros((1, 1, 2))
    c[0, 0, payload["target"]], c[0, 0, 1 - payload["target"]] = 1, -1
    net = LiRPAConvNet(model, in_size=midpoint.shape, c=c)
    x = BoundedTensor(midpoint, PerturbationLpNorm(norm=np.inf, x_L=lower.unsqueeze(0), x_U=upper.unsqueeze(0)))
    result = splitting.input_bab_approx_parallel_multi(
        net, torch.stack((lower, upper), dim=-1), x, model_ori=model,
        rhs=torch.zeros((1, 1)), branching_method="preimg")
    if captured:
        pairs, boxes = captured[-1][1:]
    else:
        pairs = [(result[1]["lA"], result[1]["lbias"])]
        boxes = [(lower.numpy(), upper.numpy())]
    regions = [dict(A=np.asarray(A).reshape(-1, 2).tolist(), b=np.asarray(b).reshape(-1).tolist(),
                    lower=np.asarray(lo).reshape(-1).tolist(), upper=np.asarray(hi).reshape(-1).tolist())
               for (A, b), (lo, hi) in zip(pairs, boxes)]
    (root / "reference-output.json").write_text(json.dumps({
        "regions": regions, "complete": bool(result[0]), "coverage": float(result[4][-1])}))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
