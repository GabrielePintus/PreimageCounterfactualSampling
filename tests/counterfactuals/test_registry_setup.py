from counterfactuals.benchmarks import create_default_registries


def test_create_default_registries_exposes_benchmark_components():
    registries = create_default_registries()

    assert set(registries) == {"method", "dataset", "model", "metric", "preprocessing"}
    assert "certcf" in registries["method"].names()
    assert "adult" in registries["dataset"].names()
    assert "sklearn_mlp" in registries["model"].names()
    assert "proximity" in registries["metric"].names()
    assert "pca" in registries["preprocessing"].names()
