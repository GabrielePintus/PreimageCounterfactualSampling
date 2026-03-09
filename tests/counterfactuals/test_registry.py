from counterfactuals.core.registry import Registry


def test_registry_register_and_create():
    registry = Registry("dummy")

    class Dummy:
        def __init__(self, value: int):
            self.value = value

    registry.register("d", Dummy)
    obj = registry.create("d", value=7)

    assert obj.value == 7
    assert registry.names() == ["d"]
