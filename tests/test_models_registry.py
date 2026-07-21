from gpu_cluster_sim.models import GPU, LeafSwitch, Node, RailSwitch, SpineSwitch
from gpu_cluster_sim.models.registry import ENTITY_KIND, KIND_TO_ENTITY


def test_entity_kind_covers_every_entity_class():
    assert set(ENTITY_KIND.values()) == {"gpu", "node", "leaf", "spine", "rail"}
    assert ENTITY_KIND[GPU] == "gpu"
    assert ENTITY_KIND[Node] == "node"
    assert ENTITY_KIND[LeafSwitch] == "leaf"
    assert ENTITY_KIND[SpineSwitch] == "spine"
    assert ENTITY_KIND[RailSwitch] == "rail"


def test_kind_to_entity_is_the_exact_inverse():
    assert KIND_TO_ENTITY == {kind: cls for cls, kind in ENTITY_KIND.items()}
    for cls, kind in ENTITY_KIND.items():
        assert KIND_TO_ENTITY[kind] is cls
