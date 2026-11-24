import pytest
import graphics.foliage_impostor_crossfade as mod

CASES = [
    ("hysteresis_weight", (1,)),
    ("facing_select", (1,)),
    ("atlas_fallback", (1,)),
    ("atlas_order", (1,)),
    ("recycle_slot", (1,)),
    ("wind_phase", (1,)),
    ("determinism", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("impostor_bind", (1,)),
    ("metrics", (1,)),
    ("crossfade", (1,)),
    ("instance_sort", (1,)),
    ("render", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
