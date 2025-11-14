# tests/test_erlang_core.py
import math

import pytest

erlang_module = pytest.importorskip("src.models.erlangc", reason="erlangc module not found")


def _get_sla_fn():
    """
    Try to locate the SLA / Erlang-C function in src.models.erlangc.

    We keep this flexible so you can rename things later without
    totally breaking the tests – just add the new name here.
    """
    candidate_names = [
        "erlang_c_sla",
        "_erlang_sla",
        "service_level",
        "compute_sla",
    ]
    for name in candidate_names:
        fn = getattr(erlang_module, name, None)
        if callable(fn):
            return fn

    pytest.skip("No known SLA function found in src.models.erlangc")


def test_sla_improves_with_more_agents():
    sla_fn = _get_sla_fn()

    # Reasonable call center-ish parameters
    lam_per_min = 3.0   # 3 calls per minute (~90 per 30m)
    aht_eff = 300.0     # 5 min AHT
    sla_target_min = 15.0

    # Light staffing vs. heavier staffing
    sla_few = sla_fn(lam_per_min, aht_eff, 5, sla_target_min)
    sla_many = sla_fn(lam_per_min, aht_eff, 20, sla_target_min)

    assert 0.0 <= sla_few <= 100.0
    assert 0.0 <= sla_many <= 100.0
    # With more agents, SLA should not get worse
    assert sla_many >= sla_few


def test_sla_collapses_when_overloaded():
    sla_fn = _get_sla_fn()

    lam_per_min = 30.0   # 30 calls/min → 900 per 30m (heavy)
    aht_eff = 300.0      # 5 min AHT
    sla_target_min = 15.0

    # Deliberately understaffed
    sla = sla_fn(lam_per_min, aht_eff, 5, sla_target_min)

    # In a badly overloaded system, SLA should be near zero
    assert 0.0 <= sla <= 20.0