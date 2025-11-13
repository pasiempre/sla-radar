from src.models.erlangc import traffic_intensity, estimate_sla_attainment

def test_traffic_intensity_basic():
    rho = traffic_intensity(lambda_rate=2.0, aht_eff_minutes=0.5, agents=4)
    assert abs(rho - 0.25) < 1e-6

def test_traffic_intensity_raises_for_non_positive_agents():
    try:
        traffic_intensity(lambda_rate=1.0, aht_eff_minutes=1.0, agents=0)
    except ValueError:
        return
    assert False, "Expected ValueError for agents <= 0"

def test_estimate_sla_attainment_overload_guard():
    sla = estimate_sla_attainment(
        lambda_rate=4.0,
        aht_eff_minutes=0.5,
        agents=2,
        sla_target_minutes=1.0
    )
    assert sla == 0.0