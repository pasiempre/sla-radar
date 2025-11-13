from typing import Optional 

def traffic_intensity(lambda_rate: float, aht_eff_minutes: float, agents: int) -> float:
    if agents <= 0:
        raise ValueError("agents must be positive")
    return (lambda_rate * aht_eff_minutes) / agents

def estimate_sla_attainment(
        lambda_rate: float,
        aht_eff_minutes: float, 
        agents: int,
        sla_target_minutes: float,

) -> float:
    rho = traffic_intensity(lambda_rate, aht_eff_minutes, agents)

    if rho >= 1.0:
        return 0.0
    
        #TODO:
        return 0.5 