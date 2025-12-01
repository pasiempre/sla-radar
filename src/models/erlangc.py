"""Erlang C queue theory calculations for call center SLA modeling."""
from __future__ import annotations
import math
from typing import Optional


def _factorial(n: int) -> float:
    """Compute factorial, returning float to avoid overflow for large n."""
    return float(math.prod(range(1, n + 1))) if n > 1 else 1.0


def traffic_intensity(lambda_rate: float, aht_eff_minutes: float, agents: int) -> float:
    """
    Calculate traffic intensity (rho) = offered load / capacity.
    
    Args:
        lambda_rate: Arrival rate (calls per unit time)
        aht_eff_minutes: Average handle time in minutes (talk + wrap)
        agents: Number of agents
    
    Returns:
        Traffic intensity (utilization). Values >= 1.0 indicate overload.
    """
    if agents <= 0:
        raise ValueError("agents must be positive")
    return (lambda_rate * aht_eff_minutes) / agents


def erlang_c_wait_probability(
    lambda_rate: float,
    mu: float,
    agents: int
) -> float:
    """
    Calculate Erlang C probability that a call must wait (Pw).
    
    Args:
        lambda_rate: Arrival rate per unit time
        mu: Service rate per agent per unit time (1/AHT)
        agents: Number of agents
    
    Returns:
        Probability that an arriving call will wait (0.0 to 1.0)
    """
    if agents <= 0 or mu <= 0 or lambda_rate <= 0:
        return 1.0
    
    a = lambda_rate / mu  # Traffic in Erlangs
    rho = a / agents      # Utilization
    
    if rho >= 1.0:
        return 1.0  # Unstable queue - all calls wait
    
    # Erlang C formula: Pw = (a^s / s!) * (s / (s - a)) / (sum_{k=0}^{s-1} a^k/k! + (a^s/s!) * (s/(s-a)))
    sum_terms = sum((a ** k) / _factorial(k) for k in range(agents))
    last_term = (a ** agents) / (_factorial(agents) * (1 - rho))
    
    return last_term / (sum_terms + last_term)


def estimate_sla_attainment(
    lambda_rate: float,
    aht_eff_minutes: float,
    agents: int,
    sla_target_minutes: float,
) -> float:
    """
    Estimate SLA attainment using Erlang C formula.
    
    SLA = 1 - Pw * exp(-(agents - a) * t / AHT)
    
    Args:
        lambda_rate: Arrival rate (calls per minute)
        aht_eff_minutes: Average handle time in minutes
        agents: Number of agents
        sla_target_minutes: SLA target time in minutes (e.g., 1.0 for "answered within 1 min")
    
    Returns:
        Estimated SLA attainment as decimal (0.0 to 1.0)
    """
    if agents <= 0 or aht_eff_minutes <= 0 or lambda_rate <= 0:
        return 0.0
    
    rho = traffic_intensity(lambda_rate, aht_eff_minutes, agents)
    
    if rho >= 1.0:
        return 0.0  # Queue is unstable, SLA collapses
    
    mu = 1.0 / aht_eff_minutes  # Service rate
    a = lambda_rate / mu        # Traffic in Erlangs
    
    Pw = erlang_c_wait_probability(lambda_rate, mu, agents)
    
    # SLA formula: probability of being answered within target time
    # SL(t) = 1 - Pw * exp(-(s*mu - lambda) * t)
    exponent = -(agents * mu - lambda_rate) * sla_target_minutes
    
    # Guard against numerical overflow
    if exponent > 700:
        return 0.0
    if exponent < -700:
        return 1.0
    
    sla = 1.0 - Pw * math.exp(exponent)
    return max(0.0, min(1.0, sla))