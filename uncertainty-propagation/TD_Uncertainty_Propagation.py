"""

Simplified white-box annual heating energy model for teaching and Monte-Carlo
uncertainty propagation in energy auditing.

Functions available:
- sample_parameters(specs, n): sample parameters from user-provided distribution specs.
- simulate_hourly_heating(params, weather_t_out, gains_t_int): returns hourly delivered heat (W) and electrical energy (W_elec) arrays.
- annual_energy_kwh(params, weather_t_out, gains_t_int, dt_hours=1.0): returns annual energy in kWh.
- monte_carlo_annual_energy(param_samples, weather_t_out, gains_t_int, dt_hours=1.0, n_jobs=1): vectorized Monte-Carlo wrapper.

Example usage is provided in the __main__ block.

Author: Samih Akkari, 2025
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sqlalchemy import null

# physical constants
RHO_AIR = 1.2       # kg/m3
CP_AIR = 1005       # J/(kg K)

def simulate_hourly_heating(params: Dict[str, float],
                             weather_t_out: np.ndarray,
                             gains_t_int: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate hourly heating loads for one year (8760 values).

    Returns:
      delivered_heat_W: array of heat that must be delivered to meet setpoint (W)
      electrical_power_W: array of electrical power consumed by heating system (W)

    Model:
      UA_total = sum(U_i * A_i)
      Q_trans = UA_total * (T_set - T_out)        (W)
      Q_inf = rho * cp * (ACH/3600) * V * (T_set - T_out)  (W)
      Q_int given by gains_t_int * Qint_mult (W)
      Q_needed = Q_trans + Q_inf - Q_int
      delivered_heat = max(Q_needed, 0)
      electrical_power = delivered_heat / COP

    All arrays must have the same length (8760).
    """
    hours = weather_t_out.size
    # validate
    T_out = np.asarray(weather_t_out, dtype=float)
    Q_int_base = np.asarray(gains_t_int, dtype=float)
    T_set = params.get('T_set', 20.0)
    T_set_arr = np.ones(hours, dtype=float) * T_set

    # build UA
    UA_total = (params['U_wall'] * params['A_wall']
                + params['U_roof'] * params['A_roof']
                + params['U_win'] * params['A_win'])

    # transmission heat loss (W)
    Q_trans = UA_total * (T_set_arr - T_out)

    # infiltration heat loss (W)
    # ACH is in 1/h, convert to kg/s via volumetric flow.
    ACH = params['ACH']
    V = params['V']
    m_dot = (ACH * V) / 3600.0 * RHO_AIR  # kg/s
    Q_inf = m_dot * CP_AIR * (T_set_arr - T_out)  # W

    # internal gains
    Q_int = Q_int_base * params.get('Qint_mult', 1.0)

    # required delivered heat (W)
    Heat_needed = Q_trans + Q_inf - Q_int
    Heat_needed = np.maximum(Heat_needed, 0.0)
    Heat_needed[1776 : 6912] = 0.0

    # electrical power assuming COP (COP>0)
    COP = params.get('COP', 3.0)
    # avoid division by zero
    cop = max(COP, 1e-3)
    Heat_consumption = Heat_needed / cop

    return Heat_needed, Heat_consumption


def annual_energy_kwh(params: Dict[str, float],
                       weather_t_out: np.ndarray,
                       gains_t_int: np.ndarray,
                       dt_hours: float = 1.0) -> float:
    """Compute annual electrical energy consumed for heating (kWh).

    dt_hours: timestep length in hours (default 1.0 for hourly)
    """
    _, electrical_power = simulate_hourly_heating(params, weather_t_out, gains_t_int)
    # energy in kWh
    E_kwh = np.sum(electrical_power) * dt_hours / 1000.0
    return float(E_kwh)


def sample_parameters(specs: Dict[str, Dict[str, Any]], n: int, rng: np.random.Generator = None) -> List[Dict[str, float]]:
    """Sample parameter dictionaries according to a simple spec format.

    specs: mapping param_name -> {'dist': 'uniform'|'triangular'|'normal_clipped'|'lognormal', 'params': ...}
      - uniform: params = (low, high)
      - triangular: params = (left, mode, right)
      - normal_clipped: params = (mu, sigma, low, high)
      - lognormal: params = (mu_ln, sigma_ln)  (log-space)

    Returns list of n parameter dicts.
    """
    if rng is None:
        rng = np.random.default_rng()

    samples = []
    for i in range(n):
        p = {}
        for name, spec in specs.items():
            dist = spec.get('dist', 'uniform')
            pr = spec.get('params')
            if dist == 'uniform':
                low, high = pr
                val = rng.uniform(low, high)
            elif dist == 'triangular':
                left, mode, right = pr
                val = rng.triangular(left, mode, right)
            elif dist == 'normal_clipped':
                mu, sigma, low, high = pr
                val = rng.normal(mu, sigma)
                val = float(np.clip(val, low, high))
            elif dist == 'lognormal':
                mu_ln, sigma_ln = pr
                val = rng.lognormal(mean=mu_ln, sigma=sigma_ln)
            elif dist == 'constant':
                val = pr
            else:
                raise ValueError(f"Unknown dist '{dist}' for parameter '{name}'")
            p[name] = float(val)
        samples.append(p)
    return samples


def monte_carlo_annual_energy(param_samples: List[Dict[str, float]],
                              weather_t_out: np.ndarray,
                              gains_t_int: np.ndarray,
                              dt_hours: float = 1.0) -> np.ndarray:
    """Run annual_energy_kwh for each parameter sample and return array of energies (kWh).

    This function is straightforward and not parallelized to keep dependencies minimal.
    """
    energies = np.empty(len(param_samples), dtype=float)
    for i, p in enumerate(param_samples):
        energies[i] = annual_energy_kwh(p, weather_t_out, gains_t_int, dt_hours=dt_hours)
    return energies


# -------------------- example usage --------------------
if __name__ == '__main__':
    
    df = pd.read_csv("Data.csv", delimiter=";")
    Text = df["Text"].values
    
    # small demo that runs a Monte-Carlo with synthetic weather and gains
    hours = len(Text)


    # nominal internal gains profile: 1.0 kW during daytime (8-20), 0.2 kW otherwise
    gains = np.zeros(hours)
    hours_in_day = np.arange(24)
    mask = (hours_in_day >= 8) & (hours_in_day < 20)
    # tile for whole year
    gains_hour = np.where(mask, 1000.0, 200.0)  # W
    gains = np.tile(gains_hour, int(hours/24))


    # sampling specs (students would define these)
    specs = {
        'U_wall': {'dist': 'triangular', 'params': (0.3, 0.4, 0.6)},
        'A_wall': {'dist': 'constant', 'params': 200.0},
        'U_roof': {'dist': 'uniform', 'params': (0.2, 0.5)},
        'A_roof': {'dist': 'constant', 'params': 100.0},
        'U_win': {'dist': 'uniform', 'params': (1.2, 2.5)},
        'A_win': {'dist': 'constant', 'params': 22.5},
        'ACH': {'dist': 'triangular', 'params': (0.3, 0.6, 1.0)},
        'V': {'dist': 'normal_clipped', 'params': (600.0, 30.0, 540.0, 660.0)},
        'COP': {'dist': 'normal_clipped', 'params': (3.0, 0.3, 2.0, 4.0)},
        'Qint_mult': {'dist': 'uniform', 'params': (0.7, 1.3)},
        'T_set': {'dist': 'uniform', 'params': (19.0, 21.0)}
    }

    rng = np.random.default_rng(42)
    N = 500
    samples = sample_parameters(specs, N, rng=rng)
    energies = monte_carlo_annual_energy(samples, Text, gains)

    import statistics
    mean_E = float(np.mean(energies))
    p50 = float(np.percentile(energies, 50))
    p90 = float(np.percentile(energies, 90))

    print(f"Monte Carlo runs: {N}")
    print(f"Mean annual heating energy (kWh): {mean_E:0.1f}")
    print(f"P50: {p50:0.1f} kWh, P90: {p90:0.1f} kWh")

    # simple histogram if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        plt.hist(energies, bins=30, density=False)
        plt.xlabel('Annual heating energy (kWh)')
        plt.title('MC distribution (demo)')
        plt.axvline(p50, color='C1', linestyle='--', label='P50')
        plt.axvline(p90, color='C2', linestyle='--', label='P90')
        plt.legend()
        plt.show()
    except Exception:
        pass

