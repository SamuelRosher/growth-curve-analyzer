import numpy as np
import pandas as pd
from models import gompertz


def simulate_growth(
    conditions: dict,
    n_timepoints: int = 25,
    t_end: float = 24.0,
    noise_sd: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:

    np.random.seed(seed)
    time = np.linspace(0, t_end, n_timepoints)
    data = {"time_h": time}

    for name, (K, mu_max, lam) in conditions.items():
        od_clean = gompertz(time, K, mu_max, lam)
        noise = np.random.normal(loc=0, scale=noise_sd, size=len(time))
        od_noisy = np.clip(od_clean + noise, a_min=0, a_max=None)
        data[name] = od_noisy

    return pd.DataFrame(data)


if __name__ == "__main__":
    conditions = {
        "WT":        (1.20, 0.45, 2.0),
        "delta_rpoS": (0.85, 0.28, 4.5),
    }

    df = simulate_growth(conditions)
    output_path = "data/example_data.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved simulated data to {output_path}")
    print(f"\nTrue parameters used:")
    for name, (K, mu, lam) in conditions.items():
        td = round(0.693 / mu, 2)
        print(f"  {name}: K={K}, mu_max={mu}/hr, lag={lam}hr, t_d={td}hr")
    print(f"\nData preview:")
    print(df.head())
    