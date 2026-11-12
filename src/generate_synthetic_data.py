
from pathlib import Path

import numpy as np
import pandas as pd


# Configurações
SEED = 42
NUM_TRIPS = 10000

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic"
OUTPUT_FILE = OUTPUT_DIR / "fleet_trips.csv"

VEHICLE_CAPACITIES = {
    "Small Van": 3.0,
    "Van": 11.0,
    "Light Truck": 15.0,
    "Medium Truck": 25.0,
    "Heavy Truck": 50.0,
    "Trailer": 90.0,
}

REGIONS = ["North", "South", "East", "West", "Central"]


def generate_data(num_trips=NUM_TRIPS, seed=SEED):
    rng = np.random.default_rng(seed)

    vehicle_models = list(VEHICLE_CAPACITIES)

    vehicles = rng.choice(vehicle_models, size=num_trips)
    capacities = np.array(
        [VEHICLE_CAPACITIES[v] for v in vehicles]
    )

    regions = rng.choice(REGIONS, size=num_trips)

    # Perfis de ocupação simulados
    occupancy_profiles = rng.choice(
        ["low", "medium", "high"],
        size=num_trips,
        p=[0.30, 0.45, 0.25],
    )

    occupancy = np.empty(num_trips)

    for profile, low, high in [
        ("low", 0.05, 0.30),
        ("medium", 0.30, 0.70),
        ("high", 0.70, 1.00),
    ]:
        mask = occupancy_profiles == profile
        occupancy[mask] = rng.uniform(low, high, mask.sum())

    volumes = capacities * occupancy

    dates = pd.Timestamp("2026-01-01") + pd.to_timedelta(
        rng.integers(0, 180, num_trips), unit="D"
    )

    df = pd.DataFrame({
        "trip_id": [f"TRIP_{i:06d}" for i in range(num_trips)],
        "date": dates,
        "region": regions,
        "vehicle_model": vehicles,
        "capacity_m3": capacities,
        "cargo_volume_m3": np.round(volumes, 4),
        "occupancy_rate": np.round(occupancy, 4),
    })

    df["occupancy_category"] = pd.cut(
        df["occupancy_rate"],
        bins=[-0.01, 0.20, 0.50, 1.00],
        labels=["Critical", "Moderate", "Efficient"],
    )

    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = generate_data()
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Generated {len(df):,} synthetic trips")
    print(f"Output: {OUTPUT_FILE}")
    print()
    print(df.head(10).to_string(index=False))
    print()
    print("Occupancy distribution:")
    print(df["occupancy_category"].value_counts())


if __name__ == "__main__":
    main()
