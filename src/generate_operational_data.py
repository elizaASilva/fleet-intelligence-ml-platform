from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N_TRIPS = 10_000
ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / 'data' / 'synthetic' / 'operational_trips.csv'

CAPACITY = {'Small Van': 3.0, 'Van': 11.0, 'Light Truck': 15.0, 'Medium Truck': 25.0, 'Heavy Truck': 50.0, 'Trailer': 90.0}
REGIONS = ['North', 'South', 'East', 'West', 'Central']
CATEGORIES = ['Fashion', 'Electronics', 'Home', 'Beauty', 'Mixed']
BASE_PARCEL_M3 = {'Fashion': .012, 'Electronics': .025, 'Home': .065, 'Beauty': .008, 'Mixed': .028}


def generate_data(n_trips=N_TRIPS, seed=SEED):
    rng = np.random.default_rng(seed)
    route_ids = np.array([f'ROUTE_{i:03d}' for i in range(1, 81)])
    route_origins = rng.choice(REGIONS, size=len(route_ids))
    route_destinations = rng.choice(REGIONS, size=len(route_ids))
    route_category = rng.choice(CATEGORIES, size=len(route_ids))
    route_index = rng.integers(0, len(route_ids), size=n_trips)
    route = route_ids[route_index]
    origin = route_origins[route_index]
    destination = route_destinations[route_index]
    category = route_category[route_index]

    date = pd.Timestamp('2026-01-01') + pd.to_timedelta(rng.integers(0, 180, n_trips), unit='D')
    departure_hour = rng.integers(5, 23, size=n_trips)
    departure_time = date + pd.to_timedelta(departure_hour, unit='h') + pd.to_timedelta(rng.integers(0, 60, n_trips), unit='m')
    # A route can have multiple executions on the same day.
    order = np.lexsort((departure_time.to_numpy(), route))
    sequence = np.empty(n_trips, dtype=int)
    sequence[order] = pd.DataFrame({'route': route[order], 'date': date.to_numpy()[order]}).groupby(['route', 'date']).cumcount().to_numpy() + 1

    vehicle_models = np.array(list(CAPACITY))
    vehicle_model = rng.choice(vehicle_models, n_trips, p=[.20, .30, .20, .15, .10, .05])
    capacity = np.array([CAPACITY[v] for v in vehicle_model])
    # Route-level demand, vehicle capacity, and execution sequence influence occupancy.
    route_demand = rng.uniform(.30, .85, size=len(route_ids))[route_index]
    target_occupancy = np.clip(route_demand - .06 * (sequence - 1) + rng.normal(0, .13, n_trips), .04, .98)
    avg_parcel_volume = np.array([BASE_PARCEL_M3[c] for c in category]) * rng.lognormal(0, .25, n_trips)
    parcel_count = np.maximum(1, np.rint(capacity * target_occupancy / avg_parcel_volume)).astype(int)
    true_volume = parcel_count * avg_parcel_volume
    occupancy = true_volume / capacity

    # The observed field is deliberately imperfect; the true value is retained for evaluation only.
    quality = rng.choice(['valid', 'missing', 'incorrect'], n_trips, p=[.78, .15, .07])
    observed = true_volume * rng.normal(1, .025, n_trips)
    observed[quality == 'missing'] = np.nan
    observed[quality == 'incorrect'] *= rng.choice([.15, 2.8], size=(quality == 'incorrect').sum())

    df = pd.DataFrame({
        'trip_id': [f'OP_{i:06d}' for i in range(n_trips)],
        'date': date,
        'departure_time': departure_time,
        'route_id': route,
        'origin_region': origin,
        'destination_region': destination,
        'trip_sequence': sequence,
        'vehicle_id': [f'VEH_{i % 750:04d}' for i in range(n_trips)],
        'vehicle_model': vehicle_model,
        'capacity_m3': capacity,
        'parcel_category': category,
        'parcel_count': parcel_count,
        'observed_volume_m3': np.round(observed, 4),
        'volume_quality': quality,
        'true_volume_m3': np.round(true_volume, 4),
        'true_occupancy_rate': np.round(occupancy, 4),
    })
    df['occupancy_category'] = pd.cut(df['true_occupancy_rate'], [-np.inf, .20, .50, np.inf], labels=['Critical', 'Moderate', 'Efficient'])
    return df.sort_values(['date', 'route_id', 'departure_time', 'trip_id']).reset_index(drop=True)


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df = generate_data()
    df.to_csv(OUTPUT, index=False)
    print(f'Generated {len(df):,} operational trips')
    print(f'Output: {OUTPUT}')
    print('\nVolume quality:')
    print(df['volume_quality'].value_counts().to_string())
    print('\nSample:')
    print(df.head(5).to_string(index=False))
    assert len(df) == N_TRIPS
    assert df.trip_id.is_unique
    assert df.true_occupancy_rate.between(0, 1).all()
    assert df.loc[df.volume_quality.eq('missing'), 'observed_volume_m3'].isna().all()
    print('\nValidation passed')


if __name__ == '__main__':
    main()
