"""Train and audit a reproducible synthetic cargo-volume estimator (portfolio demo)."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / 'data' / 'synthetic' / 'operational_trips.csv'
OUTPUT = ROOT / 'data' / 'synthetic' / 'predicted_trips.csv'
MODEL_DIR = ROOT / 'models'
RANDOM_STATE = 42

NUMERIC = ['parcel_count', 'capacity_m3', 'trip_sequence', 'departure_hour', 'day_of_week']
CATEGORICAL = ['vehicle_model', 'parcel_category', 'origin_region', 'destination_region']
FEATURES = NUMERIC + CATEGORICAL


def add_features(df):
    df = df.copy()
    departure = pd.to_datetime(df['departure_time'], errors='raise')
    df['departure_hour'] = departure.dt.hour
    df['day_of_week'] = departure.dt.dayofweek
    return df


def scores(y_true, y_pred):
    if len(y_true) == 0:
        return None
    return {
        'mae_m3': round(float(mean_absolute_error(y_true, y_pred)), 4),
        'rmse_m3': round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        'r2': round(float(r2_score(y_true, y_pred)), 4) if len(y_true) >= 2 else None,
        'rows': int(len(y_true)),
    }


def make_model(algorithm):
    if algorithm == 'random_forest':
        prep = ColumnTransformer([
            ('numeric', SimpleImputer(strategy='median'), NUMERIC),
            ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
        ])
        estimator = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=3,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    else:
        prep = ColumnTransformer([
            ('numeric', SimpleImputer(strategy='median'), NUMERIC),
            ('category', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CATEGORICAL),
        ])
        estimator = HistGradientBoostingRegressor(
            max_iter=160, learning_rate=0.07,
            l2_regularization=1.0, random_state=RANDOM_STATE,
        )
    return Pipeline([('preprocessing', prep), ('regressor', estimator)])


def main():
    if not INPUT.is_file():
        raise FileNotFoundError(f'Dataset not found: {INPUT}. Run generate_operational_data.py first.')
    df = add_features(pd.read_csv(INPUT))
    required = set(FEATURES + [
        'date', 'trip_id', 'true_volume_m3', 'volume_quality',
        'capacity_m3', 'observed_volume_m3',
    ])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if df['trip_id'].isna().any() or df['trip_id'].duplicated().any():
        raise ValueError('Missing or duplicate trip IDs.')
    if df['true_volume_m3'].isna().any():
        raise ValueError('Missing synthetic ground-truth volumes.')
    if not np.isfinite(df['capacity_m3']).all() or (df['capacity_m3'] <= 0).any():
        raise ValueError('All vehicle capacities must be finite and greater than zero.')
    if not df['volume_quality'].isin(['valid', 'missing', 'incorrect']).all():
        raise ValueError('Unexpected volume_quality value.')

    # Temporal holdout: later dates are unseen during model selection.
    dates = sorted(pd.to_datetime(df['date']).unique())
    if len(dates) < 2:
        raise ValueError('Not enough dates to form train/test split.')
    cutoff = dates[min(int(len(dates) * 0.8), len(dates) - 1)]
    train_mask = pd.to_datetime(df['date']) < cutoff
    test_mask = ~train_mask
    train, test = df.loc[train_mask], df.loc[test_mask]
    if train.empty or test.empty:
        raise ValueError('Not enough dates to form train/test split.')

    results = {}
    models = {}
    test_predictions = {}
    for algorithm in ['random_forest', 'gradient_boosting']:
        model = make_model(algorithm)
        model.fit(train[FEATURES], train['true_volume_m3'])
        pred = np.maximum(0, model.predict(test[FEATURES]))
        results[algorithm] = scores(test['true_volume_m3'], pred)
        models[algorithm] = model
        test_predictions[algorithm] = pred
        print(f'{algorithm}: {results[algorithm]}')

    winner = min(results, key=lambda name: results[name]['rmse_m3'])
    best = models[winner]
    capacity = df['capacity_m3'].to_numpy(dtype=float)
    observed = pd.to_numeric(df['observed_volume_m3'], errors='coerce')
    labeled_valid = df['volume_quality'].eq('valid')
    physically_valid = observed.notna() & np.isfinite(observed) & observed.ge(0) & observed.le(df['capacity_m3'])
    use_observed = labeled_valid & physically_valid
    invalid_labeled_valid = labeled_valid & ~physically_valid

    # Predictions for train rows are in-sample; only the held-out test scores
    # above and the test-subset scores below measure generalization.
    raw_pred = np.maximum(0, best.predict(df[FEATURES]))
    bounded_pred = np.minimum(raw_pred, capacity)
    df['predicted_volume_m3'] = np.round(bounded_pred, 4)
    df['estimated_volume_m3'] = np.where(
        use_observed, observed.to_numpy(), bounded_pred,
    )
    df['estimated_volume_m3'] = np.round(df['estimated_volume_m3'], 4)
    df['estimated_occupancy_rate'] = (
        df['estimated_volume_m3'] / df['capacity_m3']
    ).round(4)
    df['volume_source'] = np.where(use_observed, 'observed', 'model')
    df['volume_audit_status'] = np.select(
        [invalid_labeled_valid, use_observed, df['volume_quality'].eq('missing')],
        ['valid_label_failed_capacity_check', 'observed_valid', 'missing_estimated'],
        default='incorrect_estimated',
    )
    df['observed_exceeds_capacity'] = observed.gt(df['capacity_m3']).fillna(False)
    df['is_temporal_test'] = test_mask.to_numpy()

    if ((df['estimated_volume_m3'] < 0) | (df['estimated_volume_m3'] > df['capacity_m3'] + 1e-8)).any():
        raise AssertionError('Final estimated volume violates vehicle capacity.')
    if ((df['estimated_occupancy_rate'] < 0) | (df['estimated_occupancy_rate'] > 1)).any():
        raise AssertionError('Final occupancy is outside [0, 1].')

    # Evaluate only truly held-out rows where model inference is used.
    repair_test = test_mask & ~use_observed
    repaired_test_pred = df.loc[repair_test, 'predicted_volume_m3']
    repair_metrics = scores(df.loc[repair_test, 'true_volume_m3'], repaired_test_pred)
    report = {
        'selected_model': winner,
        'train_rows': int(len(train)),
        'test_rows': int(len(test)),
        'test_start_date': str(pd.Timestamp(cutoff).date()),
        'holdout_metrics': results,
        'holdout_model_repair_metrics': repair_metrics,
        'audit': {
            'total_rows': int(len(df)),
            'observed_kept': int(use_observed.sum()),
            'model_estimated': int((~use_observed).sum()),
            'valid_labels_rejected_by_physical_check': int(invalid_labeled_valid.sum()),
            'observed_exceeds_capacity': int(df['observed_exceeds_capacity'].sum()),
            'final_capacity_violations': int((df['estimated_volume_m3'] > df['capacity_m3']).sum()),
            'audit_status_counts': {str(k): int(v) for k, v in df['volume_audit_status'].value_counts().items()},
        },
        'features': FEATURES,
        'note': (
            'Synthetic demo only. Ground truth is used for training and evaluation, never as an input feature. '
            'Holdout metrics refer to future dates; training-row predictions are in-sample. '
            'Capacity is treated as a hard physical bound for this demo.'
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_DIR / 'cargo_volume_model.joblib')
    df.to_csv(OUTPUT, index=False)
    (MODEL_DIR / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'\nSelected model: {winner}')
    print(f'Train: {len(train):,} | Test: {len(test):,} | Test from: {report["test_start_date"]}')
    print(f'Audit: {report["audit"]}')
    print(f'Holdout repair subset: {repair_metrics}')
    print(f'Model: {MODEL_DIR / "cargo_volume_model.joblib"}')
    print(f'Predictions: {OUTPUT}')
    print('Training, evaluation and audit completed successfully.')


if __name__ == '__main__':
    main()
