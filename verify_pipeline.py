"""
verify_pipeline.py — Sanity check without needing the full CSV.
Run: python verify_pipeline.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

print("="*55)
print("GRIDLOCK ORACLE — Pipeline Verification")
print("="*55)

# 1. Synthetic mini-dataset matching your schema
print("\n[1] Creating synthetic dataset...")
np.random.seed(42)
n = 200

df = pd.DataFrame({
    'id': [f'FKID{i:06d}' for i in range(n)],
    'event_type': np.random.choice(['planned','unplanned'], n, p=[0.1, 0.9]),
    'latitude': np.random.uniform(12.80, 13.27, n),
    'longitude': np.random.uniform(77.48, 77.75, n),
    'event_cause': np.random.choice(['accident','breakdown','event','construction'], n),
    'requires_road_closure': np.random.choice([True,False], n, p=[0.3,0.7]),
    'start_datetime': pd.date_range('2023-01-01', periods=n, freq='6h'),
    'end_datetime': pd.date_range('2023-01-01 02:00', periods=n, freq='6h'),
    'resolved_datetime': pd.date_range('2023-01-01 01:30', periods=n, freq='6h'),
    'closed_datetime': pd.date_range('2023-01-01 02:00', periods=n, freq='6h'),
    'status': 'resolved',
    'authenticated': 'yes',
    'modified_datetime': pd.date_range('2023-01-01', periods=n, freq='6h'),
    'priority': np.random.choice(['High','Medium','Low'], n, p=[0.2,0.5,0.3]),
    'corridor': np.random.choice(['Outer Ring Road','NH 44','Hosur Road','Tumkur Road','Mysore Road'], n),
    'zone': np.random.choice(['Central Zone 1','Central Zone 2','East Zone','West Zone','North Zone'], n),
    'junction': np.random.choice(['MekhriCircle','SilkBoardJunction','HebbalFlyover','MarathahalliBridge','UrvashiJunction', None], n),
    'police_station': np.random.choice(['Cubbon Park','Indiranagar','Jayanagar'], n),
    'created_date': pd.date_range('2023-01-01', periods=n, freq='6h'),
    'client_id': np.random.randint(1,5,n),
    'gba_identifier': np.random.choice(['Bengaluru Central Corporation', None], n),
    'kgid': [f'KG{i}' for i in range(n)],
})
# Add missing columns
for col in ['endlatitude','endlongitude','end_address','address','description',
            'veh_type','veh_no','cargo_material','reason_breakdown','age_of_truck',
            'route_path','assigned_to_police_id','citizen_accident_id','comment',
            'police_station','meta_data','map_file','direction',
            'created_by_id','last_modified_by_id','closed_by_id','resolved_by_id',
            'resolved_at_address','resolved_at_latitude','resolved_at_longitude']:
    df[col] = np.nan
df['police_station'] = 'Cubbon Park'

print(f"   ✅ Created {len(df)} synthetic rows, {len(df.columns)} columns")

# 2. Preprocessing
print("\n[2] Running preprocessing...")
from utils.preprocess import (
    load_and_clean, compute_junction_risk, compute_corridor_stats,
    compute_zone_stats, enrich_features, get_feature_columns,
    save_lookup_tables
)

os.makedirs("temp", exist_ok=True)

temp_csv = "temp/synthetic_gridlock.csv"

df.to_csv(temp_csv, index=False)
df_clean = load_and_clean(temp_csv)
print(f"   ✅ Cleaned. Shape: {df_clean.shape}")

jr = compute_junction_risk(df_clean)
cs = compute_corridor_stats(df_clean)
zs = compute_zone_stats(df_clean)
print(f"   ✅ Junction risk rows: {len(jr)}")
print(f"   ✅ Corridor stats rows: {len(cs)}")
save_lookup_tables(jr, cs, zs, 'models/')

df_enriched = enrich_features(df_clean, jr, cs, zs)
print(f"   ✅ Enriched. DS sample: {df_enriched['disruption_score'].describe()[['mean','min','max']].round(2).to_dict()}")

# 3. Model training
print("\n[3] Training XGBoost model...")
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle, shap

FEATURES = get_feature_columns()
df_enriched['duration_mins'] = df_enriched['duration_mins'].fillna(60)
X = df_enriched[FEATURES]
y = df_enriched['disruption_score']
mask = y.notna()
X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, verbosity=0)
model.fit(X_train, y_train)
mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"   ✅ Model trained. MAE: {mae:.4f}")

with open('models/xgb_disruption_model.pkl','wb') as f: pickle.dump(model, f)
explainer = shap.TreeExplainer(model)
with open('models/shap_explainer.pkl','wb') as f: pickle.dump(explainer, f)
df_enriched.to_csv('models/enriched_dataset.csv', index=False)
print("   ✅ Model + explainer + enriched dataset saved to models/")

# 4. Predictor
print("\n[4] Testing predictor...")
from predictor import GridlockPredictor
predictor = GridlockPredictor('models/')

test_event = {
    'event_type': 'unplanned',
    'priority': 'High',
    'requires_road_closure': True,
    'start_datetime': '2024-01-15 18:30:00',
    'duration_mins': 120,
    'junction': 'MekhriCircle',
    'corridor': 'Outer Ring Road',
    'zone': 'Central Zone 2',
}
result = predictor.predict(test_event)
print(f"   ✅ Prediction successful!")
print(f"      Disruption Score : {result['disruption_score']}/10")
print(f"      Impact Tier      : {result['impact_tier']}")
print(f"      Personnel        : {result['resources']['personnel']}")
print(f"      Barricades       : {result['resources']['barricades']}")
print(f"      Clearance (est.) : {result['resources']['estimated_clearance_mins']} min")
print(f"      Top SHAP Feature : {result['shap_explanations'][0]['feature']}")

print("\n" + "="*55)
print("✅ ALL CHECKS PASSED — Pipeline is ready!")
print("   Next step: python train_model.py --data data/flipkart_gridlock.csv")
print("="*55)