"""
Explore apple variety dataset for CTA analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define column names based on apparent structure
column_names = [
    'variety', 'harvest_code', 'location', 'treatment', 'state', 'quality_overall', 
    'watercore', 'tree_vigor', 'fruit_drop', 'red_pct', 'color_pattern', 
    'comments', 'coarse_texture', 'harvest_load', 'fruit_load', 'eating_quality',
    'scab_susc', 'brix', 'flavor', 'flesh_color', 'firm_or_soft', 'firmness_value',
    'ground_color', 'over_color', 'image_url', 'size', 'shape', 'weight_grams',
    'harvest_window', 'russeting', 'season', 'biennial', 'storage_quality',
    'evaluation_date', 'drop_status', 'rootstock', 'preharvest_drop_pct',
    'drop_location', 'texture', 'CAR_susc', 'fireblight_susc', 'skin_feel',
    'skin_thick', 'starch_index', 'DA_meter', 'tree_habit', 'tree_vigor2'
]

# Load data
df = pd.read_csv('apples.csv', names=column_names[:47], low_memory=False)

print("Dataset Overview")
print("=" * 50)
print(f"Total samples: {len(df)}")
print(f"Total varieties: {df['variety'].nunique()}")

# Clean and analyze key features
print("\n\nKey Features for Neural Network:")
print("=" * 50)

# 1. Brix (sugar content) - critical for routing
df['brix_numeric'] = pd.to_numeric(df['brix'], errors='coerce')
print(f"\nBrix (sugar): {df['brix_numeric'].notna().sum()} samples")
print(f"  Range: {df['brix_numeric'].min():.1f} - {df['brix_numeric'].max():.1f}")
print(f"  Mean: {df['brix_numeric'].mean():.1f}")

# 2. Firmness
df['firmness_numeric'] = pd.to_numeric(df['firmness_value'], errors='coerce')
print(f"\nFirmness: {df['firmness_numeric'].notna().sum()} samples")
print(f"  Range: {df['firmness_numeric'].min():.1f} - {df['firmness_numeric'].max():.1f}")

# 3. Red percentage
df['red_pct_numeric'] = pd.to_numeric(df['red_pct'], errors='coerce')
print(f"\nRed color %: {df['red_pct_numeric'].notna().sum()} samples")

# 4. Starch index (maturity indicator)
df['starch_numeric'] = pd.to_numeric(df['starch_index'], errors='coerce')
print(f"\nStarch index: {df['starch_numeric'].notna().sum()} samples")

# Create routing labels based on quality and characteristics
print("\n\nRouting Label Creation:")
print("=" * 50)

def create_routing_label(row):
    """Create routing decision based on quality and other factors"""
    quality = str(row['quality_overall']).lower() if pd.notna(row['quality_overall']) else ''
    eating_quality = str(row['eating_quality']).lower() if pd.notna(row['eating_quality']) else ''
    firmness = row['firm_or_soft']
    storage = str(row['storage_quality']).lower() if pd.notna(row['storage_quality']) else ''
    
    # Fresh market criteria
    if any(x in quality for x in ['excellent', 'very good']):
        return 'fresh_premium'
    elif 'good' in quality and 'soft' not in str(firmness):
        return 'fresh_standard'
    elif any(x in quality for x in ['poor', 'fair-poor']) or 'soft' in str(firmness):
        return 'juice'
    elif 'fair' in quality:
        return 'juice' if 'soft' in str(firmness) else 'fresh_standard'
    else:
        return 'unknown'

df['routing'] = df.apply(create_routing_label, axis=1)
print("\nRouting distribution:")
print(df['routing'].value_counts())

# Identify premium varieties
premium_varieties = [
    'Honeycrisp', 'Cosmic Crisp', 'Ambrosia', 'Jazz', 'Envy', 
    'SweeTango', 'SnapDragon', 'RubyFrost', 'Opal', 'Kanzi'
]

df['is_premium'] = df['variety'].isin(premium_varieties)
print(f"\nPremium varieties in dataset: {df['is_premium'].sum()} samples")

# Analyze misrouting potential
print("\n\nPotential Misrouting Analysis:")
print("=" * 50)

# Premium varieties that might be misrouted
premium_misrouted = df[(df['is_premium']) & (df['routing'] == 'juice')]
print(f"Premium varieties routed to juice: {len(premium_misrouted)} samples")
if len(premium_misrouted) > 0:
    print("Examples:")
    for _, row in premium_misrouted.head(3).iterrows():
        print(f"  - {row['variety']}: {row['quality_overall']}, firmness={row['firm_or_soft']}")

# Feature engineering for neural network
print("\n\nFeature Engineering:")
print("=" * 50)

# Extract numeric features
numeric_features = []
feature_names = []

# Size encoding
size_map = {'small': 1, 'small-medium': 2, 'medium': 3, 'medium-large': 4, 'large': 5}
df['size_numeric'] = df['size'].map(lambda x: size_map.get(str(x).split('-')[0], 3) if pd.notna(x) else 3)

# Season encoding
season_map = {'early': 1, 'early-middle': 2, 'middle': 3, 'middle-late': 4, 'late': 5}
df['season_numeric'] = df['season'].map(lambda x: season_map.get(str(x), 3) if pd.notna(x) else 3)

# Create feature matrix for samples with essential data
essential_cols = ['brix_numeric', 'firmness_numeric', 'red_pct_numeric', 
                  'size_numeric', 'season_numeric', 'starch_numeric']

df_clean = df.dropna(subset=['brix_numeric', 'routing'])
print(f"\nSamples with essential features: {len(df_clean)}")

# Variety distribution in clean dataset
print("\nTop varieties in clean dataset:")
print(df_clean['variety'].value_counts().head(10))

# Save processed data
output_file = 'apples_processed.csv'
df_clean.to_csv(output_file, index=False)
print(f"\nProcessed data saved to: {output_file}")

# Summary statistics for paper
print("\n\nSummary Statistics for Paper:")
print("=" * 50)
print(f"Total varieties analyzed: {df_clean['variety'].nunique()}")
print(f"Total samples with complete data: {len(df_clean)}")
print(f"Premium variety samples: {df_clean['is_premium'].sum()}")
print(f"Average Brix (sugar): {df_clean['brix_numeric'].mean():.1f}")
print(f"Routing distribution:")
for route, count in df_clean['routing'].value_counts().items():
    pct = count / len(df_clean) * 100
    print(f"  {route}: {count} ({pct:.1f}%)")