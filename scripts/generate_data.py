import pandas as pd
import numpy as np

# Parameters
n_runs = 50000
n_features = 100
missing_rate = 0.1
anomaly_rate = 0.05

# Generate sensor readings
np.random.seed(42)
sensor_data = np.random.normal(0, 1, size=(n_runs, n_features))
mask = np.random.rand(n_runs, n_features) < missing_rate
sensor_data[mask] = np.nan

# Timestamp: 6 months at 10-min intervals
dates = pd.date_range(start='2024-01-01', periods=n_runs, freq='10T')

# Labels
labels = np.where(np.random.rand(n_runs) < anomaly_rate, -1, 1)

# Assemble
df = pd.DataFrame(sensor_data, columns=[f"sensor_{i+1}" for i in range(n_features)])
df.insert(0, "run_id", range(1, n_runs+1))
df.insert(1, "timestamp", dates)
df['label'] = labels

# Save
df.to_csv('synthetic_semicon_50k.csv', index=False)