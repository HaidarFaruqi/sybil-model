import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# =============================
# MODEL (HARUS IDENTIK DENGAN TRAINING)
# =============================

class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# =============================
# LOAD MODEL CHECKPOINT
# =============================

checkpoint = torch.load("sybil_detector_real.pt")

threshold = checkpoint["threshold"]

# =============================
# FEATURES (WAJIB SAMA & URUTAN SAMA)
# =============================

features = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "Average Packet Size"
]

# =============================
# REBUILD MODEL
# =============================

input_dim = len(features)
model = AutoEncoder(input_dim)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# =============================
# REBUILD SCALER (DARI TRAINING)
# =============================

scaler = StandardScaler()
scaler.mean_ = checkpoint["scaler_mean"]
scaler.scale_ = checkpoint["scaler_scale"]
scaler.n_features_in_ = len(features)

# =============================
# LOAD TEST DATA
# =============================

df = pd.read_csv("test_traffic.csv")

df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# Pastikan semua fitur ada
missing = [f for f in features if f not in df.columns]
if len(missing) > 0:
    raise ValueError(f"Missing features in test data: {missing}")

X = df[features].values
X_scaled = scaler.transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# =============================
# DETECTION
# =============================

with torch.no_grad():
    reconstructed = model(X_tensor)
    mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1)

# Convert ke numpy
mse_np = mse.numpy()

df["Anomaly_Score"] = mse_np
df["Prediction"] = ["ANOMALY" if err > threshold else "NORMAL" for err in mse_np]

# =============================
# OUTPUT SUMMARY
# =============================

print("\n=== DETECTION SUMMARY ===")
print(df["Prediction"].value_counts())

anomaly_ratio = (df["Prediction"] == "ANOMALY").mean() * 100
print(f"\nAnomaly Percentage: {anomaly_ratio:.2f}%")

# Save result
df.to_csv("detection_result.csv", index=False)

print("\nDetection finished. Results saved to detection_result.csv")