import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# =============================
# MODEL
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
# LOAD DATA
# =============================

print("Loading CICIDS dataset...")
df = pd.read_csv("Monday-WorkingHours.pcap_ISCX.csv")

# Clean columns
df.columns = df.columns.str.strip()

# Filter benign only
if "Label" in df.columns:
    benign_df = df[df["Label"] == "BENIGN"].copy()
    print(f"Total benign flows: {len(benign_df)}")
else:
    benign_df = df.copy()

benign_df.replace([np.inf, -np.inf], 0, inplace=True)
benign_df.fillna(0, inplace=True)

# Features (MUST match inference order)
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

X = benign_df[features].values
print(f"Feature shape: {X.shape}")

# =============================
# TRAIN/VAL SPLIT
# =============================

# Split 80/20 untuk validasi threshold
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

print(f"Train samples: {len(X_train)}")
print(f"Val samples: {len(X_val)}")

# =============================
# SCALING
# =============================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)

# =============================
# TRAIN
# =============================

model = AutoEncoder(input_dim=X_train_tensor.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("\nTraining model...")
epochs = 50  # lebih banyak untuk konvergensi lebih baik

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, X_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        # Validasi
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_tensor)
            val_loss = criterion(val_output, X_val_tensor)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f}")

print("Training complete.\n")

# =============================
# THRESHOLD CALCULATION (ROBUST)
# =============================

model.eval()

# Hitung reconstruction error di VALIDATION SET (bukan training)
with torch.no_grad():
    reconstructed = model(X_val_tensor)
    mse_per_sample = torch.mean((X_val_tensor - reconstructed) ** 2, dim=1)
    mse_np = mse_per_sample.numpy()

# Analisis distribusi error
print("="*60)
print("RECONSTRUCTION ERROR ANALYSIS (Validation Set)")
print("="*60)
print(f"Mean MSE:   {mse_np.mean():.6f}")
print(f"Median MSE: {np.median(mse_np):.6f}")
print(f"Std MSE:    {mse_np.std():.6f}")
print(f"Min MSE:    {mse_np.min():.6f}")
print(f"Max MSE:    {mse_np.max():.6f}")
print()

# Percentile-based thresholds (LEBIH ROBUST)
percentiles = [90, 95, 99, 99.5, 99.9]
print("PERCENTILE-BASED THRESHOLDS:")
for p in percentiles:
    thresh = np.percentile(mse_np, p)
    print(f"  {p}th percentile: {thresh:.6f}")

print()

# Method 1: Percentile (RECOMMENDED)
# Pilih 99.5th percentile = hanya 0.5% benign yang akan jadi false positive
threshold_percentile = np.percentile(mse_np, 99.5)

# Method 2: Mean + k*std (traditional, tapi pakai multiplier lebih besar)
threshold_mean_std = mse_np.mean() + 5 * mse_np.std()  # 5x std, bukan 3x

# Method 3: Median-based (robust terhadap outlier)
threshold_median = np.median(mse_np) + 5 * np.median(np.abs(mse_np - np.median(mse_np)))

print("="*60)
print("RECOMMENDED THRESHOLDS:")
print("="*60)
print(f"1. Percentile (99.5%):     {threshold_percentile:.6f}  ← RECOMMENDED")
print(f"2. Mean + 5*std:           {threshold_mean_std:.6f}")
print(f"3. Median-based (MAD):     {threshold_median:.6f}")
print("="*60)
print()

# Pilih threshold
CHOSEN_THRESHOLD = threshold_percentile  # Ganti ini kalau mau pakai method lain

# Test false positive rate
fp_count = np.sum(mse_np > CHOSEN_THRESHOLD)
fp_rate = fp_count / len(mse_np) * 100
print(f"False Positive Rate on Val Set: {fp_rate:.2f}% ({fp_count}/{len(mse_np)} samples)")
print()

# =============================
# SAVE MODEL
# =============================

torch.save({
    "model_state_dict": model.state_dict(),
    "scaler_mean": scaler.mean_,
    "scaler_scale": scaler.scale_,
    "threshold": CHOSEN_THRESHOLD,  # simpan sebagai float
    "threshold_info": {
        "method": "percentile_99.5",
        "percentile_99.5": float(threshold_percentile),
        "mean_plus_5std": float(threshold_mean_std),
        "median_mad": float(threshold_median),
        "val_fp_rate": float(fp_rate)
    }
}, "sybil_detector_real.pt")

print("✅ Model saved to 'sybil_detector_real.pt'")
print(f"   Threshold: {CHOSEN_THRESHOLD:.6f}")
print(f"   Expected FP rate: {fp_rate:.2f}%")