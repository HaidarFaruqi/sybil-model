import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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

df = pd.read_csv("Monday-WorkingHours.pcap_ISCX.csv")

# rapikan kolom (penting untuk CICIDS)
df.columns = df.columns.str.strip()

if "Label" in df.columns:
    df = df[df["Label"] == "BENIGN"]

df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

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

X = df[features].values

# =============================
# SCALING
# =============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# =============================
# TRAIN
# =============================

model = AutoEncoder(input_dim=X_tensor.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(30):
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output, X_tensor)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# =============================
# THRESHOLD
# =============================

with torch.no_grad():
    reconstructed = model(X_tensor)
    mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1)

threshold = mse.mean() + 3 * mse.std()

# =============================
# SAVE SEMUA KE .pt
# =============================

torch.save({
    "model_state_dict": model.state_dict(),
    "scaler_mean": scaler.mean_,
    "scaler_scale": scaler.scale_,
    "threshold": threshold
}, "sybil_detector_real.pt")

print("Model saved successfully.")