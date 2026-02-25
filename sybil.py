import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# =====================
# 1. LOAD DATA
# =====================
path = r'C:\Users\ASUS\.cache\kagglehub\datasets\chethuhn\network-intrusion-dataset\versions\1'
file_name = 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
full_path = os.path.join(path, file_name)

print("Loading dataset...")
df = pd.read_csv(full_path)

df.columns = df.columns.str.strip()

features = [
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Bwd Packet Length Mean'
]

df = df[features + ['Label']]
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# =====================
# 2. FEATURE ENGINEERING
# =====================
X = df[features].values

# LOG TRANSFORM untuk stabilitas
X[:,1] = np.log1p(X[:,1])
X[:,2] = np.log1p(X[:,2])
X[:,3] = np.log1p(X[:,3])

y = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1).values.reshape(-1,1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================
# 3. MODEL
# =====================
class SybilModel(nn.Module):
    def __init__(self):
        super(SybilModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.layer(x)

model = SybilModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.FloatTensor(y)

print("Training...")
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "sybil_detector_real.pt")
joblib.dump(scaler, "scaler_sybil.pkl")

print("✅ Training selesai & model disimpan!")