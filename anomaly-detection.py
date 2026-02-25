import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# 1. Tentukan Path File (Gunakan file Friday untuk Sybil/DDoS)
path = r'C:\Users\ASUS\.cache\kagglehub\datasets\chethuhn\network-intrusion-dataset\versions\1'
file_name = 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
full_path = os.path.join(path, file_name)

# 2. Load Data (Gunakan nrows jika RAM laptop terbatas)
print(f"Sedang memuat data dari: {file_name}")
df = pd.read_csv(full_path, nrows=50000) 

# Bersihkan nama kolom (dataset ini sering punya spasi di awal nama kolom)
df.columns = df.columns.str.strip()

# 3. Preprocessing Sederhana
# Fitur penting untuk Sybil: Bwd Packet Length, Flow Duration, Total Fwd Packets
features = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Bwd Packet Length Mean']
X = df[features].fillna(0).values
y = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1).values.reshape(-1, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Arsitektur Model Hybrid IDS
class SybilModel(nn.Module):
    def __init__(self):
        super(SybilModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(len(features), 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.layer(x)

# 5. Training Singkat
model = SybilModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

print("Memulai training...")
X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.FloatTensor(y)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

# 6. Simpan Model .pt
torch.save(model.state_dict(), "sybil_detector_real.pt")
print("Model 'sybil_detector_real.pt' berhasil disimpan!")