import torch
import torch.nn as nn
import joblib
import numpy as np
from scapy.all import sniff, IP, TCP
import logging
import time
from collections import Counter

# 1. Konfigurasi Logging (Data untuk Bab IV)
logging.basicConfig(
    filename='/var/log/hybrid_alerts.log',
    level=logging.INFO,
    format='%(asctime)s [HYBRID-IDS] %(message)s'
)

# 2. Arsitektur Model (Wajib SAMA dengan saat training)
class SybilModel(nn.Module):
    def __init__(self):
        super(SybilModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(4, 16), # 4 Fitur input, 16 Hidden
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.layer(x)

# 3. Load Model dan Scaler
device = torch.device('cpu')
model = SybilModel()

try:
    model.load_state_dict(torch.load('sybil_detector_real.pt', map_location=device))
    model.eval()
    scaler = joblib.load('scaler_sybil.pkl')
    print("✅ Model & Scaler berhasil dimuat!")
except Exception as e:
    print(f"❌ Gagal memuat file: {e}")
    exit()

# Variabel bantu untuk menghitung fitur secara real-time
packet_count = 0
bwd_lengths = []
start_time = time.time()

def process_and_predict():
    global packet_count, bwd_lengths, start_time
    
    end_time = time.time()
    duration = end_time - start_time
    
    if packet_count > 0:
        # Ekstraksi 4 Fitur sesuai training kamu:
        # ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Bwd Packet Length Mean']
        fwd_pkts = packet_count * 0.6  # Estimasi pembagian fwd/bwd
        bwd_pkts = packet_count * 0.4
        bwd_mean = np.mean(bwd_lengths) if bwd_lengths else 0
        
        features = np.array([[duration, fwd_pkts, bwd_pkts, bwd_mean]])
        
        # WAJIB: Scaling data mentah
        features_scaled = scaler.transform(features)
        input_tensor = torch.FloatTensor(features_scaled)
        
        with torch.no_grad():
            score = model(input_tensor).item()
            
        # Jika skor tinggi, catat sebagai anomali
        if score > 0.85:
            msg = f"ANOMALY_DETECTED | Sybil Attack | Score: {score:.4f} | Pkts: {packet_count}"
            print(f"🚨 {msg}")
            logging.info(msg)
        else:
            print(f"Healthy Traffic (Score: {score:.4f})")

    # Reset counter untuk window berikutnya
    packet_count = 0
    bwd_lengths = []
    start_time = time.time()

def packet_callback(packet):
    global packet_count, bwd_lengths
    if IP in packet:
        packet_count += 1
        if packet.haslayer(TCP):
            bwd_lengths.append(len(packet))
        
        # Periksa setiap 2 detik (Windowing)
        if time.time() - start_time > 2:
            process_and_predict()

print("🔍 IDS AI sedang memantau Port 80 (Web)...")
sniff(filter="tcp port 80", prn=packet_callback, store=0)