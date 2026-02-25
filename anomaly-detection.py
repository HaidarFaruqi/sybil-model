import torch
import torch.nn as nn
import joblib
import numpy as np
from scapy.all import sniff, IP, TCP
import time
from collections import defaultdict

# =====================
# MODEL (HARUS SAMA)
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

device = torch.device('cpu')
model = SybilModel()
model.load_state_dict(torch.load('sybil_detector_real.pt', map_location=device))
model.eval()

scaler = joblib.load('scaler_sybil.pkl')

print("✅ Model Loaded!")

# =====================
# FLOW STORAGE
# =====================
flows = defaultdict(lambda: {
    "start_time": time.time(),
    "fwd": 0,
    "bwd": 0,
    "bwd_lengths": []
})

WINDOW = 3  # seconds

def process_flow(key):
    flow = flows[key]
    duration = (time.time() - flow["start_time"]) * 1_000_000  # microseconds
    
    if flow["fwd"] + flow["bwd"] < 2:
        return
    
    fwd = np.log1p(flow["fwd"])
    bwd = np.log1p(flow["bwd"])
    bwd_mean = np.log1p(np.mean(flow["bwd_lengths"])) if flow["bwd_lengths"] else 0
    
    features = np.array([[duration, fwd, bwd, bwd_mean]])
    features_scaled = scaler.transform(features)
    
    with torch.no_grad():
        score = model(torch.FloatTensor(features_scaled)).item()
    
    if score > 0.85:
        print(f"🚨 SYBIL/DDOS DETECTED | Score: {score:.4f} | Flow: {key}")
    else:
        print(f"Healthy | Score: {score:.4f}")

def packet_callback(packet):
    if IP in packet and TCP in packet:
        key = (
            packet[IP].src,
            packet[IP].dst,
            packet[TCP].sport,
            packet[TCP].dport
        )
        
        flow = flows[key]
        
        if packet[IP].src == key[0]:
            flow["fwd"] += 1
        else:
            flow["bwd"] += 1
            flow["bwd_lengths"].append(len(packet))
        
        if time.time() - flow["start_time"] > WINDOW:
            process_flow(key)
            del flows[key]

print("🔍 Monitoring Port 80...")
sniff(filter="tcp port 80", prn=packet_callback, store=0)