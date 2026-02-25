import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from scapy.all import sniff, IP, TCP
from collections import defaultdict
import time

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
# LOAD MODEL
# =============================

checkpoint = torch.load("sybil_detector_real.pt", weights_only=False)

features = 14  # harus sama seperti training
model = AutoEncoder(features)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

threshold = checkpoint["threshold"]
if isinstance(threshold, torch.Tensor):
    threshold = threshold.item()

scaler = StandardScaler()
scaler.mean_ = checkpoint["scaler_mean"]
scaler.scale_ = checkpoint["scaler_scale"]
scaler.n_features_in_ = features

# =============================
# FLOW STORAGE
# =============================

flows = defaultdict(lambda: {
    "start": time.time(),
    "packets": 0,
    "bytes": 0,
    "syn": 0,
    "ack": 0,
    "rst": 0,
    "psh": 0
})

# =============================
# PACKET HANDLER
# =============================

def process_packet(packet):
    if IP in packet and TCP in packet:
        ip = packet[IP]
        tcp = packet[TCP]

        key = (ip.src, ip.dst, tcp.sport, tcp.dport)

        flow = flows[key]
        flow["packets"] += 1
        flow["bytes"] += len(packet)

        if tcp.flags & 0x02:
            flow["syn"] += 1
        if tcp.flags & 0x10:
            flow["ack"] += 1
        if tcp.flags & 0x04:
            flow["rst"] += 1
        if tcp.flags & 0x08:
            flow["psh"] += 1


# =============================
# EVALUATION LOOP
# =============================

def evaluate_flows():
    while True:
        time.sleep(10)

        for key, flow in list(flows.items()):
            duration = time.time() - flow["start"]

            feature_vector = np.array([[
                duration,
                flow["packets"],
                0,
                flow["bytes"],
                0,
                flow["bytes"] / duration if duration > 0 else 0,
                flow["packets"] / duration if duration > 0 else 0,
                0, 0,
                flow["syn"],
                flow["rst"],
                flow["psh"],
                flow["ack"],
                flow["bytes"] / flow["packets"] if flow["packets"] > 0 else 0
            ]])

            X_scaled = scaler.transform(feature_vector)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

            with torch.no_grad():
                reconstructed = model(X_tensor)
                mse = torch.mean((X_tensor - reconstructed) ** 2)

            if mse.item() > threshold:
                print(f"⚠ ANOMALY DETECTED from {key[0]} → {key[1]}")

        flows.clear()


# =============================
# START IDS
# =============================

print("🚀 Real-time IDS running...")
import threading
threading.Thread(target=evaluate_flows, daemon=True).start()

sniff(filter="tcp", prn=process_packet, store=0)