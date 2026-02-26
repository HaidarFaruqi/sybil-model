import torch
import torch.nn as nn
import numpy as np
import time
import threading
from scapy.all import sniff, IP, TCP
from sklearn.preprocessing import StandardScaler

###################################
# AUTOENCODER DEFINITION
###################################

class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
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
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

###################################
# LOAD CHECKPOINT (MODEL + SCALER + THRESHOLD)
###################################

INPUT_DIM = 14

checkpoint = torch.load("sybil_detector_real.pt", weights_only=False)

model = AutoEncoder(INPUT_DIM)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# rebuild scaler
scaler = StandardScaler()
scaler.mean_ = checkpoint["scaler_mean"]
scaler.scale_ = checkpoint["scaler_scale"]
scaler.n_features_in_ = len(scaler.mean_)

THRESHOLD = checkpoint["threshold"]

print("Model, scaler, dan threshold berhasil dimuat.")

###################################
# FLOW STORAGE
###################################

FLOW_TIMEOUT = 10
flows = {}

###################################
# PACKET PROCESSING
###################################

def process_packet(packet):
    if IP in packet and TCP in packet:
        ip = packet[IP]
        tcp = packet[TCP]

        key = (ip.src, ip.dst, tcp.sport, tcp.dport)
        now = time.time()

        if key not in flows:
            flows[key] = {
                "start": now,
                "last": now,
                "packets": [],
                "fwd": [],
                "bwd": []
            }

        flow = flows[key]
        flow["last"] = now
        flow["packets"].append(packet)

        if ip.src == key[0]:
            flow["fwd"].append(packet)
        else:
            flow["bwd"].append(packet)

###################################
# FLOW EVALUATION
###################################

def evaluate_flows():
    while True:
        now = time.time()
        expired = []

        for key, flow in list(flows.items()):
            if now - flow["last"] > FLOW_TIMEOUT:

                duration = flow["last"] - flow["start"]
                total_fwd = len(flow["fwd"])
                total_bwd = len(flow["bwd"])
                total_packets = len(flow["packets"])
                total_bytes = sum(len(p) for p in flow["packets"])

                bytes_per_sec = total_bytes / duration if duration > 0 else 0
                packets_per_sec = total_packets / duration if duration > 0 else 0
                avg_packet_size = total_bytes / total_packets if total_packets > 0 else 0

                # HARUS SAMA DENGAN FITUR TRAINING
                features = np.array([[ 
                    duration,
                    total_fwd,
                    total_bwd,
                    total_bytes,
                    total_bytes,
                    bytes_per_sec,
                    packets_per_sec,
                    0, 0, 0, 0, 0, 0,
                    avg_packet_size
                ]])

                features_scaled = scaler.transform(features)
                tensor = torch.tensor(features_scaled, dtype=torch.float32)

                with torch.no_grad():
                    reconstructed = model(tensor)
                    loss = torch.mean((tensor - reconstructed) ** 2).item()

                if loss > THRESHOLD:
                    print(f"[ANOMALY] {key} | Loss: {loss:.6f}")
                else:
                    print(f"[NORMAL ] {key} | Loss: {loss:.6f}")

                expired.append(key)

        for key in expired:
            del flows[key]

        time.sleep(1)

###################################
# MAIN
###################################

if __name__ == "__main__":
    print("Flow-Based IDS Started...")
    print(f"Threshold: {THRESHOLD}")

    t = threading.Thread(target=evaluate_flows)
    t.daemon = True
    t.start()

    sniff(filter="tcp", prn=process_packet, store=0)