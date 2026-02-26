import torch
import torch.nn as nn
import numpy as np
import joblib
import time
import threading
from scapy.all import sniff, IP, TCP
from collections import defaultdict

############################
# LOAD MODEL & SCALER
############################

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


INPUT_DIM = 14
THRESHOLD = 0.01  # sesuaikan dengan training kamu

model = AutoEncoder(INPUT_DIM)
model.load_state_dict(torch.load("sybil_detector_real.pt", weights_only=False))
model.eval()

scaler = joblib.load("scaler_real.pkl")

############################
# FLOW STORAGE
############################

FLOW_TIMEOUT = 10
flows = {}

############################
# PACKET HANDLER
############################

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
                "fwd_packets": [],
                "bwd_packets": []
            }

        flow = flows[key]
        flow["last"] = now
        flow["packets"].append(packet)

        if ip.src == key[0]:
            flow["fwd_packets"].append(packet)
        else:
            flow["bwd_packets"].append(packet)

############################
# FLOW EVALUATION
############################

def evaluate_flows():
    while True:
        now = time.time()
        expired = []

        for key, flow in list(flows.items()):
            if now - flow["last"] > FLOW_TIMEOUT:

                duration = flow["last"] - flow["start"]
                total_fwd = len(flow["fwd_packets"])
                total_bwd = len(flow["bwd_packets"])
                total_packets = len(flow["packets"])
                total_bytes = sum(len(p) for p in flow["packets"])

                flow_bytes_per_sec = total_bytes / duration if duration > 0 else 0
                flow_packets_per_sec = total_packets / duration if duration > 0 else 0
                avg_packet_size = total_bytes / total_packets if total_packets > 0 else 0

                # 14 fitur (sesuaikan dengan training kamu)
                features = np.array([[ 
                    duration,
                    total_fwd,
                    total_bwd,
                    total_bytes,
                    total_bytes,
                    flow_bytes_per_sec,
                    flow_packets_per_sec,
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
                    print(f"[NORMAL]  {key} | Loss: {loss:.6f}")

                expired.append(key)

        for key in expired:
            del flows[key]

        time.sleep(1)

############################
# MAIN
############################

if __name__ == "__main__":
    print("Flow-Based IDS Started...")
    
    t = threading.Thread(target=evaluate_flows)
    t.daemon = True
    t.start()

    sniff(filter="tcp", prn=process_packet, store=0)