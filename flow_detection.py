import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from scapy.all import sniff, IP, TCP
from collections import defaultdict
import time
import threading

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

features = 14
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

print(f"✅ Model loaded. Training threshold: {threshold:.6f}")

# =============================
# BIDIRECTIONAL FLOW STORAGE
# =============================

flows = defaultdict(lambda: {
    "start": time.time(),
    "first_endpoint": None,  # tentukan arah fwd
    "fwd_packets": 0,
    "bwd_packets": 0,
    "fwd_bytes": 0,
    "bwd_bytes": 0,
    "fwd_lengths": [],
    "bwd_lengths": [],
    "syn": 0,
    "ack": 0,
    "rst": 0,
    "psh": 0
})

def normalize_flow_key(src, dst, sport, dport):
    """Bikin flow key bidirectional (canonical)"""
    endpoint_a = (src, sport)
    endpoint_b = (dst, dport)
    
    if endpoint_a < endpoint_b:
        return (endpoint_a, endpoint_b, True)  # True = paket ini fwd
    else:
        return (endpoint_b, endpoint_a, False)  # False = paket ini bwd

# =============================
# PACKET HANDLER
# =============================

def process_packet(packet):
    if IP in packet and TCP in packet:
        ip = packet[IP]
        tcp = packet[TCP]

        # Normalisasi key (bidirectional)
        (endpoint_a, endpoint_b, is_forward) = normalize_flow_key(
            ip.src, ip.dst, tcp.sport, tcp.dport
        )
        
        key = (endpoint_a, endpoint_b)
        flow = flows[key]
        
        # Set originator pertama kali
        if flow["first_endpoint"] is None:
            flow["first_endpoint"] = (ip.src, tcp.sport)
        
        # Tentukan arah paket ini
        current_endpoint = (ip.src, tcp.sport)
        is_fwd_packet = (current_endpoint == flow["first_endpoint"])
        
        packet_len = len(packet)
        
        if is_fwd_packet:
            flow["fwd_packets"] += 1
            flow["fwd_bytes"] += packet_len
            flow["fwd_lengths"].append(packet_len)
        else:
            flow["bwd_packets"] += 1
            flow["bwd_bytes"] += packet_len
            flow["bwd_lengths"].append(packet_len)

        # Flag counts (per flow, bukan per arah)
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

EVAL_INTERVAL = 10  # detik # set False setelah selesai kalibrasi
calibration_errors = []

def evaluate_flows():
    global threshold
    
    while True:
        time.sleep(EVAL_INTERVAL)
        
        now = time.time()
        evaluated = []

        for key, flow in list(flows.items()):
            duration = now - flow["start"]
            
            # Skip flow yang terlalu baru
            if duration < 1:
                continue
            
            total_packets = flow["fwd_packets"] + flow["bwd_packets"]
            total_bytes = flow["fwd_bytes"] + flow["bwd_bytes"]
            
            if total_packets == 0:
                continue
            
            # Hitung mean packet length per arah
            fwd_mean_len = np.mean(flow["fwd_lengths"]) if flow["fwd_lengths"] else 0
            bwd_mean_len = np.mean(flow["bwd_lengths"]) if flow["bwd_lengths"] else 0
            
            # 14 fitur SESUAI URUTAN TRAINING
            feature_vector = np.array([[
                duration,                                           # 0
                flow["fwd_packets"],                               # 1
                flow["bwd_packets"],                               # 2
                flow["fwd_bytes"],                                 # 3
                flow["bwd_bytes"],                                 # 4
                total_bytes / duration if duration > 0 else 0,    # 5
                total_packets / duration if duration > 0 else 0,  # 6
                fwd_mean_len,                                      # 7
                bwd_mean_len,                                      # 8
                flow["syn"],                                       # 9
                flow["rst"],                                       # 10
                flow["psh"],                                       # 11
                flow["ack"],                                       # 12
                total_bytes / total_packets                        # 13
            ]])

            X_scaled = scaler.transform(feature_vector)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

            with torch.no_grad():
                reconstructed = model(X_tensor)
                mse = torch.mean((X_tensor - reconstructed) ** 2).item()

            # Mode kalibrasi: kumpulkan error untuk bikin threshold baru
            # Mode deteksi
            if mse > threshold:
                print(f"🚨 [ANOMALY] {key[0][0]}:{key[0][1]} ↔ {key[1][0]}:{key[1][1]} | "
                        f"FWD:{flow['fwd_packets']} BWD:{flow['bwd_packets']} | "
                        f"SYN:{flow['syn']} | MSE: {mse:.6f} > {threshold:.6f}")
            else:
                print(f"✅ [NORMAL] {key[0][0]}:{key[0][1]} ↔ {key[1][0]}:{key[1][1]} | "
                        f"MSE: {mse:.6f}")
            
            evaluated.append(key)
        
        # Hapus flow yang sudah dievaluasi
        for key in evaluated:
            del flows[key]
        
        # Setelah kalibrasi 5 menit, hitung threshold baru



# =============================
# START
# =============================

print("\n" + "="*60)
print("🚀 Real-time IDS Starting...")
print("="*60)
print(f"🎯 MODE: DETECTION (threshold: {threshold:.6f})\n")

threading.Thread(target=evaluate_flows, daemon=True).start()

try:
    sniff(filter="tcp", prn=process_packet, store=0)
except KeyboardInterrupt:
    print("\n👋 IDS stopped")