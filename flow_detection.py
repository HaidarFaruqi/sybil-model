import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from scapy.all import sniff, IP, TCP
from collections import defaultdict
import time
import threading
import json
import os
from datetime import datetime

# =============================
# CONFIGURATION
# =============================

# Log file untuk Wazuh
LOG_FILE = "/var/log/ml_ids/alerts.log"

# Thresholds
THRESHOLD_CRITICAL = 100000.0   # Streaming threshold
THRESHOLD_HIGH = 10000.0
THRESHOLD_MEDIUM = 3000.0
THRESHOLD_LOW = 1000.0

# Streaming whitelist
STREAMING_IPS = [
    "142.250.", "142.251.", "172.217.", "74.125.",  # Google
    "202.152.",  # Google CDN Indonesia
    "52.123.", "40.126.", "20.190.",  # Microsoft
]

# =============================
# SETUP LOG FILE
# =============================

# Create log directory
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def write_log(log_data):
    """Write JSON log untuk Wazuh"""
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_data) + "\n")
        return True
    except Exception as e:
        print(f"❌ Failed to write log: {e}")
        return False

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

print("Loading model...")
checkpoint = torch.load("sybil_detector_real.pt", weights_only=False)

features = 14
model = AutoEncoder(features)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

scaler = StandardScaler()
scaler.mean_ = checkpoint["scaler_mean"]
scaler.scale_ = checkpoint["scaler_scale"]
scaler.n_features_in_ = features

print(f"✅ Model loaded")
print(f"📁 Log file: {LOG_FILE}")

# =============================
# FLOW STORAGE
# =============================

flows = defaultdict(lambda: {
    "start": time.time(),
    "first_endpoint": None,
    "fwd_packets": 0,
    "bwd_packets": 0,
    "fwd_bytes": 0,
    "bwd_bytes": 0,
    "fwd_lengths": [],
    "bwd_lengths": [],
    "syn": 0,
    "ack": 0,
    "rst": 0,
    "psh": 0,
    "fin": 0
})

def normalize_flow_key(src, dst, sport, dport):
    endpoint_a = (src, sport)
    endpoint_b = (dst, dport)
    
    if endpoint_a < endpoint_b:
        return (endpoint_a, endpoint_b, True)
    else:
        return (endpoint_b, endpoint_a, False)

# =============================
# PACKET HANDLER
# =============================

def process_packet(packet):
    if IP in packet and TCP in packet:
        ip = packet[IP]
        tcp = packet[TCP]

        (endpoint_a, endpoint_b, is_forward) = normalize_flow_key(
            ip.src, ip.dst, tcp.sport, tcp.dport
        )
        
        key = (endpoint_a, endpoint_b)
        flow = flows[key]
        
        if flow["first_endpoint"] is None:
            flow["first_endpoint"] = (ip.src, tcp.sport)
        
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

        if tcp.flags & 0x02: flow["syn"] += 1
        if tcp.flags & 0x10: flow["ack"] += 1
        if tcp.flags & 0x04: flow["rst"] += 1
        if tcp.flags & 0x08: flow["psh"] += 1
        if tcp.flags & 0x01: flow["fin"] += 1


# =============================
# DETECTION
# =============================

def is_streaming_service(ip):
    return any(ip.startswith(prefix) for prefix in STREAMING_IPS)


def detect_anomaly(key, flow, duration, total_packets, total_bytes, mse):
    """Pure ML detection dengan whitelist streaming"""
    
    src_ip = key[0][0]
    src_port = key[0][1]
    dst_ip = key[1][0]
    dst_port = key[1][1]
    
    # =============================
    # WHITELIST: Streaming Services
    # =============================
    is_streaming = (
        dst_port == 443 and
        duration > 30 and
        total_packets > 200 and
        is_streaming_service(dst_ip)
    )
    
    if is_streaming:
        # Streaming: very high threshold
        if mse < THRESHOLD_CRITICAL:
            return ("NORMAL", "STREAMING", 0, "Normal streaming service")
    
    # =============================
    # ML-BASED CLASSIFICATION
    # =============================
    
    if mse > THRESHOLD_HIGH:
        confidence = "CRITICAL" if mse > THRESHOLD_CRITICAL else "HIGH"
        level = 12 if mse > THRESHOLD_CRITICAL else 10
        description = f"High confidence network anomaly (MSE: {mse:.0f})"
        status = "ANOMALY"
    
    elif mse > THRESHOLD_MEDIUM:
        confidence = "MEDIUM"
        level = 7
        description = f"Medium confidence anomaly (MSE: {mse:.0f})"
        status = "ANOMALY"
    
    elif mse > THRESHOLD_LOW:
        confidence = "LOW"
        level = 5
        description = f"Suspicious network pattern (MSE: {mse:.0f})"
        status = "SUSPICIOUS"
    
    else:
        confidence = "NORMAL"
        level = 0
        description = "Normal traffic pattern"
        status = "NORMAL"
    
    return (status, confidence, level, description)


# =============================
# EVALUATION LOOP
# =============================

EVAL_INTERVAL = 10

detection_stats = {
    "total_flows": 0,
    "normal": 0,
    "suspicious": 0,
    "anomaly": 0
}

def evaluate_flows():
    while True:
        time.sleep(EVAL_INTERVAL)
        
        now = time.time()
        evaluated = []

        for key, flow in list(flows.items()):
            duration = now - flow["start"]
            
            if duration < 1:
                continue
            
            total_packets = flow["fwd_packets"] + flow["bwd_packets"]
            total_bytes = flow["fwd_bytes"] + flow["bwd_bytes"]
            
            if total_packets == 0:
                continue
            
            # Calculate features
            fwd_mean_len = np.mean(flow["fwd_lengths"]) if flow["fwd_lengths"] else 0
            bwd_mean_len = np.mean(flow["bwd_lengths"]) if flow["bwd_lengths"] else 0
            
            feature_vector = np.array([[
                duration,
                flow["fwd_packets"],
                flow["bwd_packets"],
                flow["fwd_bytes"],
                flow["bwd_bytes"],
                total_bytes / duration if duration > 0 else 0,
                total_packets / duration if duration > 0 else 0,
                fwd_mean_len,
                bwd_mean_len,
                flow["syn"],
                flow["rst"],
                flow["psh"],
                flow["ack"],
                total_bytes / total_packets
            ]])

            # ML Inference
            X_scaled = scaler.transform(feature_vector)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

            with torch.no_grad():
                reconstructed = model(X_tensor)
                mse = torch.mean((X_tensor - reconstructed) ** 2).item()

            # Detect
            status, confidence, level, description = detect_anomaly(
                key, flow, duration, total_packets, total_bytes, mse
            )
            
            # Update stats
            detection_stats["total_flows"] += 1
            detection_stats[status.lower()] += 1
            
            # Format info
            src_info = f"{key[0][0]}:{key[0][1]}"
            dst_info = f"{key[1][0]}:{key[1][1]}"
            
            # =============================
            # PRINT TO CONSOLE
            # =============================
            if status == "ANOMALY":
                emoji = "🚨" if confidence == "CRITICAL" else "⚠️"
                print(f"{emoji} [{confidence}] {src_info} ↔ {dst_info} | "
                      f"FWD:{flow['fwd_packets']} BWD:{flow['bwd_packets']} | "
                      f"MSE:{mse:.1f} | {description}")
            
            elif status == "SUSPICIOUS":
                print(f"⚠️  [SUSPICIOUS] {src_info} ↔ {dst_info} | MSE:{mse:.1f}")
            
            else:  # NORMAL
                if detection_stats["total_flows"] % 20 == 0:
                    print(f"✅ [NORMAL] {src_info} ↔ {dst_info} | MSE:{mse:.1f}")
            
            # =============================
            # WRITE TO LOG (untuk Wazuh)
            # =============================
            if status in ["ANOMALY", "SUSPICIOUS"]:  # Hanya log anomaly
                log_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "event_type": "ml_anomaly",
                    "status": status,
                    "confidence": confidence,
                    "level": level,
                    "description": description,
                    "src_ip": key[0][0],
                    "src_port": key[0][1],
                    "dst_ip": key[1][0],
                    "dst_port": key[1][1],
                    "mse": round(mse, 2),
                    "flow_duration": round(duration, 2),
                    "fwd_packets": flow["fwd_packets"],
                    "bwd_packets": flow["bwd_packets"],
                    "total_bytes": total_bytes,
                    "syn_count": flow["syn"],
                    "packets_per_sec": round(total_packets / duration, 2) if duration > 0 else 0
                }
                
                if write_log(log_entry):
                    print(f"   ↳ Logged to {LOG_FILE}")
            
            evaluated.append(key)
        
        # Cleanup
        for key in evaluated:
            del flows[key]


# =============================
# START
# =============================

print("\n" + "="*60)
print("🚀 ML Anomaly Detection IDS - Simple Log Mode")
print("="*60)
print(f"Thresholds:")
print(f"  LOW:      {THRESHOLD_LOW}")
print(f"  MEDIUM:   {THRESHOLD_MEDIUM}")
print(f"  HIGH:     {THRESHOLD_HIGH}")
print(f"  CRITICAL: {THRESHOLD_CRITICAL}")
print(f"\nLog file: {LOG_FILE}")
print("="*60 + "\n")

threading.Thread(target=evaluate_flows, daemon=True).start()

try:
    sniff(filter="tcp", prn=process_packet, store=0)
except KeyboardInterrupt:
    print("\n👋 IDS stopped")
    print(f"Total flows processed: {detection_stats['total_flows']}")
    print(f"  Normal: {detection_stats['normal']}")
    print(f"  Suspicious: {detection_stats['suspicious']}")
    print(f"  Anomaly: {detection_stats['anomaly']}")