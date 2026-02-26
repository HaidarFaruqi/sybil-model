import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from scapy.all import sniff, IP, TCP
from collections import defaultdict
import time
import threading
import json
import platform
import logging
from logging.handlers import RotatingFileHandler

# =============================
# CONFIGURATION
# =============================

# Detect OS
IS_LINUX = platform.system() == "Linux"

# Wazuh integration config
WAZUH_INTEGRATION = True
WAZUH_LOG_FILE = "/var/log/ml_ids/alerts.log"

# Thresholds
THRESHOLD_CRITICAL = 10000.0
THRESHOLD_HIGH = 3000.0
THRESHOLD_MEDIUM = 1500.0
THRESHOLD_LOW = 500.0

# =============================
# LOGGING SETUP
# =============================

# Setup rotating log handler
os.makedirs(os.path.dirname(WAZUH_LOG_FILE), exist_ok=True)

logger = logging.getLogger('ML_IDS')
logger.setLevel(logging.INFO)

# Rotating file handler (10MB max, keep 5 backups)
handler = RotatingFileHandler(
    WAZUH_LOG_FILE,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
handler.setLevel(logging.INFO)

# JSON formatter untuk Wazuh
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

# Console handler untuk debug
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console)

logger.info("ML IDS started with Wazuh integration")

# =============================
# MODEL DEFINITION
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

threshold_base = checkpoint["threshold"]
if isinstance(threshold_base, torch.Tensor):
    threshold_base = threshold_base.item()

scaler = StandardScaler()
scaler.mean_ = checkpoint["scaler_mean"]
scaler.scale_ = checkpoint["scaler_scale"]
scaler.n_features_in_ = features

logger.info(f"Model loaded. Base threshold: {threshold_base:.6f}")

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
# WAZUH INTEGRATION
# =============================

def send_to_wazuh(alert_data):
    """
    Send alert to Wazuh via log file (syslog format)
    Wazuh agent will read from /var/log/ml_ids/alerts.log
    """
    if not WAZUH_INTEGRATION:
        return
    
    try:
        # Format as JSON for Wazuh ingestion
        wazuh_event = {
            "timestamp": alert_data["timestamp"],
            "rule": {
                "level": alert_data["level"],
                "description": alert_data["description"],
                "id": "100001",
                "groups": ["ml_anomaly", "ids", "intrusion_detection"]
            },
            "agent": {
                "name": platform.node()
            },
            "data": {
                "src_ip": alert_data["src_ip"],
                "src_port": alert_data["src_port"],
                "dst_ip": alert_data["dst_ip"],
                "dst_port": alert_data["dst_port"],
                "mse": round(alert_data["mse"], 2),
                "confidence": alert_data["confidence"],
                "flow_duration": round(alert_data["duration"], 2),
                "fwd_packets": alert_data["fwd_packets"],
                "bwd_packets": alert_data["bwd_packets"],
                "total_bytes": alert_data["total_bytes"]
            }
        }
        
        # Log to file (Wazuh agent will read this)
        logger.info(json.dumps(wazuh_event))
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to send to Wazuh: {e}")
        return False


# =============================
# DETECTION
# =============================

# Streaming services whitelist
STREAMING_IPS = [
    "142.250.", "142.251.", "172.217.", "74.125.",  # Google
    "202.152.",  # Google CDN Indonesia
]

def is_streaming_service(ip):
    return any(ip.startswith(prefix) for prefix in STREAMING_IPS)


def detect_anomaly_ml_only(key, flow, duration, total_packets, total_bytes, mse):
    src_ip = key[0][0]
    src_port = key[0][1]
    dst_ip = key[1][0]
    dst_port = key[1][1]
    
    # =============================
    # WHITELIST: Streaming Pattern
    # =============================
    is_streaming = (
        dst_port == 443 and
        duration > 30 and
        total_packets > 200 and
        is_streaming_service(dst_ip)
    )
    
    if is_streaming:
        if mse < 1000000:  # 1 million MSE untuk streaming = normal
            return ("NORMAL", "STREAMING", "Normal streaming service", 0)
    
    # =============================
    # ML-BASED DETECTION
    # =============================
    
    if mse > THRESHOLD_CRITICAL:
        confidence = "CRITICAL"
        level = 12
        description = f"Critical anomaly detected (MSE: {mse:.0f})"
        status = "ANOMALY"
    
    elif mse > THRESHOLD_HIGH:
        confidence = "HIGH"
        level = 10
        description = f"High confidence anomaly (MSE: {mse:.0f})"
        status = "ANOMALY"
    
    elif mse > THRESHOLD_MEDIUM:
        confidence = "MEDIUM"
        level = 7
        description = f"Medium confidence anomaly (MSE: {mse:.0f})"
        status = "ANOMALY"
    
    elif mse > THRESHOLD_LOW:
        confidence = "LOW"
        level = 5
        description = f"Suspicious pattern detected (MSE: {mse:.0f})"
        status = "SUSPICIOUS"
    
    else:
        confidence = "NORMAL"
        level = 0
        description = "Normal traffic pattern"
        status = "NORMAL"
    
    return (status, confidence, description, level)


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

            X_scaled = scaler.transform(feature_vector)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

            with torch.no_grad():
                reconstructed = model(X_tensor)
                mse = torch.mean((X_tensor - reconstructed) ** 2).item()

            status, confidence, description, level = detect_anomaly_ml_only(
                key, flow, duration, total_packets, total_bytes, mse
            )
            
            detection_stats["total_flows"] += 1
            detection_stats[status.lower()] += 1
            
            src_info = f"{key[0][0]}:{key[0][1]}"
            dst_info = f"{key[1][0]}:{key[1][1]}"
            flow_info = f"FWD:{flow['fwd_packets']} BWD:{flow['bwd_packets']}"
            
            if status == "ANOMALY":
                emoji = "🚨" if confidence == "CRITICAL" else "⚠️"
                print(f"{emoji} [{confidence}] {src_info} ↔ {dst_info} | {flow_info} | "
                      f"MSE:{mse:.1f} | {description}")
                
                # Send to Wazuh
                alert_data = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "level": level,
                    "description": description,
                    "confidence": confidence,
                    "src_ip": key[0][0],
                    "src_port": key[0][1],
                    "dst_ip": key[1][0],
                    "dst_port": key[1][1],
                    "mse": mse,
                    "duration": duration,
                    "fwd_packets": flow["fwd_packets"],
                    "bwd_packets": flow["bwd_packets"],
                    "total_bytes": total_bytes
                }
                
                if send_to_wazuh(alert_data):
                    print(f"   ↳ Sent to Wazuh (level {level})")
            
            elif status == "SUSPICIOUS":
                print(f"⚠️  [SUSPICIOUS] {src_info} ↔ {dst_info} | {flow_info} | MSE:{mse:.1f}")
            
            else:  # NORMAL
                if detection_stats["total_flows"] % 20 == 0:
                    print(f"✅ [NORMAL] {src_info} ↔ {dst_info} | MSE:{mse:.1f}")
            
            evaluated.append(key)
        
        for key in evaluated:
            del flows[key]


# =============================
# START
# =============================

print("\n" + "="*60)
print("🚀 ML-Based Anomaly Detection IDS (Linux + Wazuh)")
print("="*60)
print(f"System: {platform.system()} {platform.release()}")
print(f"Wazuh Integration: {'Enabled' if WAZUH_INTEGRATION else 'Disabled'}")
print(f"Log File: {WAZUH_LOG_FILE}")
print(f"Thresholds: LOW={THRESHOLD_LOW} MED={THRESHOLD_MEDIUM} HIGH={THRESHOLD_HIGH} CRIT={THRESHOLD_CRITICAL}")
print("="*60 + "\n")

threading.Thread(target=evaluate_flows, daemon=True).start()

try:
    # Note: Perlu root untuk sniff!
    sniff(filter="tcp", prn=process_packet, store=0)
except PermissionError:
    print("❌ Error: Need root privileges to capture packets")
    print("   Run with: sudo python3 ids_ml_wazuh.py")
except KeyboardInterrupt:
    print("\n👋 IDS stopped")
    logger.info("ML IDS stopped")