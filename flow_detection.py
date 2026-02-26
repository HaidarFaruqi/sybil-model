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

threshold = checkpoint["threshold"]  # ← FIX TYPO
if isinstance(threshold, torch.Tensor):
    threshold = threshold.item()

scaler = StandardScaler()
scaler.mean_ = checkpoint["scaler_mean"]
scaler.scale_ = checkpoint["scaler_scale"]
scaler.n_features_in_ = features

print(f"✅ Model loaded. Base threshold: {threshold:.6f}")

# =============================
# MULTI-THRESHOLD CONFIGURATION
# =============================

THRESHOLD_HIGH = 1500.0      # High confidence anomaly → Block
THRESHOLD_MEDIUM = 500.0     # Medium confidence → Alert high priority
THRESHOLD_LOW = 200.0        # Suspicious → Alert low priority / Log

print(f"🎯 Threshold configuration:")
print(f"   HIGH (block):       {THRESHOLD_HIGH}")
print(f"   MEDIUM (alert):     {THRESHOLD_MEDIUM}")
print(f"   LOW (suspicious):   {THRESHOLD_LOW}")

# =============================
# BIDIRECTIONAL FLOW STORAGE
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

# =============================
# SOURCE IP AGGREGATION (for detecting distributed attacks)
# =============================

src_stats = defaultdict(lambda: {
    "total_flows": 0,
    "suspicious_count": 0,
    "anomaly_count": 0,
    "total_syn": 0,
    "unique_dst_ports": set(),
    "unique_dst_ips": set(),
    "last_alert": 0
})

def normalize_flow_key(src, dst, sport, dport):
    """Bikin flow key bidirectional (canonical)"""
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

        # Flag counts
        if tcp.flags & 0x02:  # SYN
            flow["syn"] += 1
        if tcp.flags & 0x10:  # ACK
            flow["ack"] += 1
        if tcp.flags & 0x04:  # RST
            flow["rst"] += 1
        if tcp.flags & 0x08:  # PSH
            flow["psh"] += 1
        if tcp.flags & 0x01:  # FIN
            flow["fin"] += 1


# =============================
# CONTEXTUAL THRESHOLD (per service)
# =============================

def get_adaptive_threshold(dst_port, duration, packet_count):
    """
    Return adaptive threshold berdasarkan service type
    """
    # DNS - expect very consistent pattern
    if dst_port == 53:
        return 150.0  # Very strict
    
    # HTTP/HTTPS - high variability normal
    if dst_port in [80, 443]:
        # Short flows (handshake) should have low MSE
        if packet_count < 10 and duration < 5:
            return 300.0
        # Established connections - more permissive
        return THRESHOLD_HIGH
    
    # SSH/RDP - long sessions normal
    if dst_port in [22, 3389]:
        if duration > 60:
            return 3000.0  # Very permissive for long sessions
        return THRESHOLD_MEDIUM
    
    # Database ports - expect consistent
    if dst_port in [3306, 5432, 1433, 27017]:
        return 600.0  # Stricter
    
    # Mail services
    if dst_port in [25, 587, 110, 143, 993, 995]:
        return 800.0
    
    # Default
    return THRESHOLD_HIGH


# =============================
# MULTI-LAYER DETECTION
# =============================

def detect_anomaly(key, flow, duration, total_packets, total_bytes, mse):
    """
    Multi-layer detection:
    1. Whitelist (skip obvious benign)
    2. Rule-based (obvious attacks)
    3. Aggregation (distributed attacks)
    4. Contextual ML (adaptive threshold)
    5. Multi-threshold ML (suspicious vs attack)
    
    Returns: ("NORMAL" | "SUSPICIOUS" | "ANOMALY", reason, severity)
    """
    
    src_ip = key[0][0]
    src_port = key[0][1]
    dst_ip = key[1][0]
    dst_port = key[1][1]
    
    # =============================
    # LAYER 0: WHITELIST (skip benign)
    # =============================
    
    # Skip monitoring/logging services
    if dst_port in [1514, 514, 5514]:  # Syslog
        return ("NORMAL", "whitelist:syslog", 0)
    
    # Skip long-lived balanced interactive sessions
    if duration > 60 and total_packets > 50:
        ratio = flow["fwd_packets"] / flow["bwd_packets"] if flow["bwd_packets"] > 0 else 999
        if 0.3 <= ratio <= 3.0 and dst_port in [22, 3389]:
            return ("NORMAL", "whitelist:interactive_session", 0)
    
    # =============================
    # LAYER 1: RULE-BASED (obvious attacks)
    # =============================
    
    pps = total_packets / duration if duration > 0 else 0
    bps = total_bytes / duration if duration > 0 else 0
    avg_packet_size = total_bytes / total_packets if total_packets > 0 else 0
    
    # Rule 1: SYN Flood (banyak SYN, no response)
    if flow["syn"] > 10 and flow["bwd_packets"] == 0:
        return ("ANOMALY", "rule:syn_flood", 10)
    
    # Rule 2: High SYN ratio (port scan signature)
    syn_ratio = flow["syn"] / total_packets if total_packets > 0 else 0
    if syn_ratio > 0.5 and total_packets > 10 and flow["bwd_packets"] < 5:
        return ("ANOMALY", "rule:high_syn_ratio", 9)
    
    # Rule 3: Packet flood (high PPS, small packets)
    if pps > 100 and avg_packet_size < 100:
        return ("ANOMALY", "rule:packet_flood", 9)
    
    # Rule 4: Unbalanced flow (potential flood/scan)
    if flow["fwd_packets"] > 100 and flow["bwd_packets"] < 10:
        return ("ANOMALY", "rule:unbalanced_flow", 8)
    
    # Rule 5: Excessive packets (volumetric attack)
    if total_packets > 500 and duration < 10:
        return ("ANOMALY", "rule:excessive_packets", 8)
    
    # Rule 6: High bandwidth (potential DDoS)
    if bps > 10_000_000:  # 10 MB/s per flow
        return ("ANOMALY", "rule:high_bandwidth", 7)
    
    # =============================
    # LAYER 2: AGGREGATION (distributed attacks)
    # =============================
    
    # Update source statistics
    src_stats[src_ip]["total_flows"] += 1
    src_stats[src_ip]["total_syn"] += flow["syn"]
    src_stats[src_ip]["unique_dst_ports"].add(dst_port)
    src_stats[src_ip]["unique_dst_ips"].add(dst_ip)
    
    stats = src_stats[src_ip]
    
    # Aggregation Rule 1: Port scanning
    if len(stats["unique_dst_ports"]) > 20:
        now = time.time()
        if now - stats["last_alert"] > 60:  # Alert throttling (1x per minute)
            stats["last_alert"] = now
            return ("ANOMALY", "aggregate:port_scan", 9)
    
    # Aggregation Rule 2: Horizontal scanning (many dst IPs)
    if len(stats["unique_dst_ips"]) > 50:
        now = time.time()
        if now - stats["last_alert"] > 60:
            stats["last_alert"] = now
            return ("ANOMALY", "aggregate:horizontal_scan", 8)
    
    # Aggregation Rule 3: High SYN from source (distributed SYN flood)
    if stats["total_syn"] > 100 and stats["total_flows"] > 10:
        syn_per_flow = stats["total_syn"] / stats["total_flows"]
        if syn_per_flow > 5:
            return ("ANOMALY", "aggregate:syn_flood_distributed", 8)
    
    # =============================
    # LAYER 3: CONTEXTUAL ML (adaptive threshold)
    # =============================
    
    adaptive_threshold = get_adaptive_threshold(dst_port, duration, total_packets)
    
    # =============================
    # LAYER 4: MULTI-THRESHOLD ML
    # =============================
    
    # Use adaptive threshold for high confidence
    if mse > adaptive_threshold:
        severity = min(10, int(mse / adaptive_threshold))  # Scale severity
        return ("ANOMALY", f"ml:high_confidence(adaptive={adaptive_threshold:.0f})", severity)
    
    # Medium threshold
    if mse > THRESHOLD_MEDIUM:
        src_stats[src_ip]["anomaly_count"] += 1
        return ("ANOMALY", f"ml:medium_confidence", 6)
    
    # Low threshold (suspicious)
    if mse > THRESHOLD_LOW:
        src_stats[src_ip]["suspicious_count"] += 1
        
        # If multiple suspicious flows from same source → escalate
        if stats["suspicious_count"] > 5:
            return ("ANOMALY", "ml:multiple_suspicious_flows", 5)
        
        return ("SUSPICIOUS", f"ml:low_confidence", 3)
    
    # Normal
    return ("NORMAL", "ml:reconstruction_ok", 0)


# =============================
# EVALUATION LOOP
# =============================

EVAL_INTERVAL = 10  # seconds

# Statistics
detection_stats = {
    "total_flows": 0,
    "normal": 0,
    "suspicious": 0,
    "anomaly": 0,
    "by_reason": defaultdict(int)
}

def evaluate_flows():
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

            # =============================
            # MULTI-LAYER DETECTION
            # =============================
            
            status, reason, severity = detect_anomaly(
                key, flow, duration, total_packets, total_bytes, mse
            )
            
            # Update statistics
            detection_stats["total_flows"] += 1
            detection_stats[status.lower()] += 1
            detection_stats["by_reason"][reason] += 1
            
            # Format output
            src_info = f"{key[0][0]}:{key[0][1]}"
            dst_info = f"{key[1][0]}:{key[1][1]}"
            flow_info = f"FWD:{flow['fwd_packets']} BWD:{flow['bwd_packets']} SYN:{flow['syn']}"
            pps = total_packets / duration if duration > 0 else 0
            
            if status == "ANOMALY":
                emoji = "🚨" if severity >= 8 else "⚠️"
                print(f"{emoji} [ANOMALY] {src_info} ↔ {dst_info} | {flow_info} | "
                      f"MSE:{mse:.1f} PPS:{pps:.0f} | Reason:{reason} Severity:{severity}")
            
            elif status == "SUSPICIOUS":
                print(f"⚠️  [SUSPICIOUS] {src_info} ↔ {dst_info} | {flow_info} | "
                      f"MSE:{mse:.1f} | Reason:{reason}")
            
            else:  # NORMAL
                # Only print normal flows occasionally (reduce noise)
                if detection_stats["total_flows"] % 10 == 0:
                    print(f"✅ [NORMAL] {src_info} ↔ {dst_info} | MSE:{mse:.1f}")
            
            evaluated.append(key)
        
        # Hapus flow yang sudah dievaluasi
        for key in evaluated:
            del flows[key]
        
        # Print statistics periodically
        if detection_stats["total_flows"] > 0 and detection_stats["total_flows"] % 50 == 0:
            print_statistics()


def print_statistics():
    """Print detection statistics"""
    total = detection_stats["total_flows"]
    normal = detection_stats["normal"]
    suspicious = detection_stats["suspicious"]
    anomaly = detection_stats["anomaly"]
    
    print("\n" + "="*60)
    print("📊 DETECTION STATISTICS")
    print("="*60)
    print(f"Total flows analyzed: {total}")
    print(f"  ✅ Normal:      {normal:4d} ({normal/total*100:5.1f}%)")
    print(f"  ⚠️  Suspicious:  {suspicious:4d} ({suspicious/total*100:5.1f}%)")
    print(f"  🚨 Anomaly:     {anomaly:4d} ({anomaly/total*100:5.1f}%)")
    print()
    print("Top detection reasons:")
    for reason, count in sorted(detection_stats["by_reason"].items(), 
                                 key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - {reason}: {count}")
    print("="*60 + "\n")


# =============================
# START
# =============================

print("\n" + "="*60)
print("🚀 Real-time Multi-Layer IDS Starting...")
print("="*60)
print("Detection layers:")
print("  1. Whitelist (skip obvious benign)")
print("  2. Rule-based (SYN flood, packet flood, scans)")
print("  3. Aggregation (distributed attacks)")
print("  4. Contextual ML (adaptive per service)")
print("  5. Multi-threshold ML (high/medium/low)")
print("="*60 + "\n")

threading.Thread(target=evaluate_flows, daemon=True).start()

try:
    sniff(filter="tcp", prn=process_packet, store=0)
except KeyboardInterrupt:
    print("\n👋 IDS stopped")
    print_statistics()