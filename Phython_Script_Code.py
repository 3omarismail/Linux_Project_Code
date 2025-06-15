import csv
import ipaddress
import socket
from collections import defaultdict, deque
import time
import threading
from scapy.all import sniff, IP, TCP, UDP, DNSQR, Raw, packet
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import joblib # For loading the trained model and scaler
import pandas as pd 

# === CONFIGURATION ===
interface = input("Enter network interface (e.g., wlan0, eth0): ").strip()
top_n = int(input("Enter how many top IPs to track: ").strip())
threshold_kb = float(input("Enter traffic threshold (in KB): ").strip())
duration_seconds = int(input("Enter monitoring duration (in seconds, 0 for continuous until plot close): ").strip())

# === IDS FEATURE AND MODEL CONFIGURATION ===

SCALER_PATH = '/home/roudy/Downloads/minmax_scaler.pkl'
MODEL_PATH = '/home/roudy/Downloads/mlp_model.pkl'
# --- HARDCODED IDS_FEATURE_NAMES ---


IDS_FEATURE_NAMES = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'num_failed_logins', 'logged_in', 'root_shell', 'su_attempted', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate']

print(f"Using hardcoded IDS_FEATURE_NAMES with {len(IDS_FEATURE_NAMES)} features.")


# --- CATEGORICAL ENCODING MAPPINGS ---
# These mappings must exactly match the encoding used during your model's training.
PROTOCOL_ENCODING = {
    'icmp': 0,
    'tcp': 1,
    'udp': 2,
    'other': 3 
}

SERVICE_ENCODING = {
    'IRC': 0, 'X11': 1, 'Z39_50': 2, 'auth': 3, 'bgp': 4, 'courier': 5, 'csnet_ns': 6,
    'ctf': 7, 'daytime': 8, 'discard': 9, 'domain': 10, 'domain_u': 11, 'echo': 12,
    'eco_i': 13, 'ecr_i': 14, 'efs': 15, 'exec': 16, 'finger': 17, 'ftp': 18,
    'ftp_data': 19, 'gopher': 20, 'hostnames': 21, 'http': 22, 'http_443': 23,
    'http_8001': 24, 'imap4': 25, 'iso_tsap': 26, 'klogin': 27, 'kshell': 28,
    'ldap': 29, 'link': 30, 'login': 31, 'mtp': 32, 'name': 33, 'netbios_dgm': 34,
    'netbios_ns': 35, 'netbios_ssn': 36, 'netstat': 37, 'nnsp': 38, 'nntp': 39,
    'ntp_u': 40, 'other': 41, 'pm_dump': 42, 'pop_2': 43, 'pop_3': 44, 'printer': 45,
    'private': 46, 'red_i': 47, 'remote_job': 48, 'rje': 49, 'shell': 50, 'smtp': 51,
    'sql_net': 52, 'ssh': 53, 'sunrpc': 54, 'supdup': 55, 'systat': 56, 'telnet': 57,
    'tim_i': 58, 'time': 59, 'urh_i': 60, 'urp_i': 61, 'uucp': 62, 'uucp_path': 63,
    'vmnet': 64, 'whois': 65
}

FLAG_ENCODING = {
    'OTH': 0, 'REJ': 1, 'RSTO': 2, 'RSTOS0': 3, 'RSTR': 4, 'S0': 5,
    'S1': 6, 'S2': 7, 'S3': 8, 'SF': 9, 'SH': 10
}


# === IDS MODEL LOADING ===
scaler = None
final_model = None

print("\n--- Loading Trained Scaler and Model ---")
try:
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print(f"MinMaxScaler loaded from {SCALER_PATH} successfully.")
    else:
        print(f"ERROR: Scaler file not found at {SCALER_PATH}. Features will not be scaled.")

    if os.path.exists(MODEL_PATH):
           final_model = joblib.load(MODEL_PATH)
        print(f"ML Model loaded from {MODEL_PATH} successfully.")
    else:
        print(f"ERROR: ML Model file not found at {MODEL_PATH}. Anomaly detection will be disabled.")

except Exception as e:
    print(f"CRITICAL ERROR: Could not load ML model or scaler: {e}. Anomaly detection will be disabled.")
    scaler = None
    final_model = None


# === DATA STRUCTURES ===
traffic_data = defaultdict(lambda: {'sent': 0, 'received': 0, 'total': 0})
traffic_per_second = defaultdict(lambda: deque(maxlen=60))
protocol_usage = defaultdict(int)
bandwidth_history = deque(maxlen=60)
total_traffic_per_second = defaultdict(int) # For total bandwidth

lock = threading.Lock()
stop_sniffing_flag = threading.Event()

# === IDS Specific Data Structures ===
# Active connections: Stores state for currently ongoing connections
# Key: (src_ip, src_port, dst_ip, dst_port, protocol)
active_connections = {}
# History of recently closed/processed connections for time-window features
# Stores (timestamp, src_ip, dst_ip, dst_port, protocol, service, flag)
connection_summary_history = deque(maxlen=10000) # Store up to 10,000 recent connections
# History of connections per host for host-based features (last 100 connections)
host_connection_history = defaultdict(lambda: deque(maxlen=100)) # Key: dst_ip
host_service_connection_history = defaultdict(lambda: deque(maxlen=100)) # Key: (dst_ip, service)

# This queue will hold extracted feature dictionaries for prediction
extracted_features_queue = deque()

# --- ADDED: Deque for recent prediction messages ---
recent_predictions_display = deque(maxlen=10) # Store the last 10 prediction messages for display

def get_display_name(ip):
    """
    Attempts to get the hostname for an IP. Falls back to IP if hostname lookup fails.
    No GeoIP lookup performed.
    """
    try:
        # Prevent reverse DNS lookup on private IPs to speed things up (optional but good practice)
        if ipaddress.ip_address(ip).is_private:
            hostname = ip
        else:
            hostname = socket.gethostbyaddr(ip)[0]
    except (socket.herror, socket.gaierror): # Catch specific error for unknown host or address
        hostname = ip # Fallback to IP if hostname lookup fails
    except Exception: # Catch any other unexpected errors
        hostname = ip # Fallback to IP
    return hostname


def get_service_name(proto, port):
    if proto == 'tcp':
        if port == 80: return 'http'
        if port == 443: return 'http_443' 
        if port == 21: return 'ftp'
        if port == 22: return 'ssh'
        if port == 23: return 'telnet'
        if port == 25: return 'smtp'
        if port == 110: return 'pop_3'
        if port == 143: return 'imap4' 
        if port == 3389: return 'remote_job' 
        if port == 139 or port == 445: return 'netbios_ns' 
        if port == 1720: return 'h323' 
        if port == 8001: return 'http_8001' 
    elif proto == 'udp':
        if port == 53: return 'domain' 
        if port == 67 or port == 68: return 'dhcp' 
        if port == 161 or port == 162: return 'snmp'
        if port == 123: return 'ntp_u'
    elif proto == 'icmp':
        return 'eco_i' 
    
    return 'other' 


def get_tcp_flags_kdd_style(flags_int):
    # This function maps Scapy's TCP flag integer (bitmask) to a KDD-style string flag.
    # The returned string MUST be a key in FLAG_ENCODING dictionary.
    
    # SF: Normal connection setup and tear-down (SYN, SYN-ACK, FIN, ACK)
    if (flags_int & 0x02) and (flags_int & 0x10) and (flags_int & 0x01): # SYN, ACK, FIN
        return 'SF'
    elif (flags_int & 0x02) and (flags_int & 0x10): # SYN, ACK
        return 'SF' 
    elif (flags_int & 0x01) and (flags_int & 0x10): # FIN, ACK
        return 'SF' 
    # S0: Connection establishment attempt, no reply (SYN only)
    elif (flags_int & 0x02): # SYN
        return 'S0'
    # REJ: Connection rejected (RST, ACK)
    elif (flags_int & 0x04) and (flags_int & 0x10): # RST, ACK
        return 'REJ'
    # RSTO: Reset (RST only)
    elif (flags_int & 0x04): # RST
        return 'RSTO'
    # FIN: Connection tear-down (FIN only - can be normal or odd)
    elif (flags_int & 0x01): # FIN
        return 'SF' # Often treated as SF if part of a normal flow ending. Can be 'FIN' if specific.
    # OTH: Other/unclassified flags
    elif flags_int == 0x00: # No flags set
        return 'OTH'
    
    return 'OTH' # Default for any other combination not explicitly handled


def process_packet(packet):
    if IP in packet:
        ip_src = packet[IP].src
        ip_dst = packet[IP].dst
        size = len(packet)
        timestamp = time.time() 

        proto = 'other' # Default to 'other' in lowercase for consistency
        sport = None
        dport = None
        flags_kdd = 'OTH' # Default KDD-style flag
        service = 'other' # Default service

        if TCP in packet:
            proto = 'tcp'
            sport = packet[TCP].sport
            dport = packet[TCP].dport
            flags_kdd = get_tcp_flags_kdd_style(packet[TCP].flags)
            service = get_service_name('tcp', dport)
        elif UDP in packet:
            proto = 'udp'
            sport = packet[UDP].sport
            dport = packet[UDP].dport
            service = get_service_name('udp', dport)
        elif packet[IP].proto == 1: # ICMP
            proto = 'icmp'
            service = get_service_name('icmp', None) 
            sport = None
            dport = None

        connection_key = (ip_src, sport, ip_dst, dport, proto)

        with lock:
            traffic_data[ip_src]['sent'] += size
            traffic_data[ip_src]['total'] += size
            traffic_data[ip_dst]['received'] += size
            traffic_data[ip_dst]['total'] += size
            traffic_per_second[ip_src].append((int(timestamp), size))
            traffic_per_second[ip_dst].append((int(timestamp), size))
            protocol_usage[proto] += size
            total_traffic_per_second[int(timestamp)] += size

            # === IDS Feature Extraction Logic ===
            if connection_key not in active_connections:
                # New connection
                active_connections[connection_key] = {
                    'start_time': timestamp,
                    'last_packet_time': timestamp,
                    'src_bytes': size,
                    'dst_bytes': 0, # Assuming first packet is from src
                    'protocol_type': proto,
                    'service': service,
                    'flag': flags_kdd, # Initial flag for the connection
                    'src_ip': ip_src,
                    'dst_ip': ip_dst,
                    'dst_port': dport,
                    'serror_count': 0, 
                    'rerror_count': 0, 
                }
                # For first packet, check if it's an error type
                if flags_kdd == 'S0' or flags_kdd == 'REJ':
                    active_connections[connection_key]['serror_count'] += 1
                if flags_kdd == 'RSTO':
                    active_connections[connection_key]['rerror_count'] += 1
                
            else:
                # Existing connection
                conn = active_connections[connection_key]
                conn['last_packet_time'] = timestamp
                if packet[IP].src == ip_src:
                    conn['src_bytes'] += size
                else: 
                    conn['dst_bytes'] += size
                
                if flags_kdd in FLAG_ENCODING: 
                     conn['flag'] = flags_kdd

                if flags_kdd == 'S0' or flags_kdd == 'REJ':
                    conn['serror_count'] += 1
                if flags_kdd == 'RSTO':
                    conn['rerror_count'] += 1

            # --- Print Packet Details without Location ---
            dns_name = ""
            if packet.haslayer(DNSQR):
                try:
                    dns_name = f"DNS Query: {packet[DNSQR].qname.decode(errors='ignore')}"
                except Exception:
                    dns_name = "DNS Query: (decode error)"

            src_display = get_display_name(ip_src)
            dst_display = get_display_name(ip_dst)

            print(f"""
---- PACKET ----
Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}
Protocol: {proto.upper()}
From: {src_display} ({ip_src}:{sport if sport is not None else 'N/A'})
To:   {dst_display} ({ip_dst}:{dport if dport is not None else 'N/A'})
Size: {size} bytes
Flag: {flags_kdd}
Service: {service}
{dns_name}
----------------
""")

def extract_features_from_closed_connections():
    current_time = time.time()
    connections_to_close = []
    
    with lock: # Ensure thread-safe access to active_connections
        for conn_key, conn_data in active_connections.items():
            if (current_time - conn_data['last_packet_time'] > 5): 
                connections_to_close.append(conn_key)

                # --- OPTIMIZED FEATURE EXTRACTION ---
                # Build the features dictionary only for features in IDS_FEATURE_NAMES
                features = {}
                duration = current_time - conn_data['start_time']

                # Map all possible raw features to their calculation/value
                # Only include features here if they are in IDS_FEATURE_NAMES or needed for other calculations.
                # Initialize with common basic features
                all_possible_features_calculated = {
                    'duration': duration,
                    'protocol_type': PROTOCOL_ENCODING.get(conn_data['protocol_type'], PROTOCOL_ENCODING['other']),
                    'service': SERVICE_ENCODING.get(conn_data['service'], SERVICE_ENCODING['other']),
                    'flag': FLAG_ENCODING.get(conn_data['flag'], FLAG_ENCODING['OTH']),
                    'src_bytes': conn_data['src_bytes'],
                    'dst_bytes': conn_data['dst_bytes'],
                    'land': 1 if conn_data['src_ip'] == conn_data['dst_ip'] else 0,
                    'wrong_fragment': 0, 
                    'urgent': 0, 
                    'hot': 0, 
                    'num_failed_logins': 0,
                    'logged_in': 0, 
                    'num_compromised': 0,
                    'root_shell': 0,
                    'su_attempted': 0,
                    'num_root': 0,
                    'num_file_creations': 0,
                    'num_shells': 0,
                    'num_access_files': 0,
                    'num_outbound_cmds': 0,
                    'is_host_login': 0,
                    'is_guest_login': 0,
                }

                # Add connection summary to history for time/host-based features FIRST
                conn_summary_ts = int(current_time)
                conn_summary = (conn_summary_ts, conn_data['src_ip'], conn_data['dst_ip'],
                                conn_data['dst_port'], conn_data['protocol_type'],
                                conn_data['service'], conn_data['flag'], conn_data['serror_count'], conn_data['rerror_count'])
                connection_summary_history.append(conn_summary)
                host_connection_history[conn_data['dst_ip']].append(conn_summary)
                host_service_connection_history[(conn_data['dst_ip'], conn_data['service'])].append(conn_summary)

                # Now calculate time-based and host-based features and add them to all_possible_features_calculated
                recent_history_2s = [s for s in connection_summary_history if int(current_time) - s[0] <= 2]
                conn_count_2s = sum(1 for s in recent_history_2s if s[2] == conn_data['dst_ip'])
                srv_count_2s = sum(1 for s in recent_history_2s if s[2] == conn_data['dst_ip'] and s[5] == conn_data['service'])

                all_possible_features_calculated['count'] = conn_count_2s
                all_possible_features_calculated['srv_count'] = srv_count_2s

                # S/R Error Rates (within 2-second window)
                if conn_count_2s > 0:
                    serror_count_2s = sum(1 for s in recent_history_2s if s[2] == conn_data['dst_ip'] and (s[6] == 'S0' or s[6] == 'REJ'))
                    rerror_count_2s = sum(1 for s in recent_history_2s if s[2] == conn_data['dst_ip'] and s[6] == 'RSTO')
                    all_possible_features_calculated['serror_rate'] = serror_count_2s / conn_count_2s
                    all_possible_features_calculated['rerror_rate'] = rerror_count_2s / conn_count_2s
                else: # Default if no connections
                    all_possible_features_calculated['serror_rate'] = 0
                    all_possible_features_calculated['rerror_rate'] = 0
                
                if srv_count_2s > 0:
                    srv_serror_count_2s = sum(1 for s in recent_history_2s if s[2] == conn_data['dst_ip'] and s[5] == conn_data['service'] and (s[6] == 'S0' or s[6] == 'REJ'))
                    srv_rerror_count_2s = sum(1 for s in recent_history_2s if s[2] == conn_data['dst_ip'] and s[5] == conn_data['service'] and s[6] == 'RSTO')
                    all_possible_features_calculated['srv_serror_rate'] = srv_serror_count_2s / srv_count_2s
                    all_possible_features_calculated['srv_rerror_rate'] = srv_rerror_count_2s / srv_count_2s
                else: # Default if no service connections
                    all_possible_features_calculated['srv_serror_rate'] = 0
                    all_possible_features_calculated['srv_rerror_rate'] = 0

                all_possible_features_calculated['same_srv_rate'] = srv_count_2s / conn_count_2s if conn_count_2s > 0 else 0

                unique_services_2s = len(set(s[5] for s in recent_history_2s if s[2] == conn_data['dst_ip']))
                all_possible_features_calculated['diff_srv_rate'] = (unique_services_2s - (srv_count_2s if conn_data['service'] != 'other' else 0)) / conn_count_2s if conn_count_2s > 0 else 0 
                
                all_possible_features_calculated['srv_diff_host_rate'] = 0 


                # === Host-based features (last 100 connections to same destination host) ===
                recent_host_connections_100 = list(host_connection_history[conn_data['dst_ip']]) 
                recent_host_srv_connections_100 = list(host_service_connection_history[(conn_data['dst_ip'], conn_data['service'])])

                all_possible_features_calculated['dst_host_count'] = len(recent_host_connections_100)
                all_possible_features_calculated['dst_host_srv_count'] = len(recent_host_srv_connections_100)

                # Host-based rates
                dst_host_count = all_possible_features_calculated['dst_host_count']
                dst_host_srv_count = all_possible_features_calculated['dst_host_srv_count']

                all_possible_features_calculated['dst_host_same_srv_rate'] = dst_host_srv_count / dst_host_count if dst_host_count > 0 else 0
                
                unique_dst_host_services_100 = len(set(s[5] for s in recent_host_connections_100))
                all_possible_features_calculated['dst_host_diff_srv_rate'] = (unique_dst_host_services_100 - dst_host_srv_count) / dst_host_count if dst_host_count > 0 else 0

                all_possible_features_calculated['dst_host_same_src_port_rate'] = 0
                all_possible_features_calculated['dst_host_srv_diff_host_rate'] = 0 

                # Host-based error rates (last 100 connections to this host)
                if dst_host_count > 0:
                    dst_host_serror_count = sum(1 for s in recent_host_connections_100 if (s[6] == 'S0' or s[6] == 'REJ'))
                    dst_host_rerror_count = sum(1 for s in recent_host_connections_100 if s[6] == 'RSTO')
                    all_possible_features_calculated['dst_host_serror_rate'] = dst_host_serror_count / dst_host_count
                    all_possible_features_calculated['dst_host_rerror_rate'] = dst_host_rerror_count / dst_host_count
                else:
                    all_possible_features_calculated['dst_host_serror_rate'] = 0
                    all_possible_features_calculated['dst_host_rerror_rate'] = 0
                
                if dst_host_srv_count > 0:
                    dst_host_srv_serror_count = sum(1 for s in recent_host_srv_connections_100 if (s[6] == 'S0' or s[6] == 'REJ'))
                    dst_host_srv_rerror_count = sum(1 for s in recent_host_srv_connections_100 if s[6] == 'RSTO')
                    all_possible_features_calculated['dst_host_srv_serror_rate'] = dst_host_srv_serror_count / dst_host_srv_count
                    all_possible_features_calculated['dst_host_srv_rerror_rate'] = dst_host_srv_rerror_count / dst_host_srv_count
                else:
                    all_possible_features_calculated['dst_host_srv_serror_rate'] = 0
                    all_possible_features_calculated['dst_host_srv_rerror_rate'] = 0


                # Final features dictionary containing ONLY the features specified in IDS_FEATURE_NAMES
                # This ensures the dictionary has the correct size and order for the scaler/model.
                final_extracted_features = {
                    name: all_possible_features_calculated.get(name, 0) for name in IDS_FEATURE_NAMES
                }

                extracted_features_queue.append((conn_key, final_extracted_features))

        # Remove closed connections from active_connections
        for conn_key in connections_to_close:
            del active_connections[conn_key]

        # Clean up old data from history deques (connections older than 100 seconds)
        cutoff_time_history = current_time - 100
        while connection_summary_history and connection_summary_history[0][0] < cutoff_time_history:
            connection_summary_history.popleft()
        
        for ip, history_deque in list(host_connection_history.items()): 
            while history_deque and history_deque[0][0] < cutoff_time_history:
                history_deque.popleft()
            if not history_deque: 
                del host_connection_history[ip]

        for key, history_deque in list(host_service_connection_history.items()):
            while history_deque and history_deque[0][0] < cutoff_time_history:
                history_deque.popleft()
            if not history_deque:
                del host_service_connection_history[key]


def predict_and_alert():
    while extracted_features_queue:
        conn_key, features_dict = extracted_features_queue.popleft()
        
        if final_model is None or scaler is None:
            print(f"[IDS] Model or scaler not loaded. Skipping prediction for {conn_key}")
            continue

        # Prepare features for the model
        # Create a Pandas DataFrame from the extracted features with the correct column names
        # This resolves the UserWarning about missing feature names.
        features_df = pd.DataFrame([features_dict], columns=IDS_FEATURE_NAMES)

        try:
            # Scale the features using the loaded scaler
            scaled_features_array = scaler.transform(features_df) # Pass DataFrame directly
            
            # Predict
            prediction = final_model.predict(scaled_features_array)[0]
            
            # Assuming your model outputs 0 for Anomaly and 1 for Normal
            prediction_label = "NORMAL" if prediction == 1 else "ANOMALY"

            # --- MODIFIED: Store message for display in plot ---
            prediction_message = (
                f"[{time.strftime('%H:%M:%S')}] "
                f"Conn {conn_key[0]}:{conn_key[1]} -> {conn_key[2]}:{conn_key[3]} ({conn_key[4].upper()}): "
                f"Predicted: {prediction_label}"
            )
            print(f"[IDS Prediction] {prediction_message}") # Still print to console
            recent_predictions_display.append(prediction_message) # Add to display queue

            if prediction_label == "ANOMALY":
                print(f"\n!!! ANOMALY ALERT !!!")
                print(f"  Connection: {conn_key}")
                print(f"  Predicted Class: {prediction_label}")
                print(f"--------------------------------------------------")

        except Exception as e:
            error_message = f"ERROR during IDS prediction for {conn_key}: {e}"
            print(error_message)
            print(f"  Problematic features_df shape: {features_df.shape}")
            print(f"  Expected features by scaler (from IDS_FEATURE_NAMES): {len(IDS_FEATURE_NAMES)}")
            # Add error message to display if desired, or handle differently
            recent_predictions_display.append(f"[{time.strftime('%H:%M:%S')}] ERROR: {str(e)[:50]}...")


# This thread will periodically extract features and run predictions
def ids_processing_thread_func():
    while not stop_sniffing_flag.is_set():
        extract_features_from_closed_connections()
        predict_and_alert() 
        time.sleep(1)


def sniff_packets():
    print(f"Starting packet sniffing on interface: {interface}...")
    if duration_seconds > 0:
        print(f"Monitoring for {duration_seconds} seconds.")
        sniff(iface=interface, prn=process_packet, store=False, timeout=duration_seconds, stop_filter=lambda x: stop_sniffing_flag.is_set())
    else:
        print("Monitoring until plot window is closed.")
        sniff(iface=interface, prn=process_packet, store=False, stop_filter=lambda x: stop_sniffing_flag.is_set())
    print("Packet sniffing stopped.")


# === PLOTTING ===
# MODIFIED: Changed to 4 subplots and adjusted figsize
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14)) # Increased height for new subplot

def plot_per_user_curve(ax):
    ax.set_title(f"Top {top_n} IP Traffic (Last 60 Seconds)")
    ax.set_xlabel("Seconds Ago")
    ax.set_ylabel("Bytes")
    ax.grid(True)

    current_time = int(time.time())

    ip_current_traffic = defaultdict(int)
    for ip, data_points in traffic_per_second.items():
        while data_points and data_points[0][0] < current_time - 59:
            data_points.popleft()
        for ts, size in data_points:
            ip_current_traffic[ip] += size

    sorted_ips = sorted(ip_current_traffic.items(), key=lambda item: item[1], reverse=True)[:top_n]

    for ip, _ in sorted_ips:
        data_points = traffic_per_second[ip]
        aggregated_traffic_for_ip = defaultdict(int)
        for ts, size in data_points:
            aggregated_traffic_for_ip[ts] += size

        times_relative = []
        sizes_agg = []
        
        for t_rel in range(-59, 1):
            target_ts = current_time + t_rel
            times_relative.append(t_rel)
            sizes_agg.append(aggregated_traffic_for_ip.get(target_ts, 0))

        ax.plot(times_relative, sizes_agg, label=get_display_name(ip)) # Use get_display_name
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))

def plot_protocol_distribution(ax):
    ax.set_title("Protocol Usage Distribution")
    ax.axis('equal')

    labels = list(protocol_usage.keys())
    sizes = list(protocol_usage.values())
    
    total_size = sum(sizes)
    if total_size > 0:
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, pctdistance=0.85)
    else:
        ax.text(0.5, 0.5, "No traffic yet", ha='center', va='center', transform=ax.transAxes)

def plot_bandwidth(ax):
    ax.set_title("Total Bandwidth Usage Over Time (Bytes/sec)")
    ax.set_xlabel("Seconds Ago")
    ax.set_ylabel("Bytes/sec")
    ax.grid(True)

    if bandwidth_history:
        x_values = list(range(-len(bandwidth_history) + 1, 1))
        ax.plot(x_values, list(bandwidth_history), label="Total Bandwidth (Bytes/sec)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No bandwidth data yet", ha='center', va='center', transform=ax.transAxes)

# --- ADDED: New function to plot prediction messages ---
def plot_predictions(ax):
    ax.set_title("Recent IDS Predictions")
    ax.axis('off') # Turn off axes for a cleaner text display
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Display messages from bottom to top
    for i, msg in enumerate(reversed(recent_predictions_display)):
        ax.text(0.02, 0.05 + i * 0.1, msg, 
                verticalalignment='bottom', horizontalalignment='left', 
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5, alpha=0.8))


def animate(i):
    with lock:
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear() # Clear the new predictions subplot

        plot_per_user_curve(ax1)
        plot_protocol_distribution(ax2)

        current_time = int(time.time())
        bandwidth_for_last_second = total_traffic_per_second.get(current_time - 1, 0)
        bandwidth_history.append(bandwidth_for_last_second)

        keys_to_delete_from_total_traffic = [ts for ts in total_traffic_per_second if ts < current_time - 60]
        for ts in keys_to_delete_from_total_traffic:
            del total_traffic_per_second[ts]

        plot_bandwidth(ax3)
        plot_predictions(ax4) # Call the new function for predictions
    
    plt.tight_layout()

def start_plot():
    ani = FuncAnimation(fig, animate, interval=1000, cache_frame_data=False)
    plt.tight_layout()
    try:
        plt.show()
    except Exception as e:
        print(f"Error displaying plot: {e}")
    finally:
        print("Plot window closed. Stopping sniffing and IDS processing...")
        stop_sniffing_flag.set()
        fig.savefig("traffic_snapshot.png")
        print("📸 Final image saved as 'traffic_snapshot.png'")

# === MAIN EXECUTION ===
sniff_thread = threading.Thread(target=sniff_packets)
ids_thread = threading.Thread(target=ids_processing_thread_func) 

sniff_thread.start()
ids_thread.start() 

start_plot() # This blocks until plot window is closed

# Wait for both threads to finish cleanly
sniff_thread.join()
ids_thread.join()

# === SAVE RESULTS ===
print("\nGenerating final traffic summary...")
with lock:
    sorted_data = sorted(traffic_data.items(), key=lambda x: x[1]['total'], reverse=True)

with open("Result.txt", "w") as f:
    f.write("=== Traffic Summary ===\n")
    for ip, stats in sorted_data[:top_n]:
        display_name = get_display_name(ip) 
        total_kb = stats['total'] / 1024
        sent_kb = stats['sent'] / 1024
        recv_kb = stats['received'] / 1024
        result_line = f"{display_name} ({ip}) - Total: {total_kb:.2f} KB | Sent: {sent_kb:.2f} KB | Received: {recv_kb:.2f} KB"
        print(result_line)
        f.write(result_line + "\n")
        if total_kb > threshold_kb:
            alert = f"⚠ ALERT: {display_name} ({ip}) exceeded threshold with {total_kb:.2f} KB"
            print(alert)
            f.write(alert + "\n")

print("\nMonitoring complete. Results saved in Result.txt.")
print("Real-time anomaly detection alerts will appear in the console.")
