import numpy as np
import os
import glob
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, top_k_accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm
import warnings
from scapy.all import rdpcap, TCP, IP
import re
from multiprocessing import Pool, Manager
from functools import partial
import time
import argparse
from scipy.stats import skew, kurtosis
from statsmodels.stats.proportion import proportion_confint

warnings.filterwarnings('ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

"""
Variables that may need to be changed
"""
# SET THIS TO TRAIN + TEST IPs
CLIENT_IP = ["159.203.90.140", "45.55.80.202"]

# The "tau" from our thesis
ROUND_GAP_THRESHOLD = 0.025

##########################
# Packet Processing Code #
##########################

def estimate_input_prompt_size(packet_wire_length):
    """Estimates the prompt payload size by subtracting overhead."""
    # Return 0 if packet is smaller than overhead, avoid negative sizes
    return max(0, packet_wire_length - 300)


def extract_packets(pcap_file):
    """
    Extracts server response packets and estimates client input prompt size
    from a pcap file, combining heuristics from both previous functions.

    Args:
        pcap_file (str): Path to the pcap file to analyze

    Returns:
        tuple: 'response_packets' (list of tuples) and 'input_prompt_size' (int). 
        Returns (None, None) on error.
    """
    try:
        packets = rdpcap(pcap_file)
    except Exception as e:
        return (None, None)

    if not packets:
        return (None, None)

    client_packets_info = []
    server_packets_info_all = []
    estimated_input_size = 0

    # --- Pass 1: Classify packets and find largest client packet ---
    for i, pkt in enumerate(packets):
        if not (IP in pkt and TCP in pkt and hasattr(pkt, 'wirelen') and hasattr(pkt, 'time')):
            continue

        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        wirelen = pkt.wirelen
        pkt_time = float(pkt.time)

        # Identify packets FROM the client
        if src_ip in CLIENT_IP:
            client_packets_info.append({'index': i, 'wirelen': wirelen, 'time': pkt_time})

        # Identify packets TO the client
        elif dst_ip in CLIENT_IP:
            server_packets_info_all.append({'index': i, 'wirelen': wirelen, 'time': pkt_time})

    # --- Estimate Input Prompt Size ---
    if client_packets_info:
        # Largest client packet overall is the prompt
        largest_client_pkt = max(client_packets_info, key=lambda p: p['wirelen'], default=None)
        if largest_client_pkt:
            estimated_input_size = estimate_input_prompt_size(largest_client_pkt['wirelen'])


    # --- Process Server Response Packets ---
    response_packets_result = []
    if not server_packets_info_all:
        return [], estimated_input_size

    # Find Start Index
    last_ack_index = -1
    for i, pkt in enumerate(packets[:15]):
        if TCP in pkt and str(pkt[TCP].flags) == 'A' and not pkt[TCP].payload: 
            if IP in pkt and (pkt[IP].src in CLIENT_IP):
                 last_ack_index = i
    start_index = last_ack_index + 1

    # Skip Processing packets at the sart
    processing_threshold = 150
    while (start_index < len(packets)):
        pkt_at_start = packets[start_index]
        # Check if packet exists, has wirelen, and is below threshold
        if hasattr(pkt_at_start, 'wirelen') and pkt_at_start.wirelen < processing_threshold:
            if IP in pkt_at_start and (pkt_at_start[IP].dst == CLIENT_IP):
                start_index += 1
            else:
                break
        else:
            break

    if start_index >= len(packets):
        return [], estimated_input_size

    relevant_server_packets_info = [p for p in server_packets_info_all if p['index'] >= start_index]

    if len(relevant_server_packets_info) <= 2:
        return [], estimated_input_size

    # Remove Trailing Packets
    cumulative_size = 0
    packets_to_keep_count = len(relevant_server_packets_info)
    end_cutoff_bytes = 1100

    for i in range(len(relevant_server_packets_info) - 1, -1, -1):
        cumulative_size += relevant_server_packets_info[i]['wirelen']
        if cumulative_size >= end_cutoff_bytes:
            packets_to_keep_count = i 
            break

    if packets_to_keep_count <= 1:
        return [], estimated_input_size

    final_server_packets_info = relevant_server_packets_info[:packets_to_keep_count]


    start_time_ref = float(packets[0].time) # Use the time of the *first packet in the pcap*
    seen_set = set()

    # Skip retransmissions
    for pkt_info in final_server_packets_info:
        if pkt_info['wirelen'] > 60:
            original_pkt_index = pkt_info['index']
            if 0 <= original_pkt_index < len(packets):
                pkt = packets[original_pkt_index]
                if IP in pkt and TCP in pkt:
                    ip = pkt[IP]
                    tcp = pkt[TCP]
                    flow_id = (ip.src, tcp.sport, ip.dst, tcp.dport)
                    seq_num = int(tcp.seq)
                    key = (flow_id, seq_num)

                    if key in seen_set:
                        continue 
                    seen_set.add(key)

                    relative_time = pkt_info['time'] - start_time_ref
                    response_packets_result.append((pkt_info['wirelen'], relative_time))
                else:
                    pass
            else:
                pass


    if len(response_packets_result) < 2:
        # print(f"Not enough valid server data packets after final filtering in {pcap_file}")
        return [], estimated_input_size

    res = [ p for p in response_packets_result if p[0] >= 150 ]
    return res, estimated_input_size


# Modified feature extraction for MLP/RF
def extract_features(pcap_file, subject_label=None):
    packets, est_query_size = extract_packets(pcap_file)
    if not packets: return None

    features = analyze_rounds_mlp_rf(packets, subject_label, est_query_size) 
    if features:
        return features
    else:
        return None
    
def fft_features(ipds, n_coeffs=10):
    """
    Given a 1D array of inter‑packet delays, return the magnitudes
    of the first n_coeffs positive-frequency FFT bins (ignoring DC),
    padding with zeros if there aren't enough bins.
    """
    fft_vals = np.fft.rfft(ipds)
    mags     = np.abs(fft_vals)[1:n_coeffs+1]
    if len(mags) < n_coeffs:
        # pad the remainder with zeros for fixed len
        mags = np.pad(mags, (0, n_coeffs - len(mags)), 'constant')
    return mags.tolist()


# --- Random safe funcs ---
def safe_mean(arr):
    return np.mean(arr) if len(arr) > 0 else 0.0

def safe_median(arr):
    return np.median(arr) if len(arr) > 0 else 0.0

def safe_std(arr):
    return np.std(arr, ddof=1) if len(arr) > 1 else 0.0 

def safe_min(arr):
    return np.min(arr) if len(arr) > 0 else 0.0

def safe_max(arr):
    return np.max(arr) if len(arr) > 0 else 0.0

def safe_percentile(arr, p):
     return np.percentile(arr, p) if len(arr) > 0 else 0.0


# --- Token Estimation ---
def estimate_token_count(length):
    """
    Estimates the number of tokens in a packet based on observed sizes
    """
    # Parameters 
    overhead = 100
    total_bytes_per_token = 300

    # Filter out ACKs
    if length < 150: 
        return 0

    # Estimate tokens based on the average total size per token
    estimated_tokens = round((length - overhead) / total_bytes_per_token)

    # Ensure at least 1 token if it's a data packet
    return max(1, int(estimated_tokens))

def _five_number(num_arr):
    """Return (mean, median, std, min, max, p25, p75) for a 1‑D array."""
    return [
        safe_mean(num_arr), safe_median(num_arr), safe_std(num_arr),
        safe_min(num_arr), safe_max(num_arr),
        safe_percentile(num_arr, 25), safe_percentile(num_arr, 75)
    ]

MAX_BITS = 128
BIT_VEC_SLICE = slice(0, MAX_BITS)
STATS_SLICE   = slice(MAX_BITS, None)

def pad_bits(bits, length=MAX_BITS):
    """Pad or truncate the bit list to exactly `length` entries."""
    if len(bits) >= length:
        return bits[:length]
    return bits + [0] * (length - len(bits))


# --- Main Round Analysis Function ---
def analyze_rounds_mlp_rf(packets, subject_label=None, query_size=0):
    """
    Analyzes packet timings from services like TogetherAI using a round-based approach.
    Groups packets into rounds based on timing gaps, calculates inter-round timings,
    classifies rounds, and extracts features suitable for MLP/RF models.

    Args:
        packets (list): List of (packet_size, timestamp) tuples, assumed to be
                        pre-filtered application data packets from the server.
        subject_label (str, optional): The label/condition for this sample.
        query_size (int): Size of the input query in bytes.

    Returns:
        list: A list of numerical features summarizing the interaction,
              or None if analysis is not possible (e.g., too few packets/rounds).
    """
    if not packets or len(packets) < 2:
        return None # Need at least two packets to calculate any timing

    sizes = np.array([p[0] for p in packets])
    timestamps = np.array([p[1] for p in packets])

    # --- 1. Group Packets into Rounds ---
    rounds_packet_data = []

    current_round_packets = [packets[0]]
    ipds = np.diff(timestamps)

    for i in range(len(ipds)):
        # Check if the gap indicates a new round
        if ipds[i] > ROUND_GAP_THRESHOLD:
            # Finalize the previous round
            if current_round_packets:
                rounds_packet_data.append(current_round_packets)
            # Start the new round
            current_round_packets = [packets[i+1]]
        else:
            # Add packet to the current round (gap is small)
            current_round_packets.append(packets[i+1])

    # Add the very last round
    if current_round_packets:
        rounds_packet_data.append(current_round_packets)

    num_rounds = len(rounds_packet_data)
    if num_rounds < 1:
        return None

    # --- 2. Calculate Properties for Each Round ---
    round_properties = []
    total_packets_overall = 0
    total_tokens_overall = 0
    for r_packets in rounds_packet_data:
        if not r_packets: continue 

        round_total_size = sum(p[0] for p in r_packets)
        round_total_tokens = sum(estimate_token_count(p[0]) for p in r_packets)
        
        round_properties.append({
            'num_packets': len(r_packets),
            'total_size': round_total_size,
            'total_tokens': round_total_tokens,
            'start_time': r_packets[0][1],
            'end_time': r_packets[-1][1],
            'duration': r_packets[-1][1] - r_packets[0][1] # Within-round duration
        })
        total_packets_overall += len(r_packets)
        total_tokens_overall += round_total_tokens

    # --- 3. Calculate Inter-Round Timings and Classify Rounds ---
    round_start_times = [r['start_time'] for r in round_properties]
    inter_round_ipds = np.array([])
    round_flags = []  # 1 = speculative, 0 = original

    if num_rounds > 1:
        inter_round_ipds = np.diff(round_start_times)

    # For each round, assign flags for each token: first k-1 speculative, last original
    for r_packets in rounds_packet_data:
        token_counts = [estimate_token_count(p[0]) for p in r_packets]
        total_tokens = sum(token_counts)
        if total_tokens == 0:
            continue
        # For all but the last token in the round, speculative (1)
        if total_tokens > 1:
            round_flags.extend([1] * (total_tokens - 1))
        # Last token in the round is original (0)
        round_flags.append(0)

    # --- 4. Extract Features for MLP/RF ---
    
    # Overall Stats
    overall_start_time = round_properties[0]['start_time']
    overall_end_time = round_properties[-1]['end_time']
    total_duration_overall = overall_end_time - overall_start_time if num_rounds > 0 else 0.0
    avg_tokens_per_sec = total_tokens_overall / total_duration_overall if total_duration_overall > 0 else 0.0
    avg_rounds_per_sec = num_rounds / total_duration_overall if total_duration_overall > 0 else 0.0
    avg_packets_per_sec = total_packets_overall / total_duration_overall if total_duration_overall > 0 else 0.0

    # Round Property Stats
    tokens_per_round = [r['total_tokens'] for r in round_properties]
    size_per_round = [r['total_size'] for r in round_properties]
    packets_per_round = [r['num_packets'] for r in round_properties]
    duration_per_round = [r['duration'] for r in round_properties] # Within-round duration

    # Inter-Round IPD Stats (only if num_rounds > 1)
    ir_ipd_mean = safe_mean(inter_round_ipds)
    ir_ipd_median = safe_median(inter_round_ipds)
    ir_ipd_std = safe_std(inter_round_ipds)
    ir_ipd_min = safe_min(inter_round_ipds)
    ir_ipd_max = safe_max(inter_round_ipds)
    ir_ipd_p25 = safe_percentile(inter_round_ipds, 25)
    ir_ipd_p75 = safe_percentile(inter_round_ipds, 75)

    # Round Classification Stats
    speculative_round_ratio = sum(f == 1 for f in round_flags) / len(round_flags) if len(round_flags) > 0 else 0.0
    original_round_ratio = sum(f == 0 for f in round_flags) / len(round_flags) if len(round_flags) > 0 else 0.0

    # Count transitions between round types (spec -> orig or orig -> spec)
    transitions = 0
    if len(round_flags) > 1:
        for j in range(len(round_flags) - 1):
            if round_flags[j] != round_flags[j+1]:
                transitions += 1
    transition_rate = transitions / len(round_flags) if len(round_flags) > 0 else 0.0


    if len(timestamps) > 1:
        ipds_all  = np.diff(timestamps)
    else:
        ipds_all  = np.array([0.0])

    tokens_per_pkt = np.array([estimate_token_count(s) for s in sizes])

    # Per‑packet size stats  (7 scalars)
    pkt_size_stats = _five_number(sizes)

    # Global IPD stats       (7 scalars)
    ipd_stats      = _five_number(ipds_all)

    # Token‑per‑packet stats (7 scalars)
    tok_pp_stats   = _five_number(tokens_per_pkt)

    # Some more
    spectral_feats = fft_features(ipds_all, n_coeffs=10)


    # Assemble the feature vector
    old_features = [
        # Overall
        num_rounds,
        total_packets_overall,
        total_tokens_overall,
        total_duration_overall,
        avg_tokens_per_sec,
        avg_rounds_per_sec,
        avg_packets_per_sec,
        # Tokens per round stats
        safe_mean(tokens_per_round),
        safe_median(tokens_per_round),
        safe_std(tokens_per_round),
        safe_min(tokens_per_round),
        safe_max(tokens_per_round),
        # Size per round stats
        safe_mean(size_per_round),
        safe_median(size_per_round),
        safe_std(size_per_round),
        safe_min(size_per_round),
        safe_max(size_per_round),
        # Packets per round stats
        safe_mean(packets_per_round),
        safe_median(packets_per_round),
        safe_std(packets_per_round),
        # Duration per round stats (within-round)
        safe_mean(duration_per_round),
        safe_median(duration_per_round),
        safe_std(duration_per_round),
        # Inter-round IPD stats
        ir_ipd_mean,
        ir_ipd_median,
        ir_ipd_std,
        ir_ipd_min,
        ir_ipd_max,
        ir_ipd_p25,
        ir_ipd_p75,
        # Round classification stats
        speculative_round_ratio,
        original_round_ratio,
        transitions,
        transition_rate,
        # Input query size
        query_size
    ]

    features = (
        old_features +
        pkt_size_stats +        # 7
        ipd_stats      +        # 7
        tok_pp_stats            # 7
    )

    features = features + spectral_feats
    features += [skew(ipds_all), kurtosis(ipds_all)]

    features += [
        safe_mean(tokens_per_round) / (safe_mean(duration_per_round) + 1e-6),
        safe_max(size_per_round) * speculative_round_ratio,
        transitions / (len(round_flags) + 1),
        total_packets_overall / query_size if query_size > 0 else 1,
        total_tokens_overall / query_size if query_size > 0 else 1,
    ]

    """
    Not utilised currently. Did not seem to do anything.

    # Rolling stats
    window_size = 3
    rolling_avgs_ipds = [np.mean(ipds[i:i+window_size]) for i in range(len(ipds)-window_size+1)]
    rolling_avgs_size = [np.mean(sizes[i:i+window_size]) for i in range(len(sizes)-window_size+1)]

    features += _five_number(rolling_avgs_ipds)
    features += _five_number(rolling_avgs_size)
    """

    raw_bits = pad_bits(round_flags)

    """
    Not utilised currently. Did not seem to do anything.

    size_per_round = [sum(p[0] for p in r) for r in rounds_packet_data]
    ir_size_deltas = np.diff(size_per_round) if len(size_per_round) > 1 else np.array([0.0])
    ir_size_stats = _five_number(ir_size_deltas)
    ipds_all = np.diff(timestamps) if len(timestamps) > 1 else np.array([0.0])
    accel = np.diff(ipds_all) if len(ipds_all) > 1 else np.array([0.0])
    accel_stats = _five_number(accel)
    corr = (np.corrcoef(sizes[:len(ipds_all)], ipds_all)[0,1]
            if len(ipds_all) > 1 else 0.0)
    features = np.concatenate([features, ir_size_stats, accel_stats, [corr]])
    """

    features = np.concatenate([raw_bits, features])
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features.tolist()


#############################
# Data Processing Functions #
#############################

def process_single_subject(subject, data_dir, target_prompts, subject_to_label, progress_dict, model_type):
    """ Process a single subject's pcap files. """
    subject_features = []
    subject_labels = []
    subject_file_paths = []
    subject_prompt_ids = []
    
    subject_path = os.path.join(data_dir, subject)

    # Update more frequently based on files processed, not just prompts
    total_files = len(glob.glob(os.path.join(subject_path, "query_*.pcap")))
    files_processed = 0

    for i, prompt_id in enumerate(target_prompts):
        prompt_pattern = f"query_{prompt_id}_*.pcap"
        matching_files = glob.glob(os.path.join(subject_path, prompt_pattern))
        
        for pcap_file in matching_files:
            files_processed += 1
            extracted_data = extract_features(pcap_file, subject)
            
            if extracted_data is not None and (isinstance(extracted_data, list) or extracted_data.size > 0):
                subject_features.append(extracted_data)
                subject_labels.append(subject_to_label[subject])
                subject_file_paths.append(pcap_file)
                subject_prompt_ids.append(prompt_id)
        
            # Update progress more frequently
            progress_dict[subject] = min(100, files_processed / total_files * 100)

    # Mark completion
    progress_dict[subject] = 100
    
    return subject_features, subject_labels, subject_file_paths, subject_prompt_ids

def process_dataset(data_dir, model_type="mlp"):
    """ Process all pcap files in the dataset, using parallel processing. """
    subject_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if not subject_dirs:
        raise FileNotFoundError(f"No subject directories (e.g., 'lang_english') found in {data_dir}")

    subject_to_label = {subject: i for i, subject in enumerate(sorted(subject_dirs))}
    label_to_subject = {i: subject for subject, i in subject_to_label.items()}
    
    all_prompts = set()
    # Limit prompt scanning for efficiency if dataset is huge
    print("Scanning for unique prompts...")
    for subject in tqdm(subject_dirs, desc="Scanning Subjects"):
        subject_path = os.path.join(data_dir, subject)
        pcap_files = glob.glob(os.path.join(subject_path, "query_*.pcap"))
        for i, pcap_file in enumerate(pcap_files):
            # if i >= scan_limit_per_subject: break # Limit scanning
            match = re.search(r"query_(\d+)_", os.path.basename(pcap_file))
            if match:
                all_prompts.add(int(match.group(1)))
    
    target_prompts = sorted(list(all_prompts))
    if not target_prompts:
         raise ValueError("No prompts found matching the 'query_ID_' pattern.")
    print(f"Found {len(target_prompts)} unique prompts")
    
    with Manager() as manager:
        progress_dict = manager.dict({subject: 0 for subject in subject_dirs})
        
        process_func = partial(
            process_single_subject,
            data_dir=data_dir,
            target_prompts=target_prompts,
            subject_to_label=subject_to_label,
            progress_dict=progress_dict,
            model_type=model_type
        )
        
        num_processes = min(os.cpu_count(), len(subject_dirs)) # Use available CPUs, but not more than subjects
        print(f"Starting processing pool with {num_processes} workers...")
        with Pool(processes=num_processes) as pool:
            # Add a timeout to imap_unordered
            results_iterator = pool.imap_unordered(process_func, subject_dirs)
            
            subject_pbars = {
                subject: tqdm(
                    total=100, desc=f"{subject:<20}", position=i, leave=False, 
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f}'
                ) for i, subject in enumerate(sorted(subject_dirs))
            }

            # Process results as they complete and update progress
            all_results = []
            processed_subjects = 0
            while processed_subjects < len(subject_dirs):
                # Update progress bars frequently
                for subject, pbar in subject_pbars.items():
                     current = progress_dict.get(subject, 0)
                     # Ensure progress doesn't exceed 100 due to timing
                     pbar.n = min(current, 100.0) 
                     pbar.refresh()

                # Check for completed results without blocking indefinitely
                try:
                    # Use timeout version to prevent infinite blocking
                    result = next(results_iterator, None)
                    if result is not None:
                        all_results.append(result)
                        processed_subjects += 1
                    else:
                        # Check if any workers are still alive/working
                        if pool._pool and any(p.is_alive() for p in pool._pool):
                            time.sleep(0.5)  # Workers still going, wait a bit
                        else:
                            print("All workers finished but missing some results")
                            break
                except StopIteration:
                    break
                except Exception as e:
                    print(f"Error getting result: {e}")
                    time.sleep(0.5)
                    continue
                
                time.sleep(0.1)  # Small delay to prevent busy-waiting

            # Final update and close progress bars
            for subject, pbar in subject_pbars.items():
                pbar.n = 100 # Ensure all bars reach 100%
                pbar.refresh()
                pbar.close()
            print("\n" * (len(subject_dirs) + 1) + "Processing complete.") # Move cursor down
    
    all_features, all_labels, all_file_paths, all_prompt_ids = [], [], [], []
    valid_samples = 0
    for feats, labels, file_paths, prompt_ids in all_results:
        all_features.extend(feats)
        all_labels.extend(labels)
        all_file_paths.extend(file_paths)
        all_prompt_ids.extend(prompt_ids)
        valid_samples += len(feats)
    
    print(f"Total valid samples extracted: {valid_samples}")
    if valid_samples == 0:
        raise ValueError("No valid features could be extracted from any pcap file.")

    # For non-sequence models, convert to numpy array here
    if model_type not in ["cnn", "lstm"]:
        all_features = np.array(all_features)
        # Check for NaN/inf values introduced during feature calculation
        if np.isnan(all_features).any() or np.isinf(all_features).any():
            print("Warning: NaN or Inf found in features. Replacing with 0.")
            all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)

    all_labels = np.array(all_labels)
    
    return all_features, all_labels, all_file_paths, all_prompt_ids, label_to_subject



#####################
# Model Training    #
#####################

def train_model(X_train, y_train, X_test, y_test, scaler=None, model_type="mlp", label_to_subject=None):
    """ Trains an arbitrary model """
    num_classes = len(np.unique(y_train))
    class_weights_dict = None
    
    # Calculate class weights for handling imbalance
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)
    if len(unique_classes) > 1: # Avoid issues if only one class in split
        class_weights = total_samples / (num_classes * class_counts)
        class_weights_dict = dict(zip(unique_classes, class_weights))
        print("Calculated class weights:", class_weights_dict)
    
    X_train_bits  = X_train[:, BIT_VEC_SLICE].reshape(-1, MAX_BITS, 1)
    X_test_bits   = X_test[:,  BIT_VEC_SLICE].reshape(-1, MAX_BITS, 1)
    X_train_stats = X_train[:, STATS_SLICE]
    X_test_stats  = X_test[:,  STATS_SLICE]

    # {'mlp__alpha': 0.01, 'mlp__hidden_layer_sizes': (64,), 'mlp__learning_rate_init': 0.0005}
    if model_type == "mlp":
        # Test accuracy: ~0.42
        model = MLPClassifier(
            hidden_layer_sizes=(512,256,128,64),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            alpha=0.0001,
            batch_size=128,
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=42,
            verbose=True
        )
        model.fit(X_train_stats, y_train)
        
    elif model_type == "lgbm":
        # test accuracy: 0.5007
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=10,
            num_leaves=64,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        model.fit(X_train_stats, y_train)
        
    elif model_type == "xgb":
        # test Accuracy: 0.5150
        from xgboost import XGBClassifier
        model = XGBClassifier(
            objective='multi:softprob',
            num_class=num_classes,
            eval_metric='mlogloss',
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            verbosity=1,
            random_state=42
        )
        model.fit(X_train_stats, y_train)
        
    elif model_type == "svc":
        # test accuracy: 0.3960
        from sklearn.svm import SVC
        model = SVC(
            kernel='rbf',       # 'linear' can be faster for baseline
            C=1.0,
            gamma='scale',      # Let sklearn choose based on feature variance
            decision_function_shape='ovr',
            probability=True,
            class_weight='balanced',
            verbose=True
        )
        model.fit(X_train_stats, y_train)

    elif model_type == "stacking":
        # Test accuracy: 0.5203
        from sklearn.ensemble import StackingClassifier
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from sklearn.linear_model import LogisticRegression
        estimators = [
            ('xgb', XGBClassifier(
                objective='multi:softprob',
                num_class=num_classes,
                eval_metric='mlogloss',
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                verbosity=0,
                random_state=42
            )),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(512,256,128,64),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                alpha=0.0001,
                batch_size=128,
                max_iter=300,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=42,
                verbose=False
            ))
        ]
        model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=5,
            n_jobs=-1
        )
        model.fit(X_train_stats, y_train)
        
    elif model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=500, # Fewer trees initially
            max_depth=None,
            min_samples_split=5, # Higher min split
            min_samples_leaf=3, # Higher min leaf
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced_subsample', # Use built-in balancing
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        model.fit(X_train_stats, y_train)
    
    elif model_type == "cnn":
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Concatenate
        num_classes  = len(np.unique(y_train))

        def build_cnn_with_stats(input_shape_bits=(128,1), input_shape_stats=(200,), num_classes=10):
            # Input 1: round_flags
            bit_input = Input(shape=input_shape_bits, name="bit_input")
            x = Conv1D(32, 3, activation='relu')(bit_input)
            x = MaxPooling1D(2)(x)
            x = Conv1D(64, 3, activation='relu')(x)
            x = MaxPooling1D(2)(x)
            x = Flatten()(x)

            # Input 2: numerical stats
            stats_input = Input(shape=input_shape_stats, name="stats_input")

            # Combine them
            combined = Concatenate()([x, stats_input])
            z = Dense(128, activation='relu')(combined)
            z = Dropout(0.5)(z)
            z = Dense(64, activation='relu')(z)
            output = Dense(num_classes, activation='softmax')(z)

            model = Model(inputs=[bit_input, stats_input], outputs=output)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return model

        model = build_cnn_with_stats(
            input_shape_bits=(MAX_BITS, 1),
            input_shape_stats=X_train_stats.shape[1:],
            num_classes=num_classes
        )
        model.fit(
            [X_train_bits, X_train_stats],
            y_train,
            epochs=20,
            batch_size=64,
            verbose=1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # --- Evaluation ---
    y_pred_train, y_pred_test = None, None
    train_accuracy, test_accuracy = 0.0, 0.0

    if model_type == "cnn":
        y_pred_train = model.predict([X_train_bits, X_train_stats]).argmax(axis=1)
        y_pred_test  = model.predict([X_test_bits,  X_test_stats]).argmax(axis=1)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy  = accuracy_score(y_test,  y_pred_test)
    else:
        train_accuracy = model.score(X_train_stats, y_train)
        test_accuracy  = model.score(X_test_stats, y_test)
        y_pred_train = model.predict(X_train_stats)
        y_pred_test = model.predict(X_test_stats)

    print(f"\n--- {model_type.upper()} Model Evaluation ---")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    print("\nTest Set Classification Report:")
    # Ensure labels are mapped correctly if label_to_subject is provided
    target_names = None
    if label_to_subject:
        # Sort labels by numerical value to match classification_report output
        sorted_labels = sorted(label_to_subject.keys())
        target_names = [label_to_subject[l] for l in sorted_labels]
        
    report = classification_report(y_test, y_pred_test, target_names=target_names)
    print(report)

    model_data = {
        'model': model,
        'scaler': scaler, # Scaler might be None for sequence models if scaling embedded
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'classification_report': report,
        'y_pred_test': y_pred_test, # Store predictions for analysis
        'y_test': y_test,
        'X_test_stats': X_test_stats,
    }
        
    return model_data


###########################
# Results Analysis Module #
###########################

def analyze_results(model_data, label_to_subject, model_dir, model_type):
    """ Analyze model performance and visualize results. """
    print("\n--- Analyzing Results ---")
    print(f"Training accuracy: {model_data['train_accuracy']:.4f}")
    print(f"Test accuracy: {model_data['test_accuracy']:.4f}")
    print("\nTest Classification Report (reprinted):")
    print(model_data['classification_report'])
    
    y_test = model_data['y_test']
    y_pred = model_data['y_pred_test']
    model = model_data['model']
    X_test_stats = model_data['X_test_stats']
    proba_test = model.predict_proba(X_test_stats)
    n_test = len(y_test)

    top1 = accuracy_score(y_test, model.predict(X_test_stats))
    top3 = top_k_accuracy_score(y_test, proba_test, k=3)
    x1 = int(top1 * n_test)
    x3 = int(top3 * n_test)

    # 'wilson' method
    ci1_low, ci1_high = proportion_confint(x1, n_test, alpha=0.05, method='wilson')
    ci3_low, ci3_high = proportion_confint(x3, n_test, alpha=0.05, method='wilson')
    
    print(f"Top‑1 Accuracy: {top1:.4f} (95% CI [{ci1_low:.4f}, {ci1_high:.4f}])")
    print(f"Top‑3 Accuracy: {top3:.4f} (95% CI [{ci3_low:.4f}, {ci3_high:.4f}])")
    print(top1 - ci1_low, ci1_high - top1, ((top1 - ci1_low) + (ci1_high - top1))/2)
    print(top3 - ci3_low, ci3_high - top3, ((top3 - ci3_low) + (ci3_high - top3))/2)
    
    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred, labels=sorted(label_to_subject.keys()))
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # Normalize rows

    plt.figure(figsize=(12, 10))

    # Ensure subjects list matches the order of labels in the confusion matrix
    sorted_labels = sorted(label_to_subject.keys())
    subjects = [label_to_subject[i] for i in sorted_labels]

    sns.heatmap(
        cm_normalized, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=subjects, yticklabels=subjects,
        vmin=0, vmax=1
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    #plt.title(f'Confusion Matrix ({model_type.upper()})')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f'confusion_matrix_{model_type}.png'))
    plt.close()

    print("\nPlotting per-class performance metrics...")
    try:
        report_dict = classification_report(y_test, y_pred, labels=sorted_labels, target_names=subjects, output_dict=True, zero_division=0)
    except ValueError as e:
        print(f"Error generating classification report for plotting: {e}")
        print("Skipping per-class metrics plot.")
        return

    metrics = {'Precision': [], 'Recall': [], 'F1-score': []}
    plot_subjects = []

    for subj in subjects:
        if subj in report_dict:
            metrics['Precision'].append(report_dict[subj]['precision'])
            metrics['Recall'].append(report_dict[subj]['recall'])
            metrics['F1-score'].append(report_dict[subj]['f1-score'])
            plot_subjects.append(subj)

    if not plot_subjects:
        print("No subjects found in the classification report. Skipping metrics plot.")
        return

    plt.figure(figsize=(15, 8))
    x = np.arange(len(plot_subjects)) # Use length of subjects actually plotted
    width = 0.25
    
    plt.bar(x - width, metrics['Precision'], width, label='Precision')
    plt.bar(x, metrics['Recall'], width, label='Recall')
    plt.bar(x + width, metrics['F1-score'], width, label='F1-score')
    
    plt.xlabel('Label')
    plt.ylabel('Score')
    plt.title(f'Performance Metrics by Label ({model_type.upper()})')
    plt.xticks(x, plot_subjects, rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f'class_performance_{model_type}.png'))
    plt.close()
    print("Analysis plots saved.")


def load_processed_data(model_dir):
    """ Load previously processed features and labels. """
    data_path = os.path.join(model_dir, 'processed_data.pkl')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No processed data found at {data_path}. Run without --use_cached_data first.")
    print(f"Loading cached processed data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return (
        data['features'], 
        data['labels'], 
        data['file_paths'], 
        data['prompt_ids'], 
        data['label_to_subject']
    )

def save_processed_data(features, labels, file_paths, prompt_ids, label_to_subject, model_dir):
    """ Save processed features and labels. """
    data_path = os.path.join(model_dir, 'processed_data.pkl')
    print(f"Saving processed data to {data_path}...")
    data_to_save = {
        'features': features,
        'labels': labels,
        'file_paths': file_paths,
        'prompt_ids': prompt_ids,
        'label_to_subject': label_to_subject,
    }
    with open(data_path, 'wb') as f:
        pickle.dump(data_to_save, f)


#################
# Main Function #
#################

def main():
    parser = argparse.ArgumentParser(description='Train a language classifier based on LLM network timings.')
    parser.add_argument('--model_type', type=str, default='mlp', choices=['mlp', 'rf', 'stacking', 'xgb', 'svc', 'lgbm', 'cnn'], 
                        help='Type of model to train (mlp, rf)')
    parser.add_argument('--use_cached_data', action='store_true', 
                        help='Load previously processed features/labels instead of reprocessing pcap files.')
    parser.add_argument('--scaler_type', type=str, default='standard', choices=['minmax', 'standard'],
                        help='Type of scaler to use (minmax or standard).') # Added scaler choice
    args = parser.parse_args()
    
    train_dir = "data/train"
    test_dir = "data/test"
    train_cache_dir = f"train_data/"
    test_cache_dir = f"test_data/"

    model_dir = f"model_{args.model_type}_{args.scaler_type}/"
    os.makedirs(model_dir, exist_ok=True)
    
    if args.use_cached_data:
        try:
            # Load train data
            print(f"Loading cached TRAIN data from {train_cache_dir}…")
            X_train, y_train, train_paths, train_prompt_ids, label_to_subject = load_processed_data(train_cache_dir)

            # Load test data
            print(f"Loading cached TEST data from {test_cache_dir}…")
            X_test, y_test, test_paths, test_prompt_ids, _ = load_processed_data(test_cache_dir)
        except FileNotFoundError as e:
            print(e)
            print("Attempting to process dataset instead...")
            args.use_cached_data = False 

    if not args.use_cached_data:
        print(f"Processing TRAINING data ({train_dir})…")
        X_train, y_train, train_paths, train_prompt_ids, label_to_subject = process_dataset(train_dir, model_type=args.model_type)

        print(f"Processing TEST data ({test_dir})…")
        X_test, y_test, test_paths, test_prompt_ids, _ = process_dataset(test_dir, model_type=args.model_type)

        os.makedirs(train_cache_dir, exist_ok=True)
        os.makedirs(test_cache_dir,  exist_ok=True)
        save_processed_data(X_train, y_train, train_paths, train_prompt_ids, label_to_subject, train_cache_dir)
        save_processed_data(X_test,  y_test,  test_paths,  test_prompt_ids,  label_to_subject, test_cache_dir)

    scaler = None
    if args.scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif args.scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid scaler type specified")
        
    X_train[:, STATS_SLICE] = scaler.fit_transform(X_train[:, STATS_SLICE])
    X_test[:,  STATS_SLICE] = scaler.transform(X_test[:,  STATS_SLICE])

    # --- Model Training ---
    print(f"\nTraining {args.model_type.upper()} model...")
    model_data = train_model(X_train, y_train, X_test, y_test, scaler, model_type=args.model_type, label_to_subject=label_to_subject)
    
    # --- Save Model & Data ---
    model_save_path = os.path.join(model_dir, f'language_classifier_{args.model_type}')

    with open(f"{model_save_path}_data.pkl", 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model data saved to {model_save_path}_data.pkl")
         
    analyze_results(model_data, label_to_subject, model_dir, args.model_type)
    
    print("\nDone!")

if __name__ == "__main__":
    np.random.seed(42)
    main()