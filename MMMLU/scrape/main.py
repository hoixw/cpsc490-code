import subprocess
import os
import time
import socket
import requests
import json
import threading
import csv
import random
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from requests.adapters import HTTPAdapter
import signal

MODEL = "openai/gpt-4o-mini"
OPENROUTER_API_KEY = ""
PROVIDER = "OpenAI"
LOCAL_IP = "1.1.1.1"
SCRAPE_TRAIN = True
SCRAPE_TEST = True
NUM_WORKERS = 8

# -------------------------------
# Part 1: Custom Adapter to bind a specific local port
# -------------------------------
class LocalPortAdapter(HTTPAdapter):
    def __init__(self, source_address, *args, **kwargs):
        self.source_address = source_address
        super().__init__(*args, **kwargs)
    
    def init_poolmanager(self, connections, maxsize, block=False, **kwargs):
        if self.source_address:
            kwargs['source_address'] = self.source_address
        super().init_poolmanager(connections, maxsize, block=block, **kwargs)

# -------------------------------
# Part 2: Utility to get a free local port
# -------------------------------
def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('0.0.0.0', 0))  # Let OS pick a free port
    addr, port = s.getsockname()
    s.close()
    return port

# -------------------------------
# Part 3: Functions to start/stop tcpdump and check retransmissions
# -------------------------------

def start_tcpdump(local_port, output_file):
    ## We capture all inbound traffic (matching static IP + port)
    ## Capture outbound traffic IFF > 99 bytes — ignore ACKs
    filter_expression = (
        f"((dst host {LOCAL_IP} and dst port {local_port}) or "
        f"(src host {LOCAL_IP} and src port {local_port} and greater 99))"
    )
    cmd = [
        "tcpdump",
        "-i", "any",
        "-s", "0",
        "-U",
        "-B", "4096",
        filter_expression,
        "-w", output_file
    ]
    print(f"[tcpdump] Starting capture on port {local_port}: {output_file}")
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE)
    time.sleep(2)  
    return process

def stop_tcpdump(process, output_file, timeout=30, check_interval=2, inactivity_period=5):
    start_time = time.time()
    last_size = 0
    last_change_time = start_time

    while time.time() - start_time < timeout:
        current_size = os.path.getsize(output_file)
        if current_size > last_size:
            last_size = current_size
            last_change_time = time.time()
        elif time.time() - last_change_time >= inactivity_period:
            # No new packets written for inactivity_period seconds
            break
        time.sleep(check_interval)

    process.send_signal(signal.SIGINT)
    _, stderr = process.communicate()
    output = stderr.decode("utf-8")

    metrics = {
        'packets_captured': 0,
        'packets_received': 0,
        'packets_dropped': 0
    }
    for line in output.split('\n'):
        if 'packets captured' in line:
            try:
                metrics['packets_captured'] = int(line.split()[0])
            except:
                pass
        elif 'packets received by filter' in line:
            try:
                metrics['packets_received'] = int(line.split()[0])
            except:
                pass
        elif 'packets dropped by kernel' in line:
            try:
                metrics['packets_dropped'] = int(line.split()[0])
            except:
                pass
    return metrics

def check_retransmissions(pcap_filename):
    cmd = ['tshark', '-r', pcap_filename, '-Y', 'tcp.analysis.retransmission']
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    count = len(result.stdout.splitlines())
    return count

def check_frame_wirelen(pcap_filename, target_length=125):
    """
    Check for 125-byte packets that occur BETWEEN data packets.
    We only care about delays that interrupt the actual data flow.
    """
    # First get all packets
    cmd = ['tshark', '-r', pcap_filename, '-T', 'fields', '-e', 'frame.number', '-e', 'frame.len']
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    # Parse the output into (packet_num, length) pairs
    packets = []
    for line in result.stdout.decode('utf-8').strip().split('\n'):
        if line:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    packets.append((int(parts[0]), int(parts[1])))
                except ValueError:
                    continue
    
    # Find the last ACK packet in the first 15 packets
    cmd = ['tshark', '-r', pcap_filename, '-Y', 'tcp.flags.push==0', '-T', 'fields', '-e', 'frame.number', '-c', '15']
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    ack_packets = [int(pkt) for pkt in result.stdout.decode('utf-8').strip().split('\n') if pkt]
    if not ack_packets:
        return 0  # No ACK packets found
    
    last_ack = max(ack_packets) + 1
    
    # Find the first data packet (non-125 byte packet after the last ACK)
    first_data_idx = -1
    for i, (pkt_num, length) in enumerate(packets):
        if pkt_num > last_ack and length != target_length and length > 125:
            first_data_idx = i
            break
    
    if first_data_idx == -1:
        return 0  # No data packets found
    
    # Check for target_length packets that appear AFTER first data packet 
    # and BETWEEN other data packets
    problematic_packets = 0
    saw_data_after_delay = False
    for i in range(first_data_idx+1, len(packets)):
        pkt_num, length = packets[i]
        
        # If this is a delay packet
        if length == target_length:
            # If we've seen a data packet after last delay, this is problematic
            if saw_data_after_delay:
                problematic_packets += 1
        # If this is a data packet
        elif length > 100:
            saw_data_after_delay = True
    
    return problematic_packets

# -------------------------------
# Part 4: Load and organize train.csv and test.csv and prepare folders/CSV logs
# -------------------------------
# Create main data directory and train/test subdirectories
os.makedirs("data", exist_ok=True)
os.makedirs("data/train", exist_ok=True)
os.makedirs("data/test", exist_ok=True)

# Function to check CSV import and display summary information
def check_csv_import(file_path, dataset_type):
    print(f"Checking {dataset_type} CSV import from {file_path}...")
    try:
        # Load data with explicit UTF-8 encoding
        data_df = pd.read_csv(file_path, encoding='utf-8')
        
        # Print basic information
        print(f"CSV loaded successfully. Shape: {data_df.shape}")
        print(f"Columns: {data_df.columns.tolist()}")
        
        # Check for required columns
        required_columns = ['query', 'Subject', 'language', 'id']
        missing_columns = [col for col in required_columns if col not in data_df.columns]
        
        if missing_columns:
            print(f"Warning: Missing required columns: {missing_columns}")
            print(f"Available columns: {data_df.columns.tolist()}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Count unique languages
        unique_languages = data_df['language'].unique()
        
        print(f"\nFound {len(unique_languages)} unique languages:")
        for language in unique_languages:
            language_df = data_df[data_df['language'] == language]
            count = len(language_df)
            avg_length = language_df['query'].str.len().mean()
            print(f"  - {language}: {count} entries, avg length: {avg_length:.1f} chars")
        
        # Sample entries from each language to verify
        print(f"\nSample entries from {dataset_type} for each language:")
        for language in unique_languages:
            samples = data_df[data_df['language'] == language].sample(min(2, len(data_df[data_df['language'] == language])))
            for _, row in samples.iterrows():
                print(f"  - {language}: {row['query'][:100]}...")
                
        # Check for any missing values
        missing_values = data_df.isnull().sum()
        if missing_values.sum() > 0:
            print("\nWarning: Found missing values in the dataset:")
            for col, count in missing_values.items():
                if count > 0:
                    print(f"  - {col}: {count} missing values")
        
        return data_df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise

# Load train and test data
train_df = check_csv_import("data/train.csv", "train")
test_df = check_csv_import("data/test.csv", "test")

# Define languages based on the CSVs
languages_list = sorted(list(set(train_df['language'].unique()) | set(test_df['language'].unique())))

# Make language folder names safe for filesystem
def get_safe_folder_name(language):
    return "".join(c if c.isalnum() else "_" for c in language).strip("_")

# Group queries by language and dataset type
train_language_groups = {language: [] for language in languages_list}
test_language_groups = {language: [] for language in languages_list}

# Process train data
for idx, row in train_df.iterrows():
    language = row["language"]
    if language in train_language_groups:
        train_language_groups[language].append({
            "query": row["query"],
            "subject": row["Subject"],
            "id": row["id"],
            "language_code": get_safe_folder_name(language),
            "global_index": idx,
            "sent_count": 0
        })

# Process test data
for idx, row in test_df.iterrows():
    language = row["language"]
    if language in test_language_groups:
        test_language_groups[language].append({
            "query": row["query"],
            "subject": row["Subject"],
            "id": row["id"],
            "language_code": get_safe_folder_name(language),
            "global_index": idx,
            "sent_count": 0
        })

# Create folders and logs for both train and test datasets
csv_locks = {}

for dataset_type, language_groups in [("train", train_language_groups), ("test", test_language_groups)]:
    for language in languages_list:
        # Skip if there are no queries for this language in this dataset
        if not language_groups[language]:
            continue
            
        # Create language directory inside data/[train|test] folder
        language_code = get_safe_folder_name(language)
        os.makedirs(f"data/{dataset_type}/{language_code}", exist_ok=True)
        
        csv_path = os.path.join(f"data/{dataset_type}/{language_code}", "query_metrics.csv")
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'query_id', 'query', 'response', 'pcap_file',
                    'packets_captured', 'packets_received', 'packets_dropped',
                    'prompt_tokens', 'completion_tokens', 'total_tokens',
                    'language', 'language_code', 'subject', 'id'
                ])
        
        lock_key = f"{dataset_type}_{language}"
        csv_locks[lock_key] = threading.Lock()

# -------------------------------
# Global parameters
# -------------------------------
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}
num_samples = 1   # Each query will be sent this many times
MAX_RETRIES = 15

# -------------------------------
# Part 5: Define the send_query function
# -------------------------------
def send_query(local_port, query, model, headers, output_file):
    """
    1. Starts tcpdump (saving to output_file).
    2. Creates a requests.Session bound to a free local port.
    3. Sends the query (streaming response) to the LLM.
    4. Stops tcpdump and returns response text, the parsed final SSE content, and metrics.
    """
    tcpdump_proc = start_tcpdump(local_port, output_file)

    session = requests.Session()
    adapter = LocalPortAdapter(('0.0.0.0', local_port))
    session.mount("https://", adapter)

    response_text = []
    recContent = None
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "stream": True,
        "provider": {
            'order': [PROVIDER],
            "allow_fallbacks": False
        }
    }

    buffer = ""
    done = False
    try:
        with session.post(url, headers=headers, json=payload, stream=True, timeout=60) as r:
            for chunk in r.iter_content(chunk_size=1024, decode_unicode=True):
                if done:
                    break
                buffer += chunk
                while True:
                    line_end = buffer.find('\n')
                    if line_end == -1:
                        break
                    line = buffer[:line_end].strip()
                    buffer = buffer[line_end + 1:]
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            done = True
                            break
                        try:
                            data_obj = json.loads(data)
                            recContent = data_obj
                            content = data_obj["choices"][0]["delta"].get("content")
                            if content:
                                response_text.append(content)
                        except json.JSONDecodeError:
                            pass
    except Exception as e:
        raise e

    metrics = stop_tcpdump(tcpdump_proc, output_file)
    return "".join(response_text), recContent, metrics

# -------------------------------
# Part 6: Define a task to process a single query (with retries)
# -------------------------------
def process_task(dataset_type, language, query_obj, sample_round, query_number):
    query_id = f"{query_number}_{sample_round}"
    attempt = 0
    pcap_filename = None
    language_code = query_obj["language_code"]
    
    # Formatted query is already prepared in the query_obj
    query_text = query_obj['query']
    
    while attempt < MAX_RETRIES:
        try:
            local_port = get_free_port()
            pcap_filename = os.path.join("data", dataset_type, language_code, f"query_{query_id}.pcap")
            response, recContent, metrics = send_query(local_port, query_text, MODEL, headers, pcap_filename)
            
            # Check for broken pcap files
            if metrics['packets_captured'] <= 0 or abs(metrics['packets_captured'] - metrics['packets_received']) > 2:
                print(f"[{dataset_type}/{language}] Packet issue for query {query_id} on attempt {attempt+1}. Retrying...")
                print(metrics)
                # Clean up failed pcap
                if os.path.exists(pcap_filename):
                    os.remove(pcap_filename)
                attempt += 1
                time.sleep(5)
                continue
            
            # Check for TCP retransmissions
            retransmissions = check_retransmissions(pcap_filename)
            if retransmissions > 3:
                print(f"[{dataset_type}/{language}] Found {retransmissions} TCP retransmissions for query {query_id} on attempt {attempt+1}. Retrying...")
                # Clean up pcap with retransmissions
                if os.path.exists(pcap_filename):
                    os.remove(pcap_filename)
                attempt += 1
                time.sleep(5)
                continue
            
            # Check for frame wire length
            frame_wirelen_count = check_frame_wirelen(pcap_filename)
            if frame_wirelen_count > 0:
                print(f"[{dataset_type}/{language}] Found {frame_wirelen_count} packets with frame wire length 125 (expected 1) for query {query_id} on attempt {attempt+1}. Retrying...")
                # Clean up pcap with incorrect frame wire length count
                if os.path.exists(pcap_filename):
                    os.remove(pcap_filename)
                attempt += 1
                time.sleep(5)
                continue
            
            prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
            if recContent and "usage" in recContent:
                usage = recContent["usage"]
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
            
            csv_path = os.path.join("data", dataset_type, language_code, "query_metrics.csv")
            lock_key = f"{dataset_type}_{language}"
            with csv_locks[lock_key]:
                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        query_id, query_text, response, pcap_filename,
                        metrics['packets_captured'], metrics['packets_received'], metrics['packets_dropped'],
                        prompt_tokens, completion_tokens, total_tokens,
                        language, language_code, query_obj['subject'], query_obj['id']
                    ])
            
            print(f"[{dataset_type}/{language}] Successfully processed query {query_id} (attempt {attempt+1}).")
            query_obj["sent_count"] += 1  # (For logging purposes)
            break  # exit retry loop
            
        except Exception as e:
            print(f"[{dataset_type}/{language}] Error processing query {query_id} on attempt {attempt+1}: {e}. Retrying...")
            # Clean up failed pcap
            if pcap_filename and os.path.exists(pcap_filename):
                os.remove(pcap_filename)
            attempt += 1
            time.sleep(random.uniform(10, 25))
    
    # If all retries failed, log the failure
    if attempt >= MAX_RETRIES:
        print(f"[{dataset_type}/{language}] Failed to process query {query_id} after {MAX_RETRIES} attempts")
        csv_path = os.path.join("data", dataset_type, language_code, "query_metrics.csv")
        lock_key = f"{dataset_type}_{language}"
        with csv_locks[lock_key]:
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    query_id, query_text, "FAILED", "N/A",
                    0, 0, 0, 0, 0, 0,
                    language, language_code, query_obj['subject'], query_obj['id']
                ])

# -------------------------------
# Part 7: Schedule and run all tasks
# -------------------------------
def run_all_tasks():
    tasks = []
    # Create list of all tasks first - train dataset
    print(f"Setting up train tasks...")
    if SCRAPE_TRAIN:
        for language in languages_list:
            if language in train_language_groups and train_language_groups[language]:
                print(f"  - Adding {len(train_language_groups[language])} queries for language: {language}")
                for query_number, query_obj in enumerate(train_language_groups[language], start=1):
                    for sample_round in range(1, num_samples + 1):
                        # Store task parameters as tuple: dataset_type, language, query_obj, sample_round, query_number
                        tasks.append(("train", language, query_obj, sample_round, query_number))
    
    # Create list of all tasks - test dataset
    if SCRAPE_TEST:
        print(f"Setting up test tasks...")
        for language in languages_list:
            if language in test_language_groups and test_language_groups[language]:
                print(f"  - Adding {len(test_language_groups[language])} queries for language: {language}")
                for query_number, query_obj in enumerate(test_language_groups[language], start=1):
                    for sample_round in range(1, num_samples + 1):
                        # Store task parameters as tuple: dataset_type, language, query_obj, sample_round, query_number
                        tasks.append(("test", language, query_obj, sample_round, query_number))
    
    print(f"Total tasks: {len(tasks)}")
    
    # Shuffle the tasks
    random.shuffle(tasks)
    
    # Execute shuffled tasks
    print(f"Starting execution with {NUM_WORKERS} worker threads...")
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_task, *task) for task in tasks]
        for future in futures:
            future.result()
    print("All tasks complete.")

if __name__ == "__main__":
    # Run the task processing
    print("\nRunning full task processing for MMMLU queries across languages...")
    run_all_tasks()
