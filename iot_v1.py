import requests
import time
import csv
import os

# Settings
SERVER_URL = "http://127.0.0.1:8000/infer"
LOG_FILE = "latency_experiment_results.csv"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "network_type", "client_total_ms", "server_total_ms", "network_overhead_ms", "status"])

def run_inference_test(image_path, network_label="Wi-Fi"):
    
    # Client measurement start time
    client_start = time.perf_counter()
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')} 
            
            response = requests.post(SERVER_URL, files=files, timeout=10)
            
        # Response reception completion time
        client_total_ms = (time.perf_counter() - client_start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract internal processing time provided by the server
            server_total_ms = data["server_timing_ms"]["total_ms"]
            
            # Calculate network overhead: network_overhead_ms = client_total_ms - server_total_ms
            network_overhead_ms = client_total_ms - server_total_ms
            
            # Results
            print(f"[{network_label}] Analysis complete: {data['result']['topk']}")
            print(f" - Client: {client_total_ms:.2f}ms")
            print(f" - Server: {server_total_ms:.2f}ms")
            print(f" - Network: {network_overhead_ms:.2f}ms\n")
            
            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([data["request_id"], network_label, round(client_total_ms, 2), 
                                 round(server_total_ms, 2), round(network_overhead_ms, 2), "ok"])
        else:
            print(f"Server Error: {response.status_code}")
            
    except Exception as e:
        print(f"Transfer failed: {e}")

# Actual experiment loop
if __name__ == "__main__":
    TEST_IMAGE = "broken2.jpeg" # Path to the image file used for the experiment
    NETWORK_ENV = "Wi-Fi"
    
    print(f"--- Running {NETWORK_ENV} environment test ---")
    
    for i in range(1, 101):
        print(f"[{i}/100] Transferring...")
        run_inference_test(TEST_IMAGE, network_label=NETWORK_ENV)
        time.sleep(1) # To prevent server overload