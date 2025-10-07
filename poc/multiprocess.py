import pandas as pd
import numpy as np
from geopy.distance import geodesic
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import geoip2.database
import multiprocessing
import os
import gc

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# File paths
input_file = "netflow.csv"
reader_path = "GeoLite2-City.mmdb"
output_file = "netflow_with_distance_demand.csv"
temp_dir = "temp_processing"
chunk_size = 100000  # Adjust based on your memory constraints

# Create temp directory if it doesn't exist
os.makedirs(temp_dir, exist_ok=True)

# Function to resolve IPs in chunks
def resolve_ip_chunk(ips):
    import geoip2.database
    reader = geoip2.database.Reader(reader_path)
    results = {}
    for ip in ips:
        try:
            r = reader.city(ip)
            results[ip] = (r.location.latitude, r.location.longitude)
        except:
            results[ip] = (None, None)
    reader.close()
    return results

# Function to process distances for a chunk
def process_chunk_distances(chunk_data):
    results = []
    for _, row in chunk_data.iterrows():
        src, dst = row['src_loc'], row['dst_loc']
        if src[0] is None or dst[0] is None:
            distance = np.nan
        else:
            distance = geodesic(src, dst).miles
            
        # Calculate demand
        demand = ((row['IN_BYTES'] + row['OUT_BYTES']) / 2) * 8 / 1e6
        
        results.append({
            'FLOW_ID': row['FLOW_ID'],
            'IPV4_SRC_ADDR': row['IPV4_SRC_ADDR'],
            'IPV4_DST_ADDR': row['IPV4_DST_ADDR'],
            'distance': distance,
            'demand': demand
        })
    return results

# Main processing pipeline
def main():
    # Step 1: Extract all unique IPs in chunks
    log("Extracting unique IPs in chunks...")
    unique_ips = set()
    
    for chunk_num, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
        log(f"Processing chunk {chunk_num+1} for unique IPs...")
        src_ips = set(chunk['IPV4_SRC_ADDR'].unique())
        dst_ips = set(chunk['IPV4_DST_ADDR'].unique())
        unique_ips.update(src_ips | dst_ips)
        
    log(f"Total unique IPs to resolve: {len(unique_ips)}")
    
    # Step 2: Resolve IPs in smaller batches to avoid memory issues
    log("Resolving IPs with multiprocessing...")
    ip_cache = {}
    ip_list = list(unique_ips)
    batch_size = min(10000, len(ip_list))  # Adjust based on memory
    
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for i in range(0, len(ip_list), batch_size):
            batch = ip_list[i:i+batch_size]
            log(f"Resolving IP batch {i//batch_size + 1}/{(len(ip_list) + batch_size - 1)//batch_size}")
            batch_results = executor.submit(resolve_ip_chunk, batch).result()
            ip_cache.update(batch_results)
            
    # Free up memory
    del unique_ips, ip_list
    gc.collect()
    
    # Step 3: Process the main data in chunks
    log("Processing main data in chunks...")
    processed_chunks = 0
    
    # Create header in the output file
    pd.DataFrame(columns=['FLOW_ID', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'distance', 'demand']).to_csv(
        output_file, index=False
    )
    
    for chunk_num, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
        log(f"Processing main data chunk {chunk_num+1}...")
        
        # Add location data
        chunk['src_loc'] = chunk['IPV4_SRC_ADDR'].map(ip_cache)
        chunk['dst_loc'] = chunk['IPV4_DST_ADDR'].map(ip_cache)
        
        # Process in smaller sub-chunks for distance calculation
        sub_chunk_size = min(10000, len(chunk))
        all_results = []
        
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = []
            for i in range(0, len(chunk), sub_chunk_size):
                sub_chunk = chunk.iloc[i:i+sub_chunk_size]
                futures.append(executor.submit(process_chunk_distances, sub_chunk))
            
            for future in futures:
                all_results.extend(future.result())
        
        # Save results for this chunk
        result_df = pd.DataFrame(all_results)
        result_df.to_csv(output_file, mode='a', header=False, index=False)
        processed_chunks += 1
        
        # Free memory
        del chunk, all_results, result_df
        gc.collect()
    
    log(f"✅ Completed processing {processed_chunks} chunks. Results saved to {output_file}")

if __name__ == "__main__":
    main()