import pandas as pd
import geoip2.database
from geopy.distance import geodesic
from collections import defaultdict
import os

# File paths
INPUT_FILE = "/home/rouf/data/raw/net_eco_dataset/public_only_netflow_data.csv"
OUTPUT_FILE = "/home/rouf/data/raw/net_eco_dataset/netflow_grouped_by_src_dst.csv"
GEOLITE_DB = "/home/rouf/data/raw/uploads/locationdb/GeoLite2-City.mmdb"

# Load GeoIP database once
print("Loading GeoIP database...")
reader = geoip2.database.Reader(GEOLITE_DB)

# Cache for geolocation lookups (avoid repeated lookups for same IP)
location_cache = {}

def get_location(ip):
    """Get latitude and longitude for an IP address with caching"""
    if ip in location_cache:
        return location_cache[ip]
    
    try:
        r = reader.city(str(ip))
        loc = (r.location.latitude, r.location.longitude)
    except:
        loc = (None, None)
    
    location_cache[ip] = loc
    return loc

def calculate_distance(src_ip, dst_ip):
    """Calculate geodesic distance between two IPs in kilometers"""
    src_loc = get_location(src_ip)
    dst_loc = get_location(dst_ip)
    
    if None in src_loc or None in dst_loc:
        return None
    
    try:
        return geodesic(src_loc, dst_loc).km
    except:
        return None

# Data structure to store aggregated results
aggregated_data = defaultdict(lambda: {'count': 0, 'distance': None, 'total_demand': 0.0})

print("=" * 80)
print("PROCESSING AND AGGREGATING NETFLOW DATA")
print("=" * 80)
print(f"Input: {INPUT_FILE}\n")

# Process file in chunks
chunk_size = 100000
total_flows = 0
unique_pairs = 0

for chunk_num, chunk in enumerate(pd.read_csv(INPUT_FILE, chunksize=chunk_size), 1):
    total_flows += len(chunk)
    
    for _, row in chunk.iterrows():
        src_ip = str(row['IPV4_SRC_ADDR'])
        dst_ip = str(row['IPV4_DST_ADDR'])
        
        # Calculate demand in Terabits
        demand = ((row['IN_BYTES'] + row['OUT_BYTES']) / 2) * 8 / (1e12)
        
        # Use tuple as key
        pair_key = (src_ip, dst_ip)
        
        # Update aggregated data
        aggregated_data[pair_key]['count'] += 1
        aggregated_data[pair_key]['total_demand'] += demand
        
        # Calculate distance only once per unique pair
        if aggregated_data[pair_key]['distance'] is None:
            distance = calculate_distance(src_ip, dst_ip)
            aggregated_data[pair_key]['distance'] = distance
    
    unique_pairs = len(aggregated_data)
    unique_ips_cached = len(location_cache)
    
    print(f"Chunk {chunk_num}: Flows {total_flows:,} | Pairs {unique_pairs:,} | Cached IPs {unique_ips_cached:,}", 
          end='\r', flush=True)

print(f"\n\n{'='*80}")
print("AGGREGATION COMPLETE")
print("=" * 80)
print(f"Total flows processed:           {total_flows:,}")
print(f"Unique (src, dst) pairs:         {unique_pairs:,}")
print(f"Unique IPs cached:               {len(location_cache):,}")

# Convert to DataFrame
print("\nConverting to DataFrame...")
result_data = []

for (src_ip, dst_ip), stats in aggregated_data.items():
    # Only include pairs where we could calculate distance
    if stats['distance'] is not None:
        result_data.append({
            'IPV4_SRC_ADDR': src_ip,
            'IPV4_DST_ADDR': dst_ip,
            'flow_count': stats['count'],
            'distance': stats['distance'],
            'demand': stats['total_demand']
        })

df_result = pd.DataFrame(result_data)

# Sort by flow_count descending (most common pairs first)
df_result = df_result.sort_values('flow_count', ascending=False)

# Save to CSV
print(f"Saving to {OUTPUT_FILE}...")
df_result.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print(f"Pairs with valid distance:       {len(df_result):,}")
print(f"Total flows represented:         {df_result['flow_count'].sum():,}")
print(f"Total demand (Terabits):         {df_result['demand'].sum():.6f} Tb")
print(f"\nOutput saved to: {OUTPUT_FILE}")

# Show sample statistics
print("\n" + "=" * 80)
print("TOP 10 PAIRS BY FLOW COUNT")
print("=" * 80)
print(df_result.head(10).to_string(index=False))

print("\n" + "=" * 80)
print("STATISTICS")
print("=" * 80)
print("\nDistance (km):")
print(f"  Min:    {df_result['distance'].min():.2f}")
print(f"  Max:    {df_result['distance'].max():.2f}")
print(f"  Mean:   {df_result['distance'].mean():.2f}")
print(f"  Median: {df_result['distance'].median():.2f}")

print("\nDemand (Terabits):")
print(f"  Min:    {df_result['demand'].min():.9f}")
print(f"  Max:    {df_result['demand'].max():.6f}")
print(f"  Mean:   {df_result['demand'].mean():.9f}")
print(f"  Total:  {df_result['demand'].sum():.6f}")

print("\nFlow count per pair:")
print(f"  Min:    {df_result['flow_count'].min():,}")
print(f"  Max:    {df_result['flow_count'].max():,}")
print(f"  Mean:   {df_result['flow_count'].mean():.2f}")
print(f"  Median: {df_result['flow_count'].median():.0f}")

print("\n" + "=" * 80)

# Close GeoIP reader
reader.close()