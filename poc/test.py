import pandas as pd
import numpy as np
from geopy.distance import geodesic
import geoip2.database

# Load your CSV
df = pd.read_csv("netflow.csv")

# Load the GeoIP database
reader = geoip2.database.Reader("GeoLite2-City.mmdb")

# Get location (lat, lon) from IP
def get_location(ip):
    try:
        r = reader.city(ip)
        return (r.location.latitude, r.location.longitude)
    except:
        return (None, None)

# Apply GeoIP lookup
df['src_loc'] = df['IPV4_SRC_ADDR'].apply(get_location)
df['dst_loc'] = df['IPV4_DST_ADDR'].apply(get_location)

# Compute geodesic distance
def calc_distance(row):
    if None in row['src_loc'] or None in row['dst_loc']:
        return np.nan
    return geodesic(row['src_loc'], row['dst_loc']).miles

df['distance'] = df.apply(calc_distance, axis=1)

# Compute demand in Mbps
df['demand'] = ((df['IN_BYTES'] + df['OUT_BYTES']) / 2) * 8 / 1e6

# Save results
df[['FLOW_ID', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'distance', 'demand']].to_csv("netflow_with_distance_demand.csv", index=False)
