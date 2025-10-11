# Network Flow Dataset Preparation

## Dataset Sources

Combined network flow data from five public cybersecurity datasets:

| # | Dataset | Source | Rows |
|---|---------|--------|------|
| 1 | **Appraise H2020** | [Kaggle Dataset](https://www.kaggle.com/datasets/ittibydgoszcz/appraise-h2020-real-labelled-netflow-dataset) | 15,116,160 |
| 2 | **NF-BoT-IoT-v2** | [UQ NIDS Datasets](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) | 16,933,808 |
| 3 | **NF-CSE-CIC-IDS2018-v2** | [UQ NIDS Datasets](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) | 20,115,529 |
| 4 | **NF-ToN-IoT-v2** | [UQ NIDS Datasets](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) | 27,520,260 |
| 5 | **NF-UNSW-NB15-v2** | [UQ NIDS Datasets](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) | 2,365,424 |

**Total raw flows**: 82,051,181 (3.1 GB)

**Note**: Datasets 2-5 are NetFlow V2 format from the University of Queensland's Machine Learning-Based NIDS Datasets collection.

## Processing Pipeline

### 1. Column Selection
Extracted essential columns only:
```
IPV4_SRC_ADDR, IPV4_DST_ADDR, IN_PKTS, IN_BYTES, OUT_PKTS, OUT_BYTES
```

### 2. Private IP Filtering
Removed flows containing private IPs (10.x.x.x, 172.16-31.x.x, 192.168.x.x, 127.x.x.x):

| Category | Flows | Percentage |
|----------|-------|------------|
| Both IPs local | 51,442,006 | 62.70% |
| Source local only | 5,702,755 | 6.95% |
| Destination local only | 7,429,285 | 9.05% |
| **Both IPs public (kept)** | **17,477,135** | **21.30%** |

### 3. Geolocation & Aggregation
- Geolocated public IPs using MaxMind GeoLite2 City database
- Calculated geodesic distances between IP pairs
- Aggregated flows by unique (source, destination) pairs
- Computed demand in Terabits: `demand = ((IN_BYTES + OUT_BYTES) / 2) * 8 / 1e12`

## Final Dataset

**File**: `netflow_grouped_by_src_dst.csv`

**Format**:
```csv
IPV4_SRC_ADDR,IPV4_DST_ADDR,flow_count,distance,demand
89.159.255.164,5.9.222.138,44925,462.64,0.441961
```

### Statistics
- **Unique IP pairs**: 13,566
- **Total flows**: 17,406,418
- **Total demand**: 3.607 Terabits

**Distance (km)**:
- Min: 0, Max: 19,309, Mean: 9,018, Median: 9,467

**Flow count per pair**:
- Min: 1, Max: 782,147, Mean: 1,283, Median: 5

**Demand (Tb)** — Total data volume per IP pair:
- Min: ~0.000000001, Max: 0.445, Mean: 0.000266

## Dependencies
```bash
pip install pandas geopy geoip2
```

Requires: `GeoLite2-City.mmdb` from MaxMind

## Use Cases
- Network demand modeling based on geographic distance
- Tiered pricing analysis for ISPs
- Network economics simulations with distance-dependent costs

