# Transit ISP Tiered Pricing: Network Economics Analysis

Game-theoretic analysis of tiered pricing competition between Transit ISPs using real network flow data.

## Quick Start

```bash
# 1. Prepare dataset (if needed)
cd dataset && python prepare_dataset.py

# 2. Run single scenario (10x10 strategy matrix)
python -m exp.src.main --base-path output --gamma 0.005 --beta 0.5 --alpha 2.0 --s0 0.3 --max-tiers 10

# 3. Run all scenarios (1x1, 2x2, ..., 10x10)
python -m exp.src.main --base-path output --gamma 0.005 --beta 0.5 --alpha 2.0 --s0 0.3

# 4. Run parameter sweep
cd exp && python runner.py
```

---

## 1. Dataset Preparation

### Source Data
Combined network flow data from 5 public cybersecurity datasets (82M flows, 3.1 GB):

| Dataset | Source | Rows |
|---------|--------|------|
| Appraise H2020 | [Kaggle](https://www.kaggle.com/datasets/ittibydgoszcz/appraise-h2020-real-labelled-netflow-dataset) | 15.1M |
| NF-BoT-IoT-v2 | [UQ NIDS](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) | 16.9M |
| NF-CSE-CIC-IDS2018-v2 | [UQ NIDS](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) | 20.1M |
| NF-ToN-IoT-v2 | [UQ NIDS](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) | 27.5M |
| NF-UNSW-NB15-v2 | [UQ NIDS](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) | 2.4M |

### Processing Pipeline

**Prerequisites**:
```bash
pip install pandas geopy geoip2
# Download GeoLite2-City.mmdb from MaxMind
```

**Steps**:
1. Extract essential columns: `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `IN_BYTES`, `OUT_BYTES`, `IN_PKTS`, `OUT_PKTS`
2. Filter private IPs (keep only public-to-public: 17.5M flows, 21.3%)
3. Geolocate IPs and calculate distances
4. Aggregate by unique (source, dest) pairs
5. Compute demand: `demand = ((IN_BYTES + OUT_BYTES) / 2) * 8 / 1e12` (Terabits)

**Output**: `dataset/netflow.csv`
- 13,566 unique IP pairs
- 17.4M flows aggregated
- 3.607 Tb total demand
- Distance range: 0-19,309 km (mean: 9,018 km)

---

## 2. Running Experiments

### Model Parameters

| Parameter | Symbol | Description | Example |
|-----------|--------|-------------|---------|
| `--gamma` | γ | Base transit cost per Tb | 0.005 |
| `--beta` | β | Distance-cost scaling | 0.5 |
| `--alpha` | α | Consumer valuation sensitivity | 2.0 |
| `--s0` | s₀ | Initial market share | 0.3 |
| `--max-tiers` | n | Max pricing tiers (creates n×n matrix) | 10 |

### Command Options

**Single scenario** (one strategy space size):
```bash
python -m exp.src.main --base-path output --gamma 0.001 --beta 0.3 --alpha 1.77 --s0 0.32 --max-tiers 10
```

**All scenarios** (1×1, 2×2, ..., 10×10):
```bash
python -m exp.src.main --base-path output --gamma 0.001 --beta 0.3 --alpha 1.77 --s0 0.32
```

**Parameter sweep** (multiple parameter combinations):
```bash
cd exp
# Edit params.json with desired parameter ranges
python runner.py
```

Example `params.json`:
```json
{
  "gamma": [0.001],
  "beta": [0.3],
  "alpha": [0.1, 0.6, 1.1, 1.6, 2.1],
  "s0": [0.35]
}
```

---

## 3. Output Files

Each run creates a folder: `output/run_XXXXX_gX.XX_bX.XX_aX.XX_sX.XX/NxN/`

**Generated files**:
- `payoff_matrix.csv` - Complete payoff matrix with (profit_A, profit_B) tuples
- `payoff_matrix.pdf` - Heatmap visualization with Nash equilibria marked
- `welfare_matrix.json` - Consumer surplus, producer surplus, social welfare for each cell
- `summary.json` - Nash equilibria, welfare metrics, strategy details
- `strategy_details.json` - Pricing structure for each tier strategy
- `analysis.log` - Execution log

---

## 4. Project Structure

```
tiered_pricing_network_economics/
├── dataset/
│   ├── prepare_dataset.py       # Dataset preprocessing script
│   └── netflow.csv               # Processed network flow data (output)
├── exp/
│   ├── src/
│   │   ├── models/
│   │   │   ├── pricing.py        # Cost, valuation, tier pricing
│   │   │   └── competition.py    # Payoff matrix, Nash equilibria
│   │   ├── analysis/
│   │   │   ├── welfare.py        # Surplus & efficiency metrics
│   │   │   └── visualization.py  # Heatmap generation
│   │   ├── utils/
│   │   │   ├── logger.py         # Logging
│   │   │   └── io_handler.py     # JSON/CSV I/O
│   │   └── main.py               # Pipeline orchestration
│   ├── runner.py                 # Parameter sweep runner
│   └── params.json               # Parameter configurations
└── README.md
```

---

## 5. Key Features

✅ **Endogenous P₀**: Base price calculated from network demand (no manual input)  
✅ **Nash Equilibria**: Automatic detection of pure strategy equilibria  
✅ **Welfare Analysis**: Consumer surplus, producer surplus, social welfare  
✅ **Modular Design**: 8 independent, testable components  
✅ **Comprehensive Output**: CSV matrices, PDF visualizations, JSON summaries

---

## 6. Example Workflow

```bash
# Step 1: Clone repository
git clone git@github.com:NWSL-UCF/tiered_pricing_network_economics.git
cd tiered_pricing_network_economics

# Step 2: Install dependencies
pip install pandas numpy matplotlib geopy geoip2

# Step 3: Prepare dataset (if not already done)
cd dataset
python prepare_dataset.py
cd ..

# Step 4: Run single experiment
python -m exp.src.main --base-path results --gamma 0.001 --beta 0.3 --alpha 1.77 --s0 0.32

# Step 5: View results
ls results/run_*/10x10/
# Check payoff_matrix.pdf for visualization
# Check summary.json for Nash equilibria
```

---
**Last Updated**: October 2025
