# Transit ISP Pricing Competition Analysis - Refactored Code

## Overview

Refactored from monolithic `sim.py` (1041 lines) into 8 modular files with clear separation of concerns. All original logic preserved.

## Structure

```
src/
├── models/
│   ├── pricing.py          # Cost, valuation, tier pricing calculations
│   └── competition.py      # Payoff matrix, Nash equilibria, best responses
├── analysis/
│   ├── welfare.py          # Consumer/producer surplus, efficiency metrics
│   └── visualization.py    # Payoff matrix heatmap generation
├── utils/
│   ├── logger.py           # Centralized logging
│   └── io_handler.py       # JSON/CSV I/O, summary generation
└── main.py                 # Pipeline orchestration & CLI
```

## Quick Start

**Single scenario (10x10 matrix)**:
```bash
python -m src.main --base-path output --gamma 0.005 --beta 0.5 --alpha 2.0 --s0 0.3 --max-tiers 10
```

**All scenarios (1x1, 2x2, ..., 10x10)**:
```bash
python -m src.main --base-path output --gamma 0.005 --beta 0.5 --alpha 2.0 --s0 0.3
```

**Parameter sweep** (all scenarios for all parameter combinations):
```bash
python runner.py  # Reads params.json, runs all combinations × all scenarios
```

See `MULTI_SCENARIO_GUIDE.md` for details on multi-scenario analysis.

## Key Changes

| Aspect | Old | New |
|--------|-----|-----|
| **Files** | 1 monolithic file | 8 modular files |
| **Lines/file** | 1041 | ~150 average |
| **P0 parameter** | Required input | Calculated endogenously |
| **Command** | `python sim.py` | `python -m src.main` |
| **Testability** | Difficult | Each module testable |
| **Maintainability** | Low (tight coupling) | High (loose coupling) |

## Module Summary

- **pricing.py**: PricingModel class - all pricing calculations
- **competition.py**: CompetitionModel class - game theory analysis
- **welfare.py**: WelfareAnalyzer class - social welfare metrics
- **visualization.py**: PayoffMatrixVisualizer class - plot generation
- **logger.py**: `setup_logger()` function
- **io_handler.py**: `load_json()`, `save_json()`, `save_summary()`
- **main.py**: TransitISPAnalysis class - coordinates all modules

## Output Files

Each run generates:
- `payoff_matrix.csv` - Complete payoff matrix
- `payoff_matrix.pdf` - Visualized matrix with Nash equilibria
- `summary.json` - Analysis results
- `analysis.log` - Execution log

## Benefits

✅ **Modular** - Independent, testable components  
✅ **Readable** - Clear naming, comprehensive docstrings  
✅ **Maintainable** - Easy to extend and debug  
✅ **Reusable** - Import and use components independently  
✅ **Compatible** - Same results as original code

---

**Refactored**: Oct 2025 | **Original**: `sim.py` (1041 lines) → **New**: 8 modules (~1200 lines)

