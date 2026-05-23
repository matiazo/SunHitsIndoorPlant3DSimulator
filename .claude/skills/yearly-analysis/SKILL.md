---
name: yearly-analysis
description: Run yearly sun exposure simulation and generate an interactive HTML report
disable-model-invocation: true
---

# Yearly Sun Exposure Analysis

Run the full-year sun exposure simulation and generate an interactive Plotly HTML dashboard.

## Arguments

- First positional argument: year (defaults to current year)
- `--open` — Open the generated HTML report in the browser after generation

## Steps

### 1. Run the simulation

```bash
cd C:/repo/SunHitsIndoorPlant3DSimulator && python scripts/yearly_analysis.py [YEAR]
```

This computes sun-hit data for every 5-minute interval across all days of the specified year. It takes a few minutes to complete. The script will print progress updates.

### 2. Report location

The generated HTML report is saved to:
```
examples/yearly_analysis.html
```

### 3. Open in browser (if --open)

```bash
start examples/yearly_analysis.html
```

### 4. Summarize results

After the script completes, report:
- Number of windows analyzed
- Per-window summary: days with sun, total sun-hours, date range
- The "Any Window" combined total
- Path to the generated report

## Notes

- Config is loaded from `config/default_config.json`
- The simulation scans 12:00-18:00 for heatmaps but checks all daylight hours for daily stats
- Output is a self-contained HTML file (no server needed, works offline except for Plotly CDN)
- Expect ~2-5 minutes runtime depending on machine
