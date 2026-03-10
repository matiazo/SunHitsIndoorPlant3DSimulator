#!/usr/bin/env python3
"""Yearly per-window plant sun hit analysis with interactive HTML report.

Computes sun-hit data for every 5-minute interval across all days of a year
and generates a self-contained Plotly-powered HTML dashboard.

Usage:
    python scripts/yearly_analysis.py [year]
"""

import json
import sys
import time
import calendar
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sun_hit_detector.core.models import Config
from sun_hit_detector.core.sun_position import generate_sun_data_for_date
from sun_hit_detector.core.hit_test import check_plant_hit_per_window_from_config

CONFIG_PATH = ROOT / "config" / "default_config.json"
OUTPUT_PATH = ROOT / "examples" / "yearly_analysis.html"

# Time-of-day grid for heatmaps (12:00–18:00 at 5-min steps)
HEAT_START_HOUR = 12
HEAT_END_HOUR = 18
HEAT_TIMES = []
for _h in range(HEAT_START_HOUR, HEAT_END_HOUR):
    for _m in range(0, 60, 5):
        HEAT_TIMES.append(f"{_h:02d}:{_m:02d}")
HEAT_TIME_INDEX = {t: i for i, t in enumerate(HEAT_TIMES)}
NUM_TIME_SLOTS = len(HEAT_TIMES)


def compute_yearly_data(year: int, config: Config):
    """Run the simulation for every day of *year* and return structured results."""
    loc = config.location
    window_ids = [w.id for w in config.windows]
    num_days = 366 if calendar.isleap(year) else 365

    # Per-window heatmap grids: window_id -> list[list[int]] (day x time)
    heatmaps = {wid: [[0] * NUM_TIME_SLOTS for _ in range(num_days)] for wid in window_ids}
    any_heatmap = [[0] * NUM_TIME_SLOTS for _ in range(num_days)]

    # Per-window per-day stats: window_id -> list[dict | None]
    daily_stats = {wid: [None] * num_days for wid in window_ids}
    any_daily = [None] * num_days  # combined "any window"

    start_date = date(year, 1, 1)
    t0 = time.time()

    for day_idx in range(num_days):
        current_date = start_date + timedelta(days=day_idx)

        if day_idx % 30 == 0:
            elapsed = time.time() - t0
            pct = (day_idx / num_days) * 100 if num_days else 0
            print(f"  Day {day_idx + 1:>3}/{num_days}  {current_date}  ({pct:.0f}%)  [{elapsed:.1f}s elapsed]")

        sun_data = generate_sun_data_for_date(
            latitude=loc.latitude,
            longitude=loc.longitude,
            target_date=current_date,
            timezone_offset=loc.timezone_offset,
            timezone_name=loc.timezone_name,
            interval_minutes=5,
        )

        # Track first/last hit times and count per window for this day
        win_first = {}
        win_last = {}
        win_count = {}
        any_first = None
        any_last = None
        any_count = 0

        for entry in sun_data:
            ts = entry["timestamp"]
            az = entry["azimuth_deg"]
            el = entry["elevation_deg"]
            if el <= 0:
                continue

            hits = check_plant_hit_per_window_from_config(az, el, config)
            any_hit = False

            for wid, hit in hits.items():
                if not hit:
                    continue
                any_hit = True
                # Update per-window stats
                if wid not in win_first:
                    win_first[wid] = ts
                win_last[wid] = ts
                win_count[wid] = win_count.get(wid, 0) + 1

                # Fill heatmap cell
                ti = HEAT_TIME_INDEX.get(ts)
                if ti is not None:
                    heatmaps[wid][day_idx][ti] = 1

            if any_hit:
                if any_first is None:
                    any_first = ts
                any_last = ts
                any_count += 1
                ti = HEAT_TIME_INDEX.get(ts)
                if ti is not None:
                    any_heatmap[day_idx][ti] = 1

        # Store daily stats
        date_str = current_date.isoformat()
        for wid in window_ids:
            if wid in win_first:
                daily_stats[wid][day_idx] = {
                    "date": date_str,
                    "first": win_first[wid],
                    "last": win_last[wid],
                    "minutes": win_count[wid] * 5,
                }
        if any_first is not None:
            any_daily[day_idx] = {
                "date": date_str,
                "first": any_first,
                "last": any_last,
                "minutes": any_count * 5,
            }

    elapsed = time.time() - t0
    print(f"  Done — {num_days} days computed in {elapsed:.1f}s")

    # Build summary per window
    dates_list = [(start_date + timedelta(days=i)).isoformat() for i in range(num_days)]
    summary = {}
    for wid in window_ids:
        days_with_sun = sum(1 for d in daily_stats[wid] if d is not None)
        total_min = sum(d["minutes"] for d in daily_stats[wid] if d is not None)
        dates_with_hit = [d["date"] for d in daily_stats[wid] if d is not None]
        summary[wid] = {
            "days_with_sun": days_with_sun,
            "total_hours": round(total_min / 60, 1),
            "earliest_date": dates_with_hit[0] if dates_with_hit else "—",
            "latest_date": dates_with_hit[-1] if dates_with_hit else "—",
        }

    # "Any" summary
    any_days = sum(1 for d in any_daily if d is not None)
    any_total = sum(d["minutes"] for d in any_daily if d is not None)
    any_dates = [d["date"] for d in any_daily if d is not None]
    summary["__any__"] = {
        "days_with_sun": any_days,
        "total_hours": round(any_total / 60, 1),
        "earliest_date": any_dates[0] if any_dates else "—",
        "latest_date": any_dates[-1] if any_dates else "—",
    }

    return {
        "year": year,
        "num_days": num_days,
        "window_ids": window_ids,
        "dates": dates_list,
        "times": HEAT_TIMES,
        "heatmaps": heatmaps,
        "any_heatmap": any_heatmap,
        "daily_stats": daily_stats,
        "any_daily": any_daily,
        "summary": summary,
    }


def build_html(data: dict) -> str:
    """Generate a self-contained interactive HTML report."""
    year = data["year"]
    num_days = data["num_days"]
    window_ids = data["window_ids"]
    dates = data["dates"]
    times = data["times"]
    summary = data["summary"]

    # Month tick positions for x-axis
    month_ticks = []
    month_labels = []
    d = date(year, 1, 1)
    for m in range(1, 13):
        try:
            md = date(year, m, 1)
        except ValueError:
            break
        day_of_year = (md - d).days
        if day_of_year < num_days:
            month_ticks.append(day_of_year)
            month_labels.append(md.strftime("%b"))

    # Build daily table rows (JSON for JS rendering)
    daily_rows = []
    for i in range(num_days):
        row = {"date": dates[i]}
        for wid in window_ids:
            ds = data["daily_stats"][wid][i]
            row[wid] = ds if ds else None
        ad = data["any_daily"][i]
        row["__any__"] = ad if ad else None
        daily_rows.append(row)

    # Compact heatmap data for embedding
    heatmap_json = {}
    for wid in window_ids:
        heatmap_json[wid] = data["heatmaps"][wid]
    heatmap_json["__any__"] = data["any_heatmap"]

    # Prepare the JS data blob
    js_data = {
        "year": year,
        "numDays": num_days,
        "windowIds": window_ids,
        "dates": dates,
        "times": times,
        "monthTicks": month_ticks,
        "monthLabels": month_labels,
        "heatmaps": heatmap_json,
        "summary": summary,
        "dailyRows": daily_rows,
    }

    js_data_str = json.dumps(js_data, separators=(",", ":"))

    # Summary table rows
    summary_rows_html = ""
    for wid in window_ids:
        s = summary[wid]
        summary_rows_html += (
            f"<tr><td>{wid}</td><td>{s['days_with_sun']}</td>"
            f"<td>{s['total_hours']}</td><td>{s['earliest_date']}</td>"
            f"<td>{s['latest_date']}</td></tr>\n"
        )
    s = summary["__any__"]
    summary_rows_html += (
        f'<tr class="any-row"><td><b>Any Window</b></td><td>{s["days_with_sun"]}</td>'
        f'<td>{s["total_hours"]}</td><td>{s["earliest_date"]}</td>'
        f'<td>{s["latest_date"]}</td></tr>\n'
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Yearly Sun Hit Analysis — {year}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{
    --bg: #1a1a2e;
    --card: #16213e;
    --accent: #e94560;
    --gold: #f5a623;
    --text: #e0e0e0;
    --muted: #8899aa;
    --border: #0f3460;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    line-height: 1.6;
  }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
  h1 {{ color: var(--accent); font-size: 2em; margin-bottom: 4px; }}
  h2 {{ color: var(--gold); font-size: 1.4em; margin: 30px 0 10px; }}
  h3 {{ color: var(--text); font-size: 1.1em; margin: 20px 0 8px; }}
  .subtitle {{ color: var(--muted); font-size: 0.95em; margin-bottom: 20px; }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 24px;
  }}

  /* Navigation */
  nav {{
    background: var(--card);
    border-bottom: 2px solid var(--accent);
    padding: 12px 20px;
    position: sticky; top: 0; z-index: 100;
    display: flex; gap: 20px; flex-wrap: wrap; align-items: center;
  }}
  nav a {{
    color: var(--gold);
    text-decoration: none;
    font-weight: 600;
    font-size: 0.9em;
    transition: color .2s;
  }}
  nav a:hover {{ color: var(--accent); }}
  nav .brand {{ color: var(--accent); font-weight: 700; font-size: 1.05em; margin-right: 10px; }}

  /* Summary table */
  .summary-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.95em;
  }}
  .summary-table th {{
    background: var(--border);
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    color: var(--gold);
  }}
  .summary-table td {{
    padding: 8px 14px;
    border-bottom: 1px solid #1e3054;
  }}
  .summary-table tr:hover {{ background: rgba(233,69,96,0.08); }}
  .any-row td {{ font-weight: 600; color: var(--accent); }}

  /* Heatmap containers */
  .heatmap-container {{ margin-bottom: 16px; }}
  .heatmap-container .js-plotly-plot {{ border-radius: 6px; }}

  /* Daily table */
  .daily-wrap {{
    max-height: 700px;
    overflow: auto;
    border: 1px solid var(--border);
    border-radius: 8px;
  }}
  .daily-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82em;
    white-space: nowrap;
  }}
  .daily-table thead th {{
    position: sticky; top: 0; z-index: 2;
    background: var(--border);
    padding: 8px 10px;
    color: var(--gold);
    font-weight: 600;
    text-align: center;
  }}
  .daily-table td {{
    padding: 5px 8px;
    border-bottom: 1px solid #152238;
    text-align: center;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 0.92em;
  }}
  .daily-table td.date-cell {{
    text-align: left;
    font-weight: 600;
    color: var(--muted);
    position: sticky; left: 0; z-index: 1;
    background: var(--card);
  }}
  .daily-table tr:hover {{ background: rgba(233,69,96,0.06); }}
  .cell-hit {{
    background: rgba(245,166,35,0.15);
    color: var(--gold);
  }}
  .cell-nohit {{
    color: #445566;
  }}
  .cell-any-hit {{
    background: rgba(233,69,96,0.13);
    color: var(--accent);
  }}
  .cell-sub {{ display: block; font-size: 0.8em; color: var(--muted); }}
</style>
</head>
<body>

<nav>
  <span class="brand">☀ Sun Hit Analysis {year}</span>
  <a href="#summary">Summary</a>
  <a href="#heatmaps">Window Heatmaps</a>
  <a href="#any-heatmap">Combined Heatmap</a>
  <a href="#daily-table">Daily Table</a>
</nav>

<div class="container">

<!-- Section 1: Summary -->
<section id="summary">
<h2>Yearly Summary</h2>
<p class="subtitle">Per-window statistics for {year} ({num_days} days). Shows how many days each window delivers sun to the plant.</p>
<div class="card">
<table class="summary-table">
  <thead>
    <tr><th>Window</th><th>Days with Sun</th><th>Total Sun-Hours</th><th>Earliest Date</th><th>Latest Date</th></tr>
  </thead>
  <tbody>
    {summary_rows_html}
  </tbody>
</table>
</div>
</section>

<!-- Section 2: Per-window heatmaps -->
<section id="heatmaps">
<h2>Per-Window Heatmaps</h2>
<p class="subtitle">Each heatmap shows when direct sun reaches the plant through a specific window. Gold = hit, dark = no hit.</p>
<div id="heatmap-container"></div>
</section>

<!-- Section 3: Combined heatmap -->
<section id="any-heatmap">
<h2>Combined "Any Window" Heatmap</h2>
<p class="subtitle">Shows times when the plant receives sun through <em>any</em> window.</p>
<div id="any-heatmap-container"></div>
</section>

<!-- Section 4: Daily table -->
<section id="daily-table">
<h2>Daily Time Ranges</h2>
<p class="subtitle">Time range and total minutes of sun per window for each day. Scroll to explore.</p>
<div class="card">
<div class="daily-wrap" id="daily-table-container"></div>
</div>
</section>

</div>

<script>
const D = {js_data_str};

// ---- Heatmap rendering ----
function renderHeatmap(containerId, matrix, title, colorHit, colorOff) {{
  const z = [];
  const numTimes = D.times.length;
  for (let t = 0; t < numTimes; t++) {{
    const row = [];
    for (let d = 0; d < D.numDays; d++) {{
      row.push(matrix[d][t]);
    }}
    z.push(row);
  }}
  // Build hover text
  const hoverText = [];
  for (let t = 0; t < numTimes; t++) {{
    const row = [];
    for (let d = 0; d < D.numDays; d++) {{
      const v = matrix[d][t] ? "HIT" : "no";
      row.push(D.dates[d] + " " + D.times[t] + " — " + v);
    }}
    hoverText.push(row);
  }}
  const trace = {{
    z: z,
    type: "heatmap",
    colorscale: [[0, colorOff], [1, colorHit]],
    showscale: false,
    hovertext: hoverText,
    hoverinfo: "text",
    xgap: 0.3,
    ygap: 0.3,
  }};
  // Y-axis: show every 30 min
  const yTick = []; const yLabel = [];
  for (let i = 0; i < numTimes; i++) {{
    if (i % 6 === 0) {{ yTick.push(i); yLabel.push(D.times[i]); }}
  }}
  const layout = {{
    title: {{ text: title, font: {{ color: "#e0e0e0", size: 14 }} }},
    paper_bgcolor: "#16213e",
    plot_bgcolor: "#0f0f23",
    font: {{ color: "#8899aa" }},
    margin: {{ l: 60, r: 20, t: 40, b: 40 }},
    height: 280,
    xaxis: {{
      tickvals: D.monthTicks,
      ticktext: D.monthLabels,
      title: "",
    }},
    yaxis: {{
      tickvals: yTick,
      ticktext: yLabel,
      title: "",
      autorange: "reversed",
    }},
  }};
  const div = document.createElement("div");
  div.className = "heatmap-container";
  document.getElementById(containerId).appendChild(div);
  Plotly.newPlot(div, [trace], layout, {{ responsive: true, displayModeBar: false }});
}}

// Render per-window heatmaps
D.windowIds.forEach(wid => {{
  renderHeatmap("heatmap-container", D.heatmaps[wid], wid, "#f5a623", "#0f0f23");
}});
// Combined
renderHeatmap("any-heatmap-container", D.heatmaps["__any__"], "Any Window", "#e94560", "#0f0f23");

// ---- Daily table ----
(function() {{
  const cols = [...D.windowIds, "__any__"];
  const colLabels = [...D.windowIds, "Any"];
  let html = '<table class="daily-table"><thead><tr><th>Date</th>';
  colLabels.forEach(c => {{ html += '<th>' + c + '</th>'; }});
  html += '</tr></thead><tbody>';
  D.dailyRows.forEach(row => {{
    html += '<tr><td class="date-cell">' + row.date + '</td>';
    cols.forEach((c, ci) => {{
      const d = row[c];
      if (d) {{
        const isAny = c === "__any__";
        const cls = isAny ? "cell-any-hit" : "cell-hit";
        html += '<td class="' + cls + '">' + d.first + '–' + d.last
          + '<span class="cell-sub">' + d.minutes + ' min</span></td>';
      }} else {{
        html += '<td class="cell-nohit">—</td>';
      }}
    }});
    html += '</tr>';
  }});
  html += '</tbody></table>';
  document.getElementById("daily-table-container").innerHTML = html;
}})();
</script>
</body>
</html>"""
    return html


def main():
    year = int(sys.argv[1]) if len(sys.argv) > 1 else date.today().year
    print(f"Yearly Sun Hit Analysis — {year}")
    print(f"Config: {CONFIG_PATH}")

    config = Config.from_json_file(str(CONFIG_PATH))
    window_ids = [w.id for w in config.windows]
    num_days = 366 if calendar.isleap(year) else 365

    print(f"Location: {config.location.latitude}, {config.location.longitude}")
    print(f"Windows:  {len(window_ids)} — {', '.join(window_ids)}")
    print(f"Days:     {num_days}")
    print(f"Estimated timesteps: ~{num_days * 168} (5-min intervals, ~14h/day)")
    print()
    print("Computing...")

    data = compute_yearly_data(year, config)

    print()
    print("=== Summary ===")
    for wid in window_ids:
        s = data["summary"][wid]
        print(f"  {wid:12s}  {s['days_with_sun']:>3d} days  {s['total_hours']:>7.1f} hrs  "
              f"{s['earliest_date']}  →  {s['latest_date']}")
    s = data["summary"]["__any__"]
    print(f"  {'Any':12s}  {s['days_with_sun']:>3d} days  {s['total_hours']:>7.1f} hrs  "
          f"{s['earliest_date']}  →  {s['latest_date']}")
    print()

    html = build_html(data)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Report saved to {OUTPUT_PATH}")
    print(f"Size: {len(html) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
