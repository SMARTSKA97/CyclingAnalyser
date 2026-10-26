import streamlit as st
import pandas as pd
import numpy as np
import json
import hashlib
from datetime import date, datetime
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO

st.set_page_config(page_title="Cycling Workout Analyzer", page_icon="🚴", layout="wide")
st.title("🚴 Cycling Workout Analyzer")
st.caption("Interactive ride review · drag-select blocks · editable segmentation · configurable analytics")

ALIASES = {
    "time": ["time", "elapsed_time", "timestamp", "seconds", "timer_time"],
    "watts": ["watts", "power", "power_w", "power_watts"],
    "heartrate": ["heartrate", "heart_rate", "heart-rate", "hr", "bpm"],
    "cadence": ["cadence", "cadence_rpm", "rpm"],
    "distance": ["distance", "distance_m", "distance_km", "total_distance"],
    "altitude": ["altitude", "elevation", "elevation_m", "altitude_m"],
    "speed": ["speed", "speed_kmh", "velocity"],
}

METRIC_OPTIONS = [
    "Avg Power (W)", "Max Power (W)", "Normalized Power (W)", "IF", "TSS (approx.)",
    "Power CV (%)", "Avg HR (bpm)", "Max HR (bpm)", "HR coverage (%)", "HR zero (%)",
    "Avg Cadence (rpm)", "Avg Speed", "Distance delta", "Altitude change (m)",
    "Elevation gain estimate (m)", "Power / HR (W/bpm)",
]
DEFAULT_METRICS = [
    "Avg Power (W)", "Normalized Power (W)", "Avg HR (bpm)",
    "HR coverage (%)", "Avg Cadence (rpm)",
]
BLOCK_TYPES = [
    "Warm-up", "Recovery", "Endurance", "Tempo", "Threshold", "VO2max",
    "Sprint", "Cool-down", "Free ride", "Unclassified", "Other",
]
SETUP_STORE_PATH = Path(__file__).resolve().with_name(".cycling_analysis_setups.json")
TEMPLATE_STORE_PATH = Path(__file__).resolve().with_name(".cycling_block_templates.json")
RIDE_HISTORY_PATH = Path(__file__).resolve().with_name(".cycling_ride_history.json")

def load_setup_store():
    try:
        payload = json.loads(SETUP_STORE_PATH.read_text(encoding="utf-8"))
        setups = payload.get("setups", {}) if isinstance(payload, dict) else {}
        return {str(name): value for name, value in setups.items() if isinstance(value, dict)}
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}

def save_setup_store(setups):
    temp_path = SETUP_STORE_PATH.with_name(SETUP_STORE_PATH.name + ".tmp")
    temp_path.write_text(
        json.dumps({"version": 1, "setups": setups}, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(SETUP_STORE_PATH)

def load_block_template_store():
    try:
        payload = json.loads(TEMPLATE_STORE_PATH.read_text(encoding="utf-8"))
        templates = payload.get("templates", {}) if isinstance(payload, dict) else {}
        return {str(name): value for name, value in templates.items() if isinstance(value, list)}
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}

def save_block_template_store(templates):
    temp_path = TEMPLATE_STORE_PATH.with_name(TEMPLATE_STORE_PATH.name + ".tmp")
    temp_path.write_text(
        json.dumps({"version": 1, "templates": templates}, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(TEMPLATE_STORE_PATH)

def load_ride_history():
    try:
        payload = json.loads(RIDE_HISTORY_PATH.read_text(encoding="utf-8"))
        rides = payload.get("rides", []) if isinstance(payload, dict) else []
        return [ride for ride in rides if isinstance(ride, dict)]
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError, AttributeError):
        return []

def save_ride_history(rides):
    temp_path = RIDE_HISTORY_PATH.with_name(RIDE_HISTORY_PATH.name + ".tmp")
    temp_path.write_text(
        json.dumps({"version": 1, "rides": rides}, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(RIDE_HISTORY_PATH)

def guess_ride_date(series):
    first_value = series.dropna().astype(str).head(1)
    if first_value.empty:
        return date.today()
    value = first_value.iloc[0].strip()
    if not (len(value) >= 10 and value[4:5] == "-" and value[7:8] == "-") and not (
        len(value) >= 10 and value[2:3] == "/" and value[5:6] == "/"
    ):
        return date.today()
    parsed = pd.to_datetime(value, errors="coerce")
    return parsed.date() if pd.notna(parsed) else date.today()

def detect_columns(columns):
    normalized = {str(c).strip().lower().replace(" ", "_"): c for c in columns}
    result = {}
    for standard, aliases in ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                result[standard] = normalized[alias]
                break
    return result

def parse_time(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > .8:
        return numeric - numeric.dropna().iloc[0]
    td = pd.to_timedelta(series, errors="coerce")
    if td.notna().mean() > .8:
        return td.dt.total_seconds() - td.dropna().iloc[0]
    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().mean() > .8:
        return (dt - dt.dropna().iloc[0]).dt.total_seconds()
    raise ValueError("Could not parse time. Map a numeric elapsed-time or timestamp column.")

def build_stream(raw, selected):
    out = pd.DataFrame()
    out["time_s"] = parse_time(raw[selected["time"]])
    for standard in ["watts", "heartrate", "cadence", "distance", "altitude", "speed"]:
        col = selected.get(standard)
        if col:
            out[standard] = pd.to_numeric(raw[col], errors="coerce")
    out = out.dropna(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)
    if out.empty:
        raise ValueError("No usable rows remain after parsing time.")
    out["time_s"] -= out["time_s"].iloc[0]
    return out

def default_blocks(duration_min):
    rows = []
    if duration_min > 0:
        rows.append({"Name": "Whole ride", "Start (min)": 0.0, "End (min)": duration_min,
                     "Type": "Unclassified", "Notes": ""})
    return pd.DataFrame(rows, columns=["Name", "Start (min)", "End (min)", "Type", "Notes"])

def blocks_from_setup(setup, duration_min):
    """Load a saved block template, clipping intervals to the current ride."""
    rows = setup.get("blocks", []) if isinstance(setup, dict) else []
    valid = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            start = max(0.0, min(float(row["Start (min)"]), duration_min))
            end = max(0.0, min(float(row["End (min)"]), duration_min))
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        valid.append({
            "Name": str(row.get("Name", "")).strip() or f"Block {len(valid) + 1}",
            "Start (min)": start,
            "End (min)": end,
            "Type": str(row.get("Type", "Unclassified"))
                if str(row.get("Type", "Unclassified")) in BLOCK_TYPES else "Unclassified",
            "Notes": str(row.get("Notes", "")),
        })
    if not valid:
        return default_blocks(duration_min)
    return pd.DataFrame(valid, columns=["Name", "Start (min)", "End (min)", "Type", "Notes"])

def setup_int(setup, key, default, maximum):
    try:
        return min(maximum, max(0, int(float(setup.get(key, default)))))
    except (TypeError, ValueError):
        return default

def detect_power_intervals(df, threshold_w, smoothing_samples, minimum_seconds, merge_gap_seconds):
    # Find sustained threshold crossings and merge short interruptions.
    if "watts" not in df or df.empty:
        return []
    power = pd.to_numeric(df["watts"], errors="coerce")
    window = max(1, min(len(power), int(smoothing_samples)))
    smooth = power.rolling(
        window=window, center=True, min_periods=max(1, window // 2)
    ).mean()
    active = (smooth >= float(threshold_w)).fillna(False).to_numpy()
    times = df["time_s"].to_numpy(dtype=float)
    positive_steps = np.diff(times)
    positive_steps = positive_steps[positive_steps > 0]
    sample_period = float(np.median(positive_steps)) if len(positive_steps) else 1.0

    runs = []
    start_index = None
    for index, is_active in enumerate(active):
        if is_active and start_index is None:
            start_index = index
        elif not is_active and start_index is not None:
            runs.append([times[start_index], times[index - 1] + sample_period])
            start_index = None
    if start_index is not None:
        runs.append([times[start_index], times[-1] + sample_period])

    merged = []
    for start, end in runs:
        if merged and start - merged[-1][1] <= float(merge_gap_seconds):
            merged[-1][1] = end
        else:
            merged.append([start, end])

    ride_end = float(times[-1])
    return [
        (max(0.0, start / 60), min(ride_end, end) / 60)
        for start, end in merged
        if end - start >= float(minimum_seconds) and min(ride_end, end) > start
    ]

def normalized_power(power, window=30):
    """Approximate NP from a 30-second rolling mean; require sufficient data."""
    p = pd.to_numeric(power, errors="coerce")
    if len(p) < window or p.notna().mean() < .8:
        return np.nan
    rolling = p.rolling(window, min_periods=max(1, window // 2)).mean()
    vals = rolling.dropna()
    if vals.empty:
        return np.nan
    return float(np.mean(np.power(vals, 4)) ** .25)

def calculate_metrics(part, ftp):
    result = {}
    if part.empty:
        return result

    duration_s = max(float(part.time_s.max() - part.time_s.min()), 0)
    result["Duration (min)"] = duration_s / 60
    result["Samples"] = len(part)

    if "watts" in part:
        p = part.watts.dropna()
        if not p.empty:
            avg = float(p.mean())
            npower = normalized_power(part.watts)
            result.update({
                "Avg Power (W)": avg,
                "Max Power (W)": float(p.max()),
                "Median Power (W)": float(p.median()),
                "Power SD (W)": float(p.std()),
                "Power CV (%)": float(p.std() / avg * 100) if avg > 0 else np.nan,
                "Normalized Power (W)": npower,
                "IF": npower / ftp if ftp > 0 and np.isfinite(npower) else np.nan,
                "TSS (approx.)": 100 * duration_s / 3600 *
                    (npower / ftp) ** 2 if ftp > 0 and np.isfinite(npower) else np.nan,
            })

    if "heartrate" in part:
        raw_hr = part.heartrate
        hr = raw_hr[raw_hr > 0].dropna()
        result["HR coverage (%)"] = float(raw_hr.gt(0).sum() / len(part) * 100)
        result["HR zero (%)"] = float(raw_hr.eq(0).sum() / len(part) * 100)
        if not hr.empty:
            result["Avg HR (bpm)"] = float(hr.mean())
            result["Max HR (bpm)"] = float(hr.max())
            result["Min HR (bpm)"] = float(hr.min())

    if "cadence" in part:
        cad = part.cadence[part.cadence > 0].dropna()
        if not cad.empty:
            result["Avg Cadence (rpm)"] = float(cad.mean())
            result["Max Cadence (rpm)"] = float(cad.max())

    if "speed" in part:
        speed = part.speed.dropna()
        if not speed.empty:
            result["Avg Speed"] = float(speed.mean())
            result["Max Speed"] = float(speed.max())

    if "distance" in part:
        dist = part.distance.dropna()
        if len(dist) > 1:
            result["Distance delta"] = float(dist.iloc[-1] - dist.iloc[0])

    if "altitude" in part:
        alt = part.altitude.dropna()
        if len(alt) > 1:
            result["Altitude change (m)"] = float(alt.iloc[-1] - alt.iloc[0])
            # Positive elevation gain from sample-to-sample changes; sensitive to GPS noise.
            result["Elevation gain estimate (m)"] = float(alt.diff().clip(lower=0).sum())

    if "watts" in part and "heartrate" in part:
        paired = part[["watts", "heartrate"]].dropna()
        paired = paired[(paired.heartrate > 0) & (paired.watts >= 0)]
        if len(paired) >= 10 and paired.heartrate.mean() > 0:
            result["Power / HR (W/bpm)"] = float(paired.watts.mean() / paired.heartrate.mean())

    return result

def zone_table(power, ftp):
    if ftp <= 0:
        return pd.DataFrame()
    zones = [
        ("Z1 Recovery", 0, .55),
        ("Z2 Endurance", .55, .75),
        ("Z3 Tempo", .75, .90),
        ("Z4 Threshold", .90, 1.05),
        ("Z5 VO2max", 1.05, 1.20),
        ("Z6 Anaerobic", 1.20, 1.50),
        ("Z7 Neuromuscular", 1.50, np.inf),
    ]
    p = power.dropna()
    total = len(p)
    rows = []
    for name, lo, hi in zones:
        mask = (p >= lo * ftp) & (p < hi * ftp)
        count = int(mask.sum())
        rows.append({"Zone": name, "Power range (W)": f"{lo*ftp:.0f}–" +
                     (f"{hi*ftp:.0f}" if np.isfinite(hi) else "∞"),
                     "Samples": count, "Time (%)": count / total * 100 if total else 0})
    return pd.DataFrame(rows)

def chart_stream(df, metrics, blocks=None, selected_range=None):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    secondary_metrics = {"heartrate", "cadence", "speed"}
    for metric in metrics:
        fig.add_trace(go.Scatter(
            # Markers make individual samples selectable by Plotly's box/lasso tool.
            x=df.time_s / 60, y=df[metric], name=metric, mode="lines+markers",
            marker=dict(size=4, opacity=0.25),
            connectgaps=False, hovertemplate=f"{metric}: %{{y:.1f}}<extra></extra>"
        ), secondary_y=metric in secondary_metrics)
    if blocks is not None and not blocks.empty:
        colors = ["rgba(14,165,233,0.10)", "rgba(16,185,129,0.10)",
                  "rgba(245,158,11,0.10)", "rgba(139,92,246,0.10)"]
        for i, row in blocks.iterrows():
            try:
                a, b = float(row["Start (min)"]), float(row["End (min)"])
                if b > a:
                    fig.add_vrect(x0=a, x1=b, fillcolor=colors[i % len(colors)],
                                  line_width=0, annotation_text=str(row["Name"]),
                                  annotation_position="top left")
            except Exception:
                pass
    if selected_range:
        fig.add_vrect(x0=selected_range[0], x1=selected_range[1],
                      fillcolor="rgba(239,68,68,0.16)", line_width=1,
                      line_dash="dash", line_color="red")
    fig.update_layout(height=490, hovermode="x unified", dragmode="select",
                      xaxis_title="Elapsed time (minutes)", legend_title="Stream",
                      margin=dict(l=20, r=20, t=35, b=20))
    fig.update_yaxes(title_text="Power / altitude", secondary_y=False)
    fig.update_yaxes(title_text="HR / cadence / speed", secondary_y=True)
    return fig

def metric_chart(result, metric, title, chart_type="Bar"):
    if result.empty or metric not in result:
        st.info(f"No available values for {metric}.")
        return
    if chart_type == "Line":
        trace = go.Scatter(
            x=result["Block"], y=result[metric], mode="lines+markers", name=metric,
            hovertemplate="%{x}<br>%{y:.1f}<extra></extra>",
        )
    elif chart_type == "Area":
        trace = go.Scatter(
            x=result["Block"], y=result[metric], mode="lines+markers", fill="tozeroy",
            name=metric, hovertemplate="%{x}<br>%{y:.1f}<extra></extra>",
        )
    elif chart_type == "Points":
        trace = go.Scatter(
            x=result["Block"], y=result[metric], mode="markers", name=metric,
            marker=dict(size=11), hovertemplate="%{x}<br>%{y:.1f}<extra></extra>",
        )
    else:
        trace = go.Bar(
            x=result["Block"], y=result[metric], text=result[metric].round(1),
            textposition="outside", name=metric,
        )
    fig = go.Figure(trace)
    fig.update_layout(title=title, height=340, yaxis_title=metric,
                      barmode="group",
                      margin=dict(l=15, r=15, t=50, b=25))
    st.plotly_chart(fig, use_container_width=True)

if "ride_history" not in st.session_state:
    st.session_state.ride_history = load_ride_history()

with st.expander("Ride history and calendar progression", expanded=bool(st.session_state.ride_history)):
    st.caption("Every analyzed ride is saved locally with ride-level and block-level metrics. No raw CSV data is stored.")
    history_file = st.file_uploader(
        "Import ride history (JSON)", type=["json"], key="ride_history_upload"
    )
    if history_file is not None and st.button("Import ride history", key="import_ride_history"):
        try:
            imported = json.load(history_file)
            incoming = imported.get("rides", []) if isinstance(imported, dict) else None
            if not isinstance(incoming, list):
                raise ValueError("The JSON must contain a 'rides' list.")
            merged = {ride.get("ride_id"): ride for ride in st.session_state.ride_history
                      if ride.get("ride_id")}
            for ride in incoming:
                if isinstance(ride, dict) and ride.get("ride_id"):
                    merged[ride["ride_id"]] = ride
            if not merged:
                raise ValueError("No valid historical rides were found.")
            st.session_state.ride_history = list(merged.values())
            save_ride_history(st.session_state.ride_history)
            st.success(f"Imported and saved {len(incoming)} historical ride record(s).")
        except (json.JSONDecodeError, ValueError, AttributeError, OSError) as exc:
            st.error(f"Could not import ride history: {exc}")

    history = st.session_state.ride_history
    if history:
        history_rows = []
        for ride in history:
            metrics = ride.get("metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}
            history_rows.append({
                "Date": ride.get("date"),
                "Ride": ride.get("ride_name", ride.get("source_file", "Ride")),
                "Duration (min)": metrics.get("Duration (min)"),
                "Avg Power (W)": metrics.get("Avg Power (W)"),
                "Normalized Power (W)": metrics.get("Normalized Power (W)"),
                "Avg HR (bpm)": metrics.get("Avg HR (bpm)"),
                "Avg Cadence (rpm)": metrics.get("Avg Cadence (rpm)"),
                "Distance delta": metrics.get("Distance delta"),
                "TSS (approx.)": metrics.get("TSS (approx.)"),
            })
        history_frame = pd.DataFrame(history_rows)
        history_frame["Date"] = pd.to_datetime(history_frame["Date"], errors="coerce")
        progression_metrics = [
            metric for metric in [
                "Duration (min)", "Avg Power (W)", "Normalized Power (W)",
                "Avg HR (bpm)", "Avg Cadence (rpm)", "Distance delta", "TSS (approx.)",
            ]
            if pd.to_numeric(history_frame[metric], errors="coerce").notna().any()
        ]
        if progression_metrics:
            p1, p2 = st.columns([2, 1])
            progression_metric = p1.selectbox(
                "Progression metric", progression_metrics, key="history_metric"
            )
            progression_period = p2.selectbox(
                "Calendar view", ["Each ride", "Weekly average", "Monthly average"],
                key="history_period",
            )
            trend = history_frame[["Date", "Ride", progression_metric]].copy()
            trend["Value"] = pd.to_numeric(trend[progression_metric], errors="coerce")
            trend = trend.dropna(subset=["Date", "Value"]).sort_values("Date")
            fig = go.Figure()
            if progression_period == "Each ride":
                fig.add_trace(go.Scatter(
                    x=trend["Date"], y=trend["Value"], mode="lines+markers",
                    text=trend["Ride"], name=progression_metric,
                    hovertemplate="%{x|%Y-%m-%d}<br>%{text}<br>%{y:.1f}<extra></extra>",
                ))
            else:
                period_code = "W" if progression_period == "Weekly average" else "M"
                trend["Period"] = trend["Date"].dt.to_period(period_code).dt.start_time
                grouped = trend.groupby("Period", as_index=False).agg(
                    Value=("Value", "mean"), Rides=("Ride", "count")
                )
                fig.add_trace(go.Scatter(
                    x=grouped["Period"], y=grouped["Value"], mode="lines+markers",
                    text=grouped["Rides"], name=f"Average {progression_metric}",
                    hovertemplate="%{x|%Y-%m-%d}<br>Average: %{y:.1f}<br>Rides: %{text}<extra></extra>",
                ))
            fig.update_layout(
                title=f"Historical progression · {progression_metric}",
                xaxis_title="Ride date", yaxis_title=progression_metric,
                xaxis=dict(type="date"), height=400,
                margin=dict(l=20, r=20, t=50, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            history_frame.sort_values("Date", ascending=False),
            use_container_width=True, hide_index=True,
        )
        st.download_button(
            "Export ride history",
            json.dumps({"version": 1, "rides": history}, indent=2),
            file_name="cycling_ride_history.json",
            mime="application/json",
            key="export_ride_history",
        )
        st.download_button(
            "Download ride summary CSV",
            history_frame.to_csv(index=False).encode("utf-8"),
            file_name="cycling_ride_history_summary.csv",
            mime="text/csv",
            key="export_ride_history_csv",
        )
        ride_labels = [
            f"{ride.get('date', 'Unknown date')} · {ride.get('ride_name', ride.get('source_file', 'Ride'))} · #{i + 1}"
            for i, ride in enumerate(history)
        ]
        selected_history_ride = st.selectbox(
            "Inspect a saved ride's block results", ride_labels, key="history_ride_detail"
        )
        history_index = ride_labels.index(selected_history_ride)
        saved_blocks = history[history_index].get("blocks", [])
        if isinstance(saved_blocks, list) and saved_blocks:
            st.dataframe(pd.DataFrame(saved_blocks), use_container_width=True, hide_index=True)
    else:
        st.info("Analyze a ride to start building calendar history, or import a saved history JSON.")

# --------------------- Upload and map ---------------------
uploaded = st.file_uploader("Upload cycling stream CSV", type=["csv"])
if uploaded is None:
    st.info("Start by uploading your ride CSV. The file is processed in this session.")
    st.stop()

try:
    raw = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"CSV load failed: {exc}")
    st.stop()

if "saved_setups" not in st.session_state:
    st.session_state.saved_setups = load_setup_store()
elif st.session_state.saved_setups and not SETUP_STORE_PATH.exists():
    # Preserve setups created by the earlier session-only version during hot reload.
    try:
        save_setup_store(st.session_state.saved_setups)
    except OSError:
        pass
st.session_state.setdefault("setup_revision", 0)
st.session_state.setdefault("active_setup", None)
st.session_state.setdefault("profile_blocks_pending", False)
if "saved_block_templates" not in st.session_state:
    st.session_state.saved_block_templates = load_block_template_store()

with st.expander("Reusable analysis setups", expanded=False):
    st.caption("Setups are saved locally beside this app and reload automatically after a restart. Export/import JSON to move them to another computer. Ride data is not included.")
    setup_file = st.file_uploader("Import setup library (JSON)", type=["json"], key="setup_library_upload")
    if setup_file is not None and st.button("Import setup library", key="import_setup_library"):
        try:
            imported = json.load(setup_file)
            incoming = imported.get("setups", imported) if isinstance(imported, dict) else None
            if not isinstance(incoming, dict):
                raise ValueError("The JSON must contain a 'setups' object.")
            valid_incoming = {
                str(name): setup for name, setup in incoming.items()
                if isinstance(setup, dict)
            }
            if not valid_incoming:
                raise ValueError("No valid named setups were found.")
            st.session_state.saved_setups.update(valid_incoming)
            save_setup_store(st.session_state.saved_setups)
            st.success(f"Imported and saved {len(valid_incoming)} setup(s).")
        except (json.JSONDecodeError, ValueError, AttributeError, OSError) as exc:
            st.error(f"Could not import setup library: {exc}")

    saved_setup_names = list(st.session_state.saved_setups)
    if saved_setup_names:
        selected_setup_name = st.selectbox("Saved setup", saved_setup_names, key="selected_setup_name")
        if st.button("Load selected setup", key="load_selected_setup"):
            st.session_state.active_setup = st.session_state.saved_setups[selected_setup_name]
            st.session_state.setup_revision += 1
            st.session_state.profile_blocks_pending = True
            st.success(f"Loaded “{selected_setup_name}”. Review the mappings and block template below.")
        setup_library = json.dumps({"version": 1, "setups": st.session_state.saved_setups}, indent=2)
        st.download_button(
            "Export setup library",
            setup_library,
            file_name="cycling_analysis_setups.json",
            mime="application/json",
            key="export_setup_library",
        )
    else:
        st.info("Save a setup below or import a setup library to get started.")

guesses = detect_columns(raw.columns)
cols = list(raw.columns)
active_setup = st.session_state.active_setup or {}
saved_mapping = active_setup.get("mapping", {}) if isinstance(active_setup, dict) else {}
if not isinstance(saved_mapping, dict):
    saved_mapping = {}
st.subheader("Column mapping")
st.caption("Detected mappings are suggestions. Adjust them if the CSV headers differ.")
mapping = {}
mapping_fallbacks = []
layout = st.columns(3)
for i, standard in enumerate(["time", "watts", "heartrate", "cadence", "distance", "altitude", "speed"]):
    with layout[i % 3]:
        options = ["— Not mapped —"] + cols
        preferred = saved_mapping.get(standard) or guesses.get(standard)
        if preferred not in options:
            if saved_mapping.get(standard):
                mapping_fallbacks.append(standard)
            preferred = guesses.get(standard)
        idx = options.index(preferred) if preferred in options else 0
        chosen = st.selectbox(
            f"{standard} column", options, index=idx,
            key=f"map_{standard}_{st.session_state.setup_revision}"
        )
        mapping[standard] = None if chosen == options[0] else chosen
if mapping_fallbacks:
    st.warning("The loaded setup used columns missing from this CSV for: " +
               ", ".join(mapping_fallbacks) + ". Confirm the detected mappings.")
if mapping["time"] is None:
    st.warning("Please map the time column.")
    st.stop()

try:
    df = build_stream(raw, mapping)
except Exception as exc:
    st.error(str(exc))
    st.stop()

duration_min = float(df.time_s.max() / 60)
if duration_min <= 0:
    st.error("Ride duration is zero. Check your time-column mapping.")
    st.stop()

# Persistent state keyed to uploaded file signature
file_key = f"{uploaded.name}_{uploaded.size}"
if st.session_state.get("file_key") != file_key:
    st.session_state.file_key = file_key
    st.session_state.blocks = blocks_from_setup(active_setup, duration_min) if active_setup else default_blocks(duration_min)
    st.session_state.selected_range = None
    st.session_state.analysis = None
    st.session_state.block_editor_revision = 0
    st.session_state.manual_input_revision = 0
    st.session_state.next_manual_range = (0.0, duration_min)
if st.session_state.profile_blocks_pending:
    st.session_state.blocks = blocks_from_setup(active_setup, duration_min)
    st.session_state.block_editor_revision += 1
    st.session_state.manual_input_revision += 1
    st.session_state.next_manual_range = (0.0, duration_min)
    st.session_state.analysis = None
    st.session_state.profile_blocks_pending = False
st.session_state.setdefault("block_editor_revision", 0)
st.session_state.setdefault("manual_input_revision", 0)
st.session_state.setdefault("next_manual_range", (0.0, duration_min))

st.subheader("Ride overview")
default_ride_name = uploaded.name.rsplit(".", 1)[0]
ride_name = st.text_input("Ride name (used for exports)", value=default_ride_name, key=f"ride_name_{file_key}")
ride_date = st.date_input(
    "Ride date", value=guess_ride_date(raw[mapping["time"]]),
    key=f"ride_date_{file_key}",
)
valid_hr = df.loc[df.get("heartrate", pd.Series(index=df.index, dtype=float)) > 0, "heartrate"] if "heartrate" in df else pd.Series(dtype=float)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ride duration", f"{duration_min:.2f} min")
c2.metric("Samples", f"{len(df):,}")
c3.metric("Mean power", f"{df.watts.mean():.1f} W" if "watts" in df else "—")
c4.metric("Mean HR", f"{valid_hr.mean():.1f} bpm" if len(valid_hr) else "—")

st.subheader("Create blocks from the timeline")
st.caption("Drag a rectangle across the plotted line (or use lasso select). Releasing the mouse creates a block immediately; rename it in the block table. The box must cover plotted samples.")
available_metrics = [x for x in ["watts", "heartrate", "cadence", "speed", "altitude"] if x in df]
default_metrics = [x for x in ["watts", "heartrate"] if x in available_metrics]
chart_metrics = st.multiselect("Streams", available_metrics, default=default_metrics)
if chart_metrics:
    timeline = chart_stream(df, chart_metrics, st.session_state.blocks)
    selection = st.plotly_chart(
        timeline, use_container_width=True,
        key=f"ride_timeline_{st.session_state.block_editor_revision}",
        on_select="rerun", selection_mode=("box", "lasso")
    )
    points = selection.selection.points if selection and hasattr(selection, "selection") else []
    if points:
        xs = [float(p["x"]) for p in points if p.get("x") is not None]
        if xs:
            start_min = max(0.0, min(xs))
            end_min = min(duration_min, max(xs))
            if end_min <= start_min:
                sample_step = float(df.time_s.diff().median() / 60)
                end_min = min(duration_min, start_min + max(sample_step, 0.1))
            signature = (round(start_min, 3), round(end_min, 3), len(points))
            if end_min > start_min and signature != st.session_state.get("last_drag_signature"):
                st.session_state.selected_range = [start_min, end_min]
                st.session_state.last_drag_signature = signature
                row = {
                    "Name": f"Block {len(st.session_state.blocks) + 1}",
                    "Start (min)": start_min,
                    "End (min)": end_min,
                    "Type": "Unclassified",
                    "Notes": "",
                }
                st.session_state.blocks = pd.concat(
                    [st.session_state.blocks, pd.DataFrame([row])], ignore_index=True
                )
                st.session_state.block_editor_revision += 1
                st.session_state.analysis = None
                st.toast(f"Created {row['Name']} · {start_min:.2f}–{end_min:.2f} min. Rename or classify it in the block table.")
                st.rerun()
else:
    st.info("Choose one or more streams to display.")

with st.expander("Create a block by entering its time range"):
    default_manual_range = st.session_state.next_manual_range
    manual_start, manual_end = st.slider(
        "Block range (minutes)", min_value=0.0, max_value=float(duration_min),
        value=(float(default_manual_range[0]), float(default_manual_range[1])),
        step=0.1, key=f"new_block_range_{file_key}_{st.session_state.manual_input_revision}"
    )
    manual_name = st.text_input(
        "Block name", value=f"Block {len(st.session_state.blocks) + 1}",
        key=f"new_block_name_{file_key}_{st.session_state.manual_input_revision}"
    )
    if st.button("Add this range as a block", key="add_manual_range"):
        if manual_end <= manual_start:
            st.error("End time must be after start time.")
        else:
            row = {"Name": manual_name, "Start (min)": manual_start,
                   "End (min)": manual_end, "Type": "Unclassified", "Notes": ""}
            st.session_state.blocks = pd.concat(
                [st.session_state.blocks, pd.DataFrame([row])], ignore_index=True
            )
            # Give each addition fresh widget keys so the previous name/range
            # does not silently carry over to the next block.
            next_start = min(manual_end, max(0.0, duration_min - 0.1))
            st.session_state.next_manual_range = (next_start, float(duration_min))
            st.session_state.manual_input_revision += 1
            st.session_state.block_editor_revision += 1
            st.session_state.analysis = None
            st.rerun()

if st.session_state.selected_range:
    a, b = st.session_state.selected_range
    st.success(f"Last dragged block: {a:.2f}–{b:.2f} min ({b-a:.2f} min). Rename or classify it in the block table below.")

st.subheader("Workout blocks · editable and correctable")
st.caption("Edit cells and add/remove rows, then click **Save block edits**. Save before using other controls so the latest row is committed.")
if st.session_state.blocks.empty:
    st.session_state.blocks = pd.DataFrame(columns=["Name", "Start (min)", "End (min)", "Type", "Notes"])

with st.form("block_editor_form"):
    edited = st.data_editor(
        st.session_state.blocks, num_rows="dynamic", use_container_width=True,
        key=f"block_editor_v3_{st.session_state.block_editor_revision}",
        column_config={
            "Name": st.column_config.TextColumn("Block name", required=True),
            "Start (min)": st.column_config.NumberColumn("Start (min)", min_value=0.0, max_value=duration_min, step=0.1),
            "End (min)": st.column_config.NumberColumn("End (min)", min_value=0.0, max_value=duration_min, step=0.1),
            "Type": st.column_config.SelectboxColumn("Type", options=BLOCK_TYPES),
            "Notes": st.column_config.TextColumn("Notes")
        }
    )
    save_blocks = st.form_submit_button("Save block edits", type="primary")

if save_blocks:
    st.session_state.blocks = edited.copy().reset_index(drop=True)
    st.session_state.analysis = None
    st.success("Block edits saved.")
edited = st.session_state.blocks.copy()

if not edited.empty:
    block_options = [f"{i + 1}. {str(row['Name'])}" for i, (_, row) in enumerate(edited.iterrows())]
    selected_block_label = st.selectbox("Adjust a block by dragging its time range", block_options)
    selected_block_index = block_options.index(selected_block_label)
    selected_block = edited.iloc[selected_block_index]
    start_drag, end_drag = st.slider(
        "Drag the handles to set start and end (minutes)",
        min_value=0.0, max_value=float(duration_min),
        value=(float(selected_block["Start (min)"]), float(selected_block["End (min)"])),
        step=0.1, key=f"block_range_{st.session_state.block_editor_revision}_{selected_block_index}"
    )
    if st.button("Apply range to selected block"):
        st.session_state.blocks.loc[selected_block_index, "Start (min)"] = start_drag
        st.session_state.blocks.loc[selected_block_index, "End (min)"] = end_drag
        st.session_state.block_editor_revision += 1
        st.session_state.analysis = None
        st.rerun()

valid_blocks = edited.copy()
if not valid_blocks.empty:
    for col in ["Start (min)", "End (min)"]:
        valid_blocks[col] = pd.to_numeric(valid_blocks[col], errors="coerce")
    valid_blocks = valid_blocks.dropna(subset=["Start (min)", "End (min)"])
    valid_blocks = valid_blocks[
        (valid_blocks["Start (min)"] >= 0) &
        (valid_blocks["End (min)"] > valid_blocks["Start (min)"]) &
        (valid_blocks["End (min)"] <= duration_min)
    ]

with st.expander("Block editing helpers"):
    h1, h2, h3 = st.columns(3)
    with h1:
        interval_min = st.number_input("Equal block length (min)", min_value=0.5, value=10.0, step=0.5)
        if st.button("Generate equal blocks"):
            starts = np.arange(0, duration_min, interval_min)
            st.session_state.blocks = pd.DataFrame([
                {"Name": f"Block {i+1}", "Start (min)": float(s),
                 "End (min)": float(min(s + interval_min, duration_min)),
                 "Type": "Unclassified", "Notes": ""}
                for i, s in enumerate(starts)
            ])
            st.session_state.block_editor_revision += 1
            st.session_state.analysis = None
            st.rerun()
    with h2:
        if st.button("Reset to whole ride"):
            st.session_state.blocks = default_blocks(duration_min)
            st.session_state.block_editor_revision += 1
            st.session_state.analysis = None
            st.rerun()
    with h3:
        st.write("Overlapping blocks are allowed for comparison. Invalid time ranges are excluded from analysis.")

with st.expander("Reusable block templates"):
    st.caption("Templates contain block names, ranges, types, and notes. Loaded ranges are clipped to this ride.")
    template_name = st.text_input("Template name", key="block_template_name")
    if st.button("Save current blocks as template", key="save_block_template"):
        clean_template_name = template_name.strip()
        if not clean_template_name:
            st.error("Enter a name for the block template.")
        elif valid_blocks.empty:
            st.error("Add at least one valid block before saving a template.")
        else:
            template_rows = json.loads(valid_blocks.to_json(orient="records"))
            st.session_state.saved_block_templates[clean_template_name] = template_rows
            try:
                save_block_template_store(st.session_state.saved_block_templates)
                st.success(f"Saved template “{clean_template_name}”.")
            except OSError as exc:
                st.error(f"Template is available this session but could not be saved locally: {exc}")

    template_names = list(st.session_state.saved_block_templates)
    if template_names:
        selected_template_name = st.selectbox(
            "Saved block template", template_names, key="selected_block_template"
        )
        if st.button("Load block template", key="load_block_template"):
            template_blocks = blocks_from_setup(
                {"blocks": st.session_state.saved_block_templates[selected_template_name]},
                duration_min,
            )
            st.session_state.blocks = template_blocks
            st.session_state.block_editor_revision += 1
            st.session_state.manual_input_revision += 1
            st.session_state.analysis = None
            st.rerun()
        template_library = json.dumps(
            {"version": 1, "templates": st.session_state.saved_block_templates}, indent=2
        )
        st.download_button(
            "Export block templates",
            template_library,
            file_name="cycling_block_templates.json",
            mime="application/json",
            key="export_block_templates",
        )
    else:
        st.info("Save a block template or import a template library to get started.")

    template_file = st.file_uploader(
        "Import block templates (JSON)", type=["json"], key="block_template_upload"
    )
    if template_file is not None and st.button("Import block templates", key="import_block_templates"):
        try:
            imported = json.load(template_file)
            incoming = imported.get("templates", imported) if isinstance(imported, dict) else None
            valid_templates = {
                str(name): [row for row in rows if isinstance(row, dict)]
                for name, rows in incoming.items()
                if isinstance(incoming, dict) and isinstance(rows, list)
            } if isinstance(incoming, dict) else {}
            if not valid_templates:
                raise ValueError("No valid named block templates were found.")
            st.session_state.saved_block_templates.update(valid_templates)
            save_block_template_store(st.session_state.saved_block_templates)
            st.success(f"Imported and saved {len(valid_templates)} block template(s).")
        except (json.JSONDecodeError, ValueError, AttributeError, OSError) as exc:
            st.error(f"Could not import block templates: {exc}")

st.subheader("Analysis configuration")
with st.expander("FTP, HR zones, and analysis options", expanded=True):
    c1, c2, c3 = st.columns(3)
    ftp_default = setup_int(active_setup, "ftp", 159, 2000)
    max_hr_default = setup_int(active_setup, "max_hr", 0, 250)
    resting_hr_default = setup_int(active_setup, "resting_hr", 0, 150)
    ftp = c1.number_input(
        "FTP (W)", min_value=0, max_value=2000, value=ftp_default, step=1,
        key=f"ftp_{st.session_state.setup_revision}"
    )
    max_hr = c2.number_input(
        "Max HR (bpm; optional)", min_value=0, max_value=250, value=max_hr_default, step=1,
        key=f"max_hr_{st.session_state.setup_revision}"
    )
    resting_hr = c3.number_input(
        "Resting HR (bpm; optional)", min_value=0, max_value=150, value=resting_hr_default, step=1,
        key=f"resting_hr_{st.session_state.setup_revision}"
    )
    saved_metric_defaults = active_setup.get("metrics", DEFAULT_METRICS)
    if not isinstance(saved_metric_defaults, list):
        saved_metric_defaults = DEFAULT_METRICS
    saved_metric_defaults = [m for m in saved_metric_defaults if m in METRIC_OPTIONS]
    chosen_metrics = st.multiselect(
        "Metrics to include in block comparison", METRIC_OPTIONS,
        default=saved_metric_defaults,
        key=f"analysis_metrics_{st.session_state.setup_revision}"
    )

with st.expander("Detect intervals from power"):
    if "watts" not in df:
        st.info("Map a power column to detect intervals.")
    else:
        st.caption("This finds sustained power above a threshold. It proposes time ranges only; review and edit them before analysis.")
        threshold_default = max(1, min(2000, int(round(ftp * 0.8)))) if ftp else 150
        d1, d2, d3, d4 = st.columns(4)
        threshold_w = d1.number_input(
            "Power threshold (W)", min_value=1, max_value=2000,
            value=threshold_default, step=5, key=f"detect_threshold_{st.session_state.setup_revision}"
        )
        smoothing_seconds = d2.number_input(
            "Smooth over (seconds)", min_value=1, max_value=180,
            value=15, step=1, key="detect_smoothing_seconds"
        )
        minimum_interval_seconds = d3.number_input(
            "Minimum interval (seconds)", min_value=15, max_value=7200,
            value=120, step=15, key="detect_minimum_seconds"
        )
        merge_gap_seconds = d4.number_input(
            "Merge gaps up to (seconds)", min_value=0, max_value=300,
            value=10, step=1, key="detect_merge_gap_seconds"
        )
        detection_mode = st.radio(
            "When intervals are found",
            ["Replace current blocks", "Append to current blocks"],
            horizontal=True,
            key="detect_mode",
        )
        if st.button("Detect intervals", key="detect_power_intervals"):
            steps = np.diff(df.time_s.to_numpy(dtype=float))
            steps = steps[steps > 0]
            sample_period = float(np.median(steps)) if len(steps) else 1.0
            smoothing_samples = max(1, int(round(smoothing_seconds / sample_period)))
            intervals = detect_power_intervals(
                df, threshold_w, smoothing_samples,
                minimum_interval_seconds, merge_gap_seconds,
            )
            if not intervals:
                st.warning("No intervals met these settings. Lower the threshold or minimum duration and try again.")
            else:
                detected = pd.DataFrame([
                    {
                        "Name": f"Detected {i + 1}",
                        "Start (min)": start,
                        "End (min)": end,
                        "Type": "Unclassified",
                        "Notes": f"Auto-detected at ≥{threshold_w} W",
                    }
                    for i, (start, end) in enumerate(intervals)
                ], columns=["Name", "Start (min)", "End (min)", "Type", "Notes"])
                if detection_mode == "Append to current blocks":
                    st.session_state.blocks = pd.concat(
                        [st.session_state.blocks, detected], ignore_index=True
                    )
                else:
                    st.session_state.blocks = detected
                st.session_state.block_editor_revision += 1
                st.session_state.manual_input_revision += 1
                st.session_state.analysis = None
                st.rerun()

with st.expander("Save current setup"):
    setup_name = st.text_input("Setup name", key="setup_name_input")
    if st.button("Save setup", key="save_current_setup"):
        clean_name = setup_name.strip()
        if not clean_name:
            st.error("Enter a name for this setup.")
        else:
            profile = {
                "mapping": mapping,
                "ftp": int(ftp),
                "max_hr": int(max_hr),
                "resting_hr": int(resting_hr),
                "metrics": chosen_metrics,
                "blocks": json.loads(st.session_state.blocks.to_json(orient="records")),
            }
            st.session_state.saved_setups[clean_name] = profile
            st.session_state.active_setup = profile
            try:
                save_setup_store(st.session_state.saved_setups)
                st.rerun()
            except OSError as exc:
                st.error(f"Setup is available this session but could not be saved locally: {exc}")

if st.button("Analyze blocks", type="primary"):
    if valid_blocks.empty:
        st.error("No valid blocks. Add or correct a block first.")
    else:
        records = []
        for block_no, (_, block) in enumerate(valid_blocks.iterrows(), start=1):
            start_min, end_min = float(block["Start (min)"]), float(block["End (min)"])
            part = df[(df.time_s >= start_min * 60) &
                      (df.time_s <= end_min * 60)]
            metrics = calculate_metrics(part, ftp)
            block_name = str(block.get("Name", "")).strip() or f"Block {block_no}"
            chart_label = f"{block_name} · #{block_no} ({start_min:.1f}–{end_min:.1f} min)"
            records.append({"Block": chart_label, "Block name": block_name,
                            "Type": block.get("Type", "Unclassified"),
                            "Start (min)": start_min, "End (min)": end_min,
                            **metrics})
        st.session_state.analysis = pd.DataFrame(records)
        st.session_state.analyzed_blocks = valid_blocks.copy()
        st.session_state.metric_selection = chosen_metrics
        source_id = hashlib.sha256(uploaded.getvalue()).hexdigest()
        ride_metrics = calculate_metrics(df, ftp)
        safe_metrics = {}
        for metric_name, metric_value in ride_metrics.items():
            if pd.isna(metric_value):
                safe_metrics[metric_name] = None
            elif isinstance(metric_value, (np.integer, np.floating)):
                safe_metrics[metric_name] = metric_value.item()
            else:
                safe_metrics[metric_name] = metric_value
        history_entry = {
            "ride_id": source_id,
            "date": ride_date.isoformat(),
            "ride_name": ride_name.strip() or uploaded.name.rsplit(".", 1)[0],
            "source_file": uploaded.name,
            "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "ftp": int(ftp),
            "metrics": safe_metrics,
            "blocks": json.loads(st.session_state.analysis.to_json(orient="records")),
        }
        history_by_id = {
            ride.get("ride_id"): ride for ride in st.session_state.ride_history
            if ride.get("ride_id")
        }
        history_by_id[source_id] = history_entry
        st.session_state.ride_history = list(history_by_id.values())
        try:
            save_ride_history(st.session_state.ride_history)
            st.rerun()
        except OSError as exc:
            st.error(f"Analysis is ready, but ride history could not be saved locally: {exc}")

result = st.session_state.get("analysis")
if result is not None and "Block name" not in result.columns:
    # Older session results used block names as chart categories, so duplicate names
    # could be combined visually even though each row's metric was calculated alone.
    st.session_state.analysis = None
    result = None
    st.info("Block chart labels were updated to distinguish intervals. Click “Analyze blocks” to refresh the charts.")
if result is not None and not result.empty:
    st.subheader("Block analysis · visual first")
    st.caption("These comparisons use only valid blocks. Metrics that cannot be calculated are left blank.")
    shown = [m for m in st.session_state.get("metric_selection", []) if m in result.columns]
    chartable = [m for m in shown if pd.to_numeric(result[m], errors="coerce").notna().any()]
    if chartable:
        metric_col1, metric_col2 = st.columns([2, 1])
        selected_metric = metric_col1.selectbox("Choose a metric", chartable, index=0)
        selected_chart_type = metric_col2.selectbox(
            "Chart style", ["Bar", "Line", "Area", "Points"], key="block_metric_chart_style"
        )
        metric_chart(
            result, selected_metric,
            f"Block comparison · {selected_metric}", selected_chart_type,
        )
        if len(chartable) > 1:
            with st.expander("Compare several metrics"):
                normalized = result[["Block"] + chartable].copy()
                for m in chartable:
                    vals = pd.to_numeric(normalized[m], errors="coerce")
                    lo, hi = vals.min(), vals.max()
                    normalized[m] = (vals - lo) / (hi - lo) if pd.notna(lo) and hi != lo else 0
                fig = go.Figure()
                for m in chartable:
                    fig.add_trace(go.Scatter(x=normalized.Block, y=normalized[m],
                                             mode="lines+markers", name=m))
                fig.update_layout(title="Normalized metric profiles (0–1 within each metric)",
                                  height=400, yaxis_title="Relative scale")
                st.plotly_chart(fig, use_container_width=True)

    # Overlay the same stream across selected blocks, optionally aligned to block start.
    st.subheader("Block time-series overlays")
    overlay_metrics = [metric for metric in ["watts", "heartrate", "cadence", "speed", "altitude"] if metric in df]
    if overlay_metrics and not valid_blocks.empty:
        overlay_metric = st.selectbox(
            "Stream to overlay", overlay_metrics, key="block_overlay_metric"
        )
        overlay_labels = [
            f"{str(block['Name']).strip() or f'Block {i + 1}'} · #{i + 1} "
            f"({float(block['Start (min)']):.1f}–{float(block['End (min)']):.1f} min)"
            for i, (_, block) in enumerate(valid_blocks.iterrows())
        ]
        selected_overlays = st.multiselect(
            "Blocks to compare",
            overlay_labels,
            default=overlay_labels[:min(4, len(overlay_labels))],
            key="block_overlay_selection",
        )
        overlay_alignment = st.radio(
            "Time axis",
            ["Ride time", "Align block starts"],
            horizontal=True,
            key="block_overlay_alignment",
        )
        fig = go.Figure()
        for block_label in selected_overlays:
            block_index = overlay_labels.index(block_label)
            block = valid_blocks.iloc[block_index]
            start_s = float(block["Start (min)"]) * 60
            end_s = float(block["End (min)"]) * 60
            part = df[(df.time_s >= start_s) & (df.time_s <= end_s)]
            x = (part.time_s - start_s) / 60 if overlay_alignment == "Align block starts" else part.time_s / 60
            fig.add_trace(go.Scatter(
                x=x, y=part[overlay_metric], mode="lines",
                name=block_label, connectgaps=False,
                hovertemplate=f"{overlay_metric}: %{{y:.1f}}<extra>%{{fullData.name}}</extra>",
            ))
        fig.update_layout(
            title=f"{overlay_metric} profiles by block",
            xaxis_title="Minutes from block start" if overlay_alignment == "Align block starts" else "Ride time (minutes)",
            yaxis_title=overlay_metric, height=420, hovermode="x unified",
            margin=dict(l=20, r=20, t=50, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Add valid blocks and map at least one stream to compare block time series.")

    # Zone distribution for entire ride and each block
    if "watts" in df and ftp > 0:
        st.subheader("Power-zone distribution")
        zone_blocks = list(valid_blocks.iterrows())
        zone_labels = [
            f"{str(block['Name']).strip() or f'Block {i + 1}'} · #{i + 1} "
            f"({float(block['Start (min)']):.1f}–{float(block['End (min)']):.1f} min)"
            for i, (_, block) in enumerate(zone_blocks)
        ]
        zone_scope = st.selectbox(
            "Zone analysis scope",
            ["Whole ride", "One block", "All blocks"],
            key="zone_analysis_scope",
        )
        if zone_scope == "All blocks":
            comparison_rows = []
            for label, (_, block) in zip(zone_labels, zone_blocks):
                zone_part = df[
                    (df.time_s >= float(block["Start (min)"]) * 60) &
                    (df.time_s <= float(block["End (min)"]) * 60)
                ]
                block_zones = zone_table(zone_part.watts, ftp)
                if not block_zones.empty:
                    block_zones.insert(0, "Block", label)
                    comparison_rows.append(block_zones)
            if comparison_rows:
                comparison = pd.concat(comparison_rows, ignore_index=True)
                fig = go.Figure()
                for zone_name in comparison["Zone"].drop_duplicates():
                    zone_rows = comparison[comparison["Zone"] == zone_name]
                    fig.add_trace(go.Bar(
                        x=zone_rows["Block"], y=zone_rows["Time (%)"],
                        name=zone_name, hovertemplate="%{x}<br>%{fullData.name}: %{y:.1f}%<extra></extra>",
                    ))
                fig.update_layout(
                    title="Power-zone distribution by block",
                    barmode="stack", height=430,
                    yaxis=dict(title="Share of recorded power samples (%)", range=[0, 100]),
                    xaxis_title="Block",
                    margin=dict(l=20, r=20, t=50, b=30),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(comparison, use_container_width=True, hide_index=True)
        else:
            if zone_scope == "Whole ride":
                zone_power = df.watts
                scope_label = "Whole ride"
            else:
                selected_zone_block = st.selectbox(
                    "Block", zone_labels, key="single_block_zone_choice"
                )
                selected_index = zone_labels.index(selected_zone_block)
                selected_block = zone_blocks[selected_index][1]
                zone_part = df[
                    (df.time_s >= selected_block["Start (min)"] * 60) &
                    (df.time_s <= selected_block["End (min)"] * 60)
                ]
                zone_power = zone_part.watts
                scope_label = selected_zone_block
            zones = zone_table(zone_power, ftp)
            if not zones.empty:
                fig = go.Figure(go.Bar(
                    x=zones["Zone"], y=zones["Time (%)"],
                    text=zones["Time (%)"].round(1).astype(str) + "%",
                    textposition="outside",
                ))
                fig.update_layout(title=f"Time in power zones · {scope_label}", height=350,
                                  yaxis_title="Share of recorded power samples (%)")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(zones, use_container_width=True, hide_index=True)

    # Power vs HR response chart
    if "watts" in df and "heartrate" in df:
        st.subheader("Power vs heart-rate response")
        resp = df[(df.watts.notna()) & (df.heartrate > 0)].copy()
        if not resp.empty:
            fig = go.Figure(go.Scatter(x=resp.watts, y=resp.heartrate, mode="markers",
                                       marker=dict(size=4, opacity=.45),
                                       hovertemplate="Power %{x:.0f} W<br>HR %{y:.0f} bpm<extra></extra>"))
            fig.update_layout(title="Power–HR relationship", xaxis_title="Power (W)",
                              yaxis_title="Heart rate (bpm)", height=390)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("This scatter shows the recorded relationship; it is not, by itself, a diagnosis of aerobic decoupling.")

    with st.expander("Detailed numeric results"):
        st.dataframe(result, use_container_width=True, hide_index=True)

    safe_ride_name = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in ride_name).strip().replace(" ", "_") or "cycling_ride"
    csv_data = result.to_csv(index=False).encode("utf-8")
    st.download_button("Download block metrics CSV", csv_data, f"{safe_ride_name}_block_analysis.csv", "text/csv")
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        result.to_excel(writer, index=False, sheet_name="Block Analysis")
        valid_blocks.to_excel(writer, index=False, sheet_name="Workout Blocks")
    st.download_button("Download Excel workbook", excel_buffer.getvalue(),
                       f"{safe_ride_name}_block_analysis.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with st.expander("Data preview"):
    st.dataframe(df.head(100), use_container_width=True)
