import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO

st.set_page_config(page_title="Cycling Workout Analyzer", layout="wide")
st.title("🚴 Cycling Workout Analyzer")
st.caption("Explore ride streams, define workout blocks, and inspect power / heart-rate behavior.")

ALIASES = {
    "time": ["time", "elapsed_time", "timestamp", "seconds", "timer_time"],
    "watts": ["watts", "power", "power_w", "power_watts"],
    "heartrate": ["heartrate", "heart_rate", "heart-rate", "hr", "bpm"],
    "cadence": ["cadence", "cadence_rpm", "rpm"],
    "distance": ["distance", "distance_m", "distance_km", "total_distance"],
    "altitude": ["altitude", "elevation", "elevation_m", "altitude_m"],
    "speed": ["speed", "speed_kmh", "velocity"],
}

def guess_columns(columns):
    normalized = {str(c).strip().lower().replace(" ", "_"): c for c in columns}
    found = {}
    for standard, names in ALIASES.items():
        for name in names:
            if name in normalized:
                found[standard] = normalized[name]
                break
    return found

def parse_time(series):
    """Convert elapsed seconds, numeric time, or HH:MM:SS strings to seconds."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.8:
        return numeric - numeric.dropna().iloc[0]
    parsed = pd.to_timedelta(series, errors="coerce")
    if parsed.notna().mean() > 0.8:
        return parsed.dt.total_seconds() - parsed.dropna().iloc[0]
    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().mean() > 0.8:
        return (dt - dt.dropna().iloc[0]).dt.total_seconds()
    raise ValueError("Couldn't interpret the time column. Choose a valid elapsed-time column.")

def summarize(df, blocks, ftp):
    rows = []
    for _, b in blocks.iterrows():
        start, end = float(b["Start (min)"]) * 60, float(b["End (min)"]) * 60
        part = df[(df.time_s >= start) & (df.time_s <= end)]
        if part.empty:
            continue
        row = {"Block": b["Name"], "Start (min)": start / 60, "End (min)": end / 60,
               "Duration (min)": (end-start) / 60, "Samples": len(part)}
        for col, label in [("watts", "Power"), ("heartrate", "HR"), ("cadence", "Cadence"),
                           ("speed", "Speed"), ("altitude", "Altitude")]:
            if col not in part:
                continue
            vals = part[col].dropna()
            if col in ("heartrate", "cadence", "speed"):
                vals = vals[vals > 0]
            if vals.empty:
                continue
            row[f"{label} mean"] = vals.mean()
            row[f"{label} max"] = vals.max()
            if col == "watts":
                row["Power median"] = vals.median()
                row["Power variability (SD)"] = vals.std()
                row["Power % FTP"] = vals.mean() / ftp * 100 if ftp > 0 else np.nan
            if col == "heartrate":
                raw = part[col]
                row["HR coverage (%)"] = (raw.gt(0).sum() / len(part)) * 100
                row["HR zero (%)"] = (raw.eq(0).sum() / len(part)) * 100
            if col == "altitude":
                row["Altitude change"] = vals.iloc[-1] - vals.iloc[0]
        if "distance" in part:
            d = part.distance.dropna()
            if len(d) > 1:
                row["Distance change"] = d.iloc[-1] - d.iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)

uploaded = st.file_uploader("Upload your cycling stream CSV", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to begin. The file stays in this app session.")
    st.stop()

try:
    raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Couldn't read CSV: {e}")
    st.stop()

st.subheader("1 · Map your CSV columns")
guesses = guess_columns(raw.columns)
columns = list(raw.columns)

def column_picker(label, key, required=False):
    default = guesses.get(key)
    index = columns.index(default) + 1 if default in columns else 0
    options = ["— Not available —"] + columns
    chosen = st.selectbox(label, options, index=index, key=f"col_{key}")
    if required and chosen == options[0]:
        st.warning(f"Please select the {label.lower()}.")
    return None if chosen == options[0] else chosen

c1, c2, c3 = st.columns(3)
with c1:
    time_col = column_picker("Elapsed time column (required)", "time", True)
    power_col = column_picker("Power column", "watts")
with c2:
    hr_col = column_picker("Heart-rate column", "heartrate")
    cadence_col = column_picker("Cadence column", "cadence")
with c3:
    distance_col = column_picker("Distance column", "distance")
    altitude_col = column_picker("Altitude column", "altitude")
    speed_col = column_picker("Speed column", "speed")

if time_col is None:
    st.stop()

df = pd.DataFrame()
try:
    df["time_s"] = parse_time(raw[time_col])
except Exception as e:
    st.error(str(e))
    st.stop()

mapping = {"watts": power_col, "heartrate": hr_col, "cadence": cadence_col,
           "distance": distance_col, "altitude": altitude_col, "speed": speed_col}
for standard, original in mapping.items():
    if original:
        df[standard] = pd.to_numeric(raw[original], errors="coerce")

df = df.dropna(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)
df["time_s"] = df["time_s"] - df["time_s"].iloc[0]
duration_min = float(df["time_s"].max() / 60)
if duration_min <= 0:
    st.error("Ride duration is zero; please verify the selected time column.")
    st.stop()

st.subheader("2 · Ride overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Duration", f"{duration_min:.1f} min")
m2.metric("Samples", f"{len(df):,}")
m3.metric("Mean power", f"{df['watts'].mean():.0f} W" if "watts" in df else "—")
valid_hr = df.loc[df.heartrate > 0, "heartrate"] if "heartrate" in df else pd.Series(dtype=float)
m4.metric("Mean HR", f"{valid_hr.mean():.0f} bpm" if not valid_hr.empty else "—")

available = [c for c in ["watts", "heartrate", "cadence", "speed", "altitude"] if c in df]
if not available:
    st.warning("No metrics found besides time. Select/map at least one metric column.")
    st.stop()

st.subheader("3 · Interactive ride charts")
chosen_metrics = st.multiselect("Metrics to display", available, default=[c for c in ["watts", "heartrate"] if c in available])
if chosen_metrics:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for metric in chosen_metrics:
        secondary = metric in ("heartrate", "cadence", "speed")
        fig.add_trace(go.Scatter(x=df.time_s / 60, y=df[metric], name=metric, mode="lines",
                                 connectgaps=False), secondary_y=secondary)
    fig.update_layout(height=470, hovermode="x unified", dragmode="zoom",
                      xaxis_title="Elapsed time (minutes)", legend_title="Stream")
    fig.update_yaxes(title_text="Power / altitude", secondary_y=False)
    fig.update_yaxes(title_text="HR / cadence / speed", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("4 · Define or detect workout blocks")
ftp = st.number_input("FTP (watts; used for context, not required)", min_value=0, max_value=2000,
                      value=159, step=1)
mode = st.radio("Block setup", ["Create blocks manually", "Equal-duration blocks", "Detect power changes"],
                horizontal=True)

if mode == "Create blocks manually":
    with st.form("add_block"):
        block_name = st.text_input("Block name", "New block")
        start_min = st.number_input("Start (minutes)", min_value=0.0, max_value=float(duration_min),
                                    value=0.0, step=0.5)
        end_min = st.number_input("End (minutes)", min_value=0.1, max_value=float(duration_min),
                                  value=min(10.0, duration_min), step=0.5)
        add = st.form_submit_button("Add block")
    if "blocks" not in st.session_state:
        st.session_state.blocks = pd.DataFrame(columns=["Name", "Start (min)", "End (min)"])
    if add:
        if end_min <= start_min:
            st.error("End must be after start.")
        elif end_min > duration_min:
            st.error("End time cannot exceed the ride duration.")
        else:
            st.session_state.blocks = pd.concat([st.session_state.blocks,
                pd.DataFrame([{"Name": block_name, "Start (min)": start_min, "End (min)": end_min}])],
                ignore_index=True)
elif mode == "Equal-duration blocks":
    interval = st.number_input("Block duration (minutes)", min_value=1.0, value=10.0, step=1.0)
    if st.button("Generate equal blocks"):
        starts = np.arange(0, duration_min, interval)
        st.session_state.blocks = pd.DataFrame([
            {"Name": f"Block {i+1}", "Start (min)": float(s),
             "End (min)": float(min(s+interval, duration_min))}
            for i, s in enumerate(starts)
        ])
else:
    if "watts" not in df:
        st.warning("Power data is required for power-change detection.")
    else:
        smoothing = st.slider("Smoothing window (samples)", 1, 60, 10)
        min_len = st.slider("Minimum detected block duration (minutes)", 0.5, 10.0, 2.0, 0.5)
        threshold = st.slider("Power-change threshold (watts)", 5, 100, 20, 5)
        if st.button("Detect power blocks"):
            smooth = df.watts.rolling(smoothing, min_periods=1, center=True).mean()
            # Change-point candidates; merge nearby changes by minimum duration.
            candidates = [0]
            last = 0
            for i in range(1, len(smooth)):
                if abs(float(smooth.iloc[i]) - float(smooth.iloc[last])) >= threshold:
                    if (df.time_s.iloc[i] - df.time_s.iloc[last]) >= min_len * 60:
                        candidates.append(i)
                        last = i
            candidates.append(len(df)-1)
            candidates = sorted(set(candidates))
            rows = []
            for j, (a, b) in enumerate(zip(candidates[:-1], candidates[1:])):
                s, e = df.time_s.iloc[a]/60, df.time_s.iloc[b]/60
                if e > s:
                    rows.append({"Name": f"Detected {j+1}", "Start (min)": s, "End (min)": e})
            st.session_state.blocks = pd.DataFrame(rows)

if "blocks" not in st.session_state:
    st.session_state.blocks = pd.DataFrame([
        {"Name": "Warm-up", "Start (min)": 0.0, "End (min)": min(10.0, duration_min)}
    ])

st.caption("Edit the table directly. Add/remove rows with the table controls where available; each row is one analysis interval.")
edited = st.data_editor(st.session_state.blocks, num_rows="dynamic", use_container_width=True,
                        key="block_editor", column_config={
                            "Start (min)": st.column_config.NumberColumn(min_value=0, max_value=duration_min, step=0.1),
                            "End (min)": st.column_config.NumberColumn(min_value=0, max_value=duration_min, step=0.1)
                        })
st.session_state.blocks = edited

valid_blocks = edited.copy()
if not valid_blocks.empty:
    valid_blocks["Start (min)"] = pd.to_numeric(valid_blocks["Start (min)"], errors="coerce")
    valid_blocks["End (min)"] = pd.to_numeric(valid_blocks["End (min)"], errors="coerce")
    valid_blocks = valid_blocks.dropna(subset=["Start (min)", "End (min)"])
    valid_blocks = valid_blocks[(valid_blocks["Start (min)"] >= 0) &
                                (valid_blocks["End (min)"] > valid_blocks["Start (min)"]) &
                                (valid_blocks["End (min)"] <= duration_min)]

if st.button("Analyze selected blocks", type="primary"):
    if valid_blocks.empty:
        st.error("Add at least one valid block.")
    else:
        result = summarize(df, valid_blocks, ftp)
        st.session_state.analysis_result = result

if "analysis_result" in st.session_state:
    st.subheader("5 · Block analysis")
    result = st.session_state.analysis_result
    st.dataframe(result, use_container_width=True)
    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button("Download analysis CSV", data=csv, file_name="ride_block_analysis.csv",
                       mime="text/csv")
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        result.to_excel(writer, index=False, sheet_name="Block Analysis")
    st.download_button("Download analysis Excel", data=buffer.getvalue(),
                       file_name="ride_block_analysis.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with st.expander("Data preview"):
    st.dataframe(df.head(100), use_container_width=True)
