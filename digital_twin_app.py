from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# --------------------------------------------------------------- paths ----
def _resolve_base() -> Path:
    here = Path(__file__).resolve().parent
    for cand in [here, here.parent, here.parent.parent, here.parent.parent.parent]:
        if (cand / "Wind Farm A" / "datasets").exists():
            return cand
    raise FileNotFoundError("Cannot locate Wind Farm A/datasets")


def _resolve_artifacts() -> Path:
    here = Path(__file__).resolve().parent
    # Check next to the script first, then in parents
    for cand in [here / "artifacts", here, here.parent / "artifacts"]:
        if (cand / "model_config.json").exists():
            return cand
    raise FileNotFoundError("Cannot locate artifacts/model_config.json")


BASE_DIR = _resolve_base()
DATA_DIR = BASE_DIR / "Wind Farm A" / "datasets"
EVENT_INFO_PATH = BASE_DIR / "Wind Farm A" / "comma_event_info.csv"
DESC_PATH = BASE_DIR / "Wind Farm A" / "comma_feature_description.csv"
ARTIFACTS = _resolve_artifacts()


# --------------------------------------------------- load model artifacts ----
@st.cache_resource(show_spinner="Loading model & scaler...")
def load_artifacts() -> dict:
    with open(ARTIFACTS / "model_config.json") as f:
        config = json.load(f)
    # Lazy keras import (slow)
    from keras.models import load_model

    model = load_model(str(ARTIFACTS / config["model_filename"]))
    scaler = joblib.load(str(ARTIFACTS / "robust_scaler.pkl"))
    clf_path = ARTIFACTS / "root_cause_classifier.pkl"
    clf = joblib.load(str(clf_path)) if clf_path.exists() else None
    sig_path = ARTIFACTS / "failure_type_signatures.json"
    signatures = json.loads(sig_path.read_text()) if sig_path.exists() else {"by_type": {}}
    return {
        "model": model,
        "scaler": scaler,
        "classifier": clf,
        "config": config,
        "signatures": signatures,
    }


@st.cache_data(show_spinner="Loading event info & feature descriptions...")
def load_event_info() -> tuple[pd.DataFrame, dict]:
    event_info = pd.read_csv(EVENT_INFO_PATH, parse_dates=["event_start", "event_end"])
    desc_df = pd.read_csv(DESC_PATH)
    desc = dict(zip(desc_df["sensor_name"], desc_df["description"]))
    return event_info, desc


@st.cache_data(show_spinner="Loading dataset...")
def load_dataset(event_id: int) -> pd.DataFrame:
    fp = DATA_DIR / f"comma_{event_id}.csv"
    df = pd.read_csv(fp, parse_dates=["time_stamp"]).sort_values("time_stamp").reset_index(drop=True)
    return df


def get_friendly_name(col: str, desc: dict) -> str:
    m = re.match(r"(sensor_\d+|wind_speed_\d+|reactive_power_\d+|power_\d+)", col)
    if not m:
        return col
    base = m.group(1)
    description = desc.get(base, "Unknown")
    num = re.search(r"\d+", base).group()
    stat = col.replace(base, "").strip("_")
    return f"{description} [{stat}] ({num})" if stat else f"{description} ({num})"


# --------------------------------------------------- prediction pipeline ----
def predict_rul_curve(df: pd.DataFrame, artifacts: dict, segment: str = "prediction") -> pd.DataFrame:
    config = artifacts["config"]
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    sensor_names = config["sensor_names"]
    LOOKBACK = config["lookback_rows"]
    ROWS_PER_HOUR = config["rows_per_hour"]
    MAX_RUL = config["max_rul_rows"]

    # Use the entire dataset for context (so lookback can extend back into 'train')
    # but only emit predictions where train_test == segment.
    df = df.copy()
    df[sensor_names] = (
        df[sensor_names].interpolate(method="linear", limit=6).ffill().bfill()
    )
    scaled = scaler.transform(df[sensor_names].values).astype(np.float32)

    target_idx = df.index[df["train_test"] == segment].tolist()
    if not target_idx:
        return pd.DataFrame(columns=["time_stamp", "predicted_rul_rows", "predicted_rul_hours"])

    stride = max(1, ROWS_PER_HOUR)  # one prediction per hour for the dashboard
    samples_idx = [i for i in target_idx if i >= LOOKBACK and (i - LOOKBACK) >= 0][::stride]
    if not samples_idx:
        return pd.DataFrame(columns=["time_stamp", "predicted_rul_rows", "predicted_rul_hours"])

    X = np.stack([scaled[i - LOOKBACK : i] for i in samples_idx]).astype(np.float32)
    preds = model.predict(X, verbose=0).flatten().clip(0, MAX_RUL)

    return pd.DataFrame(
        {
            "time_stamp": df.loc[samples_idx, "time_stamp"].values,
            "row_idx": samples_idx,
            "predicted_rul_rows": preds,
            "predicted_rul_hours": preds / ROWS_PER_HOUR,
        }
    )


def classify_at_window(
    df: pd.DataFrame, row_idx: int, artifacts: dict
) -> list[tuple[str, float]] | None:
    """Run the row-level classifier on the last 6 h of the lookback window
    and average per-row probabilities — more faithful to how the classifier
    was trained (one prediction per row, not per window-mean)."""
    clf = artifacts.get("classifier")
    if clf is None:
        return None
    sensor_names = artifacts["config"]["sensor_names"]
    LOOKBACK = artifacts["config"]["lookback_rows"]
    ROWS_PER_HOUR = artifacts["config"]["rows_per_hour"]
    if row_idx - LOOKBACK < 0:
        return None
    last_n = 6 * ROWS_PER_HOUR  # last 6 hours
    start = max(row_idx - last_n, row_idx - LOOKBACK)
    sub = df.loc[start:row_idx, sensor_names].dropna()
    if sub.empty:
        return None
    probs = clf.predict_proba(sub.values).mean(axis=0)
    ranked = sorted(zip(clf.classes_, probs), key=lambda x: -x[1])
    return ranked


# ----------------------------------------------------------------- ui ----
def _hero_card(event_id: int, is_anomaly: bool, description: str) -> str:
    """HTML hero card — color-coded, prominent display of the selected event.

    Uses semi-transparent background tint so it adapts to both light and dark
    Streamlit themes (instead of hard-coded white/grey gradient).
    """
    color = "#dc3545" if is_anomaly else "#28a745"
    badge = "ANOMALY" if is_anomaly else "NORMAL"
    detail = description if description else (
        "labeled as anomaly" if is_anomaly else "labeled as normal operation"
    )
    # rgba background tint = the badge color at 8% alpha — works on light & dark.
    bg_tint = f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.08)"
    return f"""
    <div style="
        background: {bg_tint};
        border-left: 6px solid {color};
        padding: 1.3rem 1.7rem;
        border-radius: 10px;
        margin: 0.5rem 0 1.3rem 0;
    ">
        <div style="display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap;">
            <div>
                <div style="font-size:0.78rem; opacity:0.7; letter-spacing:0.07em; text-transform:uppercase; font-weight:600;">
                    Currently viewing
                </div>
                <div style="font-size:1.8rem; font-weight:700; margin-top:0.15rem;">
                    Turbine event #{event_id}
                </div>
                <div style="font-size:0.95rem; opacity:0.85; margin-top:0.2rem;">
                    {detail}
                </div>
            </div>
            <span style="
                background:{color};
                color:white;
                padding:0.4rem 1.05rem;
                border-radius:999px;
                font-weight:700;
                font-size:0.78rem;
                letter-spacing:0.08em;
                white-space:nowrap;
            ">{badge}</span>
        </div>
    </div>
    """


def main():
    st.set_page_config(
        page_title="Wind Farm Digital Twin",
        page_icon="🌬️",
        layout="wide",
    )

    # Minimal global tweak — only top padding. Don't override st.metric internals;
    # Streamlit's DOM for that widget changes between versions, and overriding it
    # breaks the value display.
    st.markdown(
        """
        <style>
            .block-container { padding-top: 2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🌬️ Wind Farm Digital Twin")
    st.caption(
        "Streams the prediction-split sensor data through the trained "
        "Conv1D→BiGRU→Dense RUL model and the failure classifier — forecasts up to 72 h ahead."
    )

    artifacts = load_artifacts()
    event_info, desc = load_event_info()
    config = artifacts["config"]
    PREDICTION_HORIZON_HOURS = config["prediction_horizon_hours"]
    test_mae_h = config.get("test_mae_hours", float("nan"))

    # ----- Sidebar: model context only -----
    with st.sidebar:
        st.markdown("### 🧠 Model")
        st.metric("Test MAE on unseen events", f"{test_mae_h:.2f} h")
        st.markdown(
            f"<div style='line-height:1.7; font-size:0.92rem;'>"
            f"<b>Architecture:</b> Conv1D × 2 → BiGRU → Dense<br>"
            f"<b>Lookback:</b> {config['lookback_rows']} rows ({config['lookback_rows']//config['rows_per_hour']} h)<br>"
            f"<b>Horizon:</b> {PREDICTION_HORIZON_HOURS} h"
            f"</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Train / test event split"):
            st.markdown(f"**Train events:** `{config.get('train_eids')}`")
            st.markdown(f"**Test events:**  `{config.get('test_eids')}`")
        with st.expander("Top sensors fed to the model"):
            for i, fname in enumerate(config.get("friendly_names", [])[:10], 1):
                st.markdown(f"{i}. {fname}")

    # ----- Event selector: scrollable slider in main area -----
    options = []
    for _, r in event_info.iterrows():
        eid = int(r["event_id"])
        label = r["event_label"]
        desc_text = r.get("event_description", "")
        if not isinstance(desc_text, str):
            desc_text = ""
        options.append((eid, label, desc_text))
    options.sort(key=lambda x: x[0])
    event_ids = [o[0] for o in options]
    info_lookup = {o[0]: (o[1], o[2]) for o in options}

    st.markdown("#### 📂 Pick a turbine event")
    chosen_eid = st.select_slider(
        "Drag the slider to scrub through events",
        options=event_ids,
        value=event_ids[0],
        format_func=lambda eid: f"#{eid} ({info_lookup[eid][0]})",
        label_visibility="collapsed",
    )

    info_row = event_info[event_info["event_id"] == chosen_eid].iloc[0]
    is_anomaly = info_row["event_label"] == "anomaly"
    desc_text = info_row.get("event_description", "") or ""
    if not isinstance(desc_text, str):
        desc_text = ""

    # Hero card highlights the selection
    st.markdown(_hero_card(chosen_eid, is_anomaly, desc_text), unsafe_allow_html=True)

    # ----- Run the model -----
    with st.spinner("Running the digital twin..."):
        df = load_dataset(chosen_eid)
        pred = predict_rul_curve(df, artifacts, segment="prediction")

    if pred.empty:
        st.warning("No prediction-split rows available after the lookback window; cannot forecast.")
        return

    # Sustained-alert detection: predicted RUL must drop below threshold for
    # REQUIRE_CONSEC consecutive samples to suppress noise.
    ALERT_THRESHOLD_H = max(PREDICTION_HORIZON_HOURS - 6, 24)
    REQUIRE_CONSEC = 3
    below = (pred["predicted_rul_hours"] < ALERT_THRESHOLD_H).astype(int).values
    alert_idx = None
    run = 0
    for k, v in enumerate(below):
        run = run + 1 if v else 0
        if run >= REQUIRE_CONSEC:
            alert_idx = k - REQUIRE_CONSEC + 1
            break

    # ----- Top metric strip -----
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Forecast samples", f"{len(pred):,}")
    with c2:
        st.metric("Min predicted RUL", f"{pred['predicted_rul_hours'].min():.1f} h")
    with c3:
        avg = pred["predicted_rul_hours"].mean()
        st.metric("Mean predicted RUL", f"{avg:.1f} h")
    with c4:
        if alert_idx is not None:
            ts = pred.iloc[alert_idx]["time_stamp"]
            st.metric("⚠ First sustained alert", pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M"))
        else:
            st.metric("Status", "✅ No alert")

    st.divider()

    # ----- RUL forecast plot -----
    fig = go.Figure()

    # Filled area under the prediction line for visual weight
    fig.add_trace(
        go.Scatter(
            x=pred["time_stamp"], y=pred["predicted_rul_hours"],
            mode="lines",
            name="Predicted RUL",
            line=dict(color="#1f77b4", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(31, 119, 180, 0.10)",
            hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Predicted RUL: <b>%{y:.1f} h</b><extra></extra>",
        )
    )

    # Horizon line
    fig.add_hline(
        y=PREDICTION_HORIZON_HOURS, line_dash="dash", line_color="#28a745", line_width=1.5,
        annotation_text=f"{PREDICTION_HORIZON_HOURS} h horizon", annotation_position="right",
    )
    # Alert threshold
    fig.add_hline(
        y=ALERT_THRESHOLD_H, line_dash="dot", line_color="#fd7e14", line_width=1.5,
        annotation_text=f"alert ≤ {ALERT_THRESHOLD_H} h", annotation_position="right",
    )
    # Critical zone
    fig.add_hrect(
        y0=0, y1=24, fillcolor="#dc3545", opacity=0.06, line_width=0,
        annotation_text="critical (< 24 h)", annotation_position="top left",
    )

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=f"<b>RUL forecast — event #{chosen_eid}</b>",
            x=0, font=dict(size=18),
        ),
        xaxis_title="Date & time",
        yaxis_title="Hours until predicted failure",
        height=480,
        hovermode="x unified",
        margin=dict(l=60, r=120, t=70, b=50),
        showlegend=False,
    )
    fig.update_yaxes(range=[0, max(75, float(pred["predicted_rul_hours"].max()) * 1.05)])
    fig.update_xaxes(showgrid=True, gridcolor="#f1f3f5")
    fig.update_yaxes(showgrid=True, gridcolor="#f1f3f5")

    st.plotly_chart(fig, use_container_width=True)

    # ----- Diagnosis section -----
    st.divider()
    st.subheader("🩺 Diagnosis")
    if alert_idx is None:
        st.success(
            f"**No sustained anomaly predicted** within the {PREDICTION_HORIZON_HOURS} h horizon. "
            f"Threshold: {ALERT_THRESHOLD_H} h sustained over {REQUIRE_CONSEC}+ consecutive samples."
        )
    else:
        first_idx = int(pred.iloc[alert_idx]["row_idx"])
        first_ts = pd.Timestamp(pred.iloc[alert_idx]["time_stamp"])
        st.warning(
            f"**Sustained alert** beginning **{first_ts.strftime('%Y-%m-%d %H:%M')}** — "
            f"predicted RUL has been below {ALERT_THRESHOLD_H} h for {REQUIRE_CONSEC}+ consecutive samples. "
            f"Running the failure classifier on the alert window..."
        )
        ranked = classify_at_window(df, first_idx, artifacts)
        if not ranked:
            st.info("Classifier not available — only the RUL forecast is shown.")
        else:
            top_class, top_p = ranked[0]
            sig = artifacts["signatures"].get("by_type", {}).get(top_class)

            d1, d2 = st.columns([1, 1])
            with d1:
                st.markdown("**Top-5 candidate failure modes**")
                cdf = pd.DataFrame(ranked, columns=["failure mode", "probability"]).head(5)
                cdf["probability"] = cdf["probability"].round(3)
                st.dataframe(cdf, hide_index=True, use_container_width=True)
            with d2:
                if sig:
                    st.markdown(f"**Most likely:** `{top_class}` ({top_p:.0%})")
                    st.markdown(
                        f"Seen in **{sig['n_events']}** historical event(s); "
                        f"average duration **{sig['mean_duration_h']:.0f} h** "
                        f"(min {sig['min_duration_h']:.0f}, max {sig['max_duration_h']:.0f})."
                    )
                else:
                    # Top class is "Normal Operation" or otherwise has no signature.
                    st.markdown(f"**Most likely:** `{top_class}` ({top_p:.0%})")
                    st.info(
                        "Classifier favours `Normal Operation` even though the RUL "
                        "forecast crossed the alert threshold — the alert is likely a "
                        "soft signal (look at the runner-up class below for the most "
                        "probable failure mode if this is genuinely an anomaly)."
                    )

            # Sensor-divergence table — only shown when the top class is a known anomaly.
            # If the top class is Normal Operation, fall back to the highest-prob *anomaly*
            # class so the user still sees per-sensor context.
            display_class = top_class if sig else next(
                (c for c, _ in ranked
                 if artifacts["signatures"].get("by_type", {}).get(c)),
                None,
            )
            display_sig = artifacts["signatures"].get("by_type", {}).get(display_class) if display_class else None

            if display_sig:
                # Tolerate both key names — pipeline script wrote `top_sensors_abs_z`,
                # the notebook-builder wrote `top_sensors`. Same content either way.
                top_sensors_list = (
                    display_sig.get("top_sensors_abs_z")
                    or display_sig.get("top_sensors")
                    or []
                )
                if top_sensors_list:
                    header = (
                        f"**Top sensors driving the signature of `{display_class}`** "
                        "(z-score vs healthy baseline):"
                    )
                    st.markdown(header)
                    top_rows = [
                        {"sensor": get_friendly_name(s, desc),
                         "z_score": round(display_sig["divergence"].get(s, 0.0), 2)}
                        for s in top_sensors_list[:8]
                    ]
                    st.dataframe(pd.DataFrame(top_rows), hide_index=True, use_container_width=True)

    # ----- Raw data preview -----
    with st.expander("📊 Show raw prediction-split sensor rows"):
        cols = ["time_stamp", "train_test", "status_type_id"] + config["sensor_names"][:5]
        st.dataframe(
            df[df["train_test"] == "prediction"][cols].head(50),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
