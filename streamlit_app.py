import json
from datetime import datetime, timedelta
from urllib import request, error

import altair as alt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st


# ------------------------------
# Data Builders
# ------------------------------

def build_assets(scenario: str = "Offshore expensive") -> pd.DataFrame:
    """Create a synthetic asset register for ORACLE demo."""
    records = [
        ("T1", "Turbine T1", "Rotating", 9, 1, "Offshore", 180000, "Primary gas turbine driver for compression train."),
        ("C1", "Compressor C1", "Rotating", 10, 0, "Offshore", 210000, "Main export gas compressor."),
        ("C2", "Compressor C2", "Rotating", 8, 1, "Offshore", 170000, "Secondary compression train."),
        ("P1", "Pump P1", "Rotating", 6, 2, "Onshore", 55000, "Produced water transfer pump."),
        ("P2", "Pump P2", "Rotating", 7, 1, "Onshore", 68000, "Condensate booster pump."),
        ("TR1", "Transformer TR1", "Electrical", 9, 1, "Onshore", 120000, "Main step-up transformer for utility bus."),
        ("TR2", "Transformer TR2", "Electrical", 7, 1, "Onshore", 95000, "Backup transformer for critical loads."),
        ("SW1", "Switchgear SW1", "Electrical", 8, 1, "Onshore", 88000, "Medium-voltage switchgear lineup."),
        ("SW2", "Switchgear SW2", "Electrical", 6, 2, "Onshore", 62000, "Low-voltage distribution section."),
        ("S1", "Separator S1", "Process", 8, 1, "Offshore", 140000, "High-pressure three-phase separator."),
        ("S2", "Separator S2", "Process", 7, 1, "Offshore", 130000, "Low-pressure polishing separator."),
        ("HX1", "Heat Exchanger HX1", "Process", 7, 1, "Onshore", 76000, "Gas cooler heat exchanger."),
        ("HX2", "Heat Exchanger HX2", "Process", 6, 1, "Onshore", 72000, "Produced water heat exchanger."),
        ("V1", "Control Valve V1", "Process", 5, 2, "Onshore", 38000, "Anti-surge recycle control valve."),
        ("V2", "Control Valve V2", "Process", 6, 1, "Offshore", 52000, "Inlet choke control valve."),
        ("F1", "Flare KO Drum F1", "Process", 7, 1, "Offshore", 102000, "Flare knock-out drum for relief routing."),
    ]

    df = pd.DataFrame(
        records,
        columns=[
            "asset_id",
            "asset_name",
            "subsystem",
            "criticality",
            "redundancy_level",
            "location",
            "mobilization_cost",
            "description",
        ],
    )

    if scenario == "Onshore cheaper":
        offshore_mult, onshore_mult = 0.88, 0.72
    else:
        offshore_mult, onshore_mult = 1.25, 1.00

    loc_mult = np.where(df["location"] == "Offshore", offshore_mult, onshore_mult)
    df["mobilization_cost"] = (df["mobilization_cost"] * loc_mult).round(0).astype(int)
    return df


def build_graph(assets_df: pd.DataFrame):
    """Build directed dependency graph and systemic priority index."""
    graph = nx.DiGraph()
    for _, row in assets_df.iterrows():
        graph.add_node(
            row["asset_id"],
            asset_name=row["asset_name"],
            subsystem=row["subsystem"],
            criticality=row["criticality"],
        )

    edges = [
        ("TR1", "SW1", 0.88), ("TR2", "SW1", 0.58), ("SW1", "T1", 0.82), ("SW1", "C1", 0.64),
        ("SW2", "P1", 0.52), ("SW2", "P2", 0.56),
        ("T1", "C1", 0.92), ("T1", "C2", 0.67), ("V1", "C1", 0.52),
        ("C1", "S1", 0.84), ("C2", "S1", 0.73), ("P2", "S1", 0.43), ("V2", "S1", 0.36),
        ("S1", "HX1", 0.74), ("HX1", "S2", 0.57), ("P1", "S2", 0.42),
        ("S2", "F1", 0.61), ("F1", "V2", 0.31),
    ]
    for src, dst, w in edges:
        graph.add_edge(src, dst, propagation_weight=w)

    out_degree = dict(graph.out_degree(weight="propagation_weight"))
    betweenness = nx.betweenness_centrality(graph, weight="propagation_weight", normalized=True)

    metrics = assets_df[["asset_id", "asset_name", "criticality"]].copy()
    metrics["out_degree"] = metrics["asset_id"].map(out_degree).fillna(0)
    metrics["betweenness"] = metrics["asset_id"].map(betweenness).fillna(0)

    def norm(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        span = s.max() - s.min()
        if pd.isna(span) or span <= 1e-9:
            return pd.Series(0.0, index=s.index, dtype=float)
        return ((s - s.min()) / span).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    metrics["out_degree_n"] = norm(metrics["out_degree"])
    metrics["betweenness_n"] = norm(metrics["betweenness"])
    metrics["criticality_n"] = norm(metrics["criticality"])
    metrics["systemic_priority"] = (
        0.45 * metrics["out_degree_n"] + 0.35 * metrics["betweenness_n"] + 0.20 * metrics["criticality_n"]
    ) * 100
    metrics = metrics.sort_values("systemic_priority", ascending=False).reset_index(drop=True)

    adjacency = []
    for node in graph.nodes:
        succ = list(graph.successors(node))
        if succ:
            adjacency.append(
                f"{node} -> " + ", ".join([f"{s}({graph[node][s]['propagation_weight']:.2f})" for s in succ])
            )
        else:
            adjacency.append(f"{node} -> [no downstream dependencies]")

    return graph, metrics, adjacency


def generate_health_timeseries(assets_df: pd.DataFrame, seed: int = 42, days: int = 90):
    """Generate synthetic 90-day health trends and anomalies by subsystem patterns."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(datetime.now().date() - timedelta(days=days - 1), periods=days, freq="D")

    modes = np.array(["Normal", "High Load", "Start-Stop"])
    mode_shift = {"Normal": 0.0, "High Load": -3.2, "Start-Stop": -1.7}

    all_data, summary = [], []

    for _, asset in assets_df.iterrows():
        subsystem = asset["subsystem"]
        base = rng.uniform(73, 96)
        noise = rng.normal(0, 1.8, days)
        mode_series = rng.choice(modes, size=days, p=[0.62, 0.23, 0.15])

        if subsystem == "Rotating":
            gradual = np.linspace(0, rng.uniform(9, 24), days)
            step = np.zeros(days)
            step[rng.integers(52, 76):] = rng.uniform(4, 10)
            degradation = gradual + step
        elif subsystem == "Electrical":
            degradation = np.linspace(0, rng.uniform(5, 13), days)
        else:
            gradual = np.linspace(0, rng.uniform(3, 11), days)
            cyc = 2.2 * np.sin(np.linspace(0, 5 * np.pi, days))
            degradation = gradual - cyc

        health = []
        for i in range(days):
            h = base - degradation[i] + mode_shift[mode_series[i]] + noise[i]
            health.append(np.clip(h, 5, 100))

        asset_ts = pd.DataFrame(
            {
                "date": dates,
                "asset_id": asset["asset_id"],
                "asset_name": asset["asset_name"],
                "subsystem": subsystem,
                "operating_mode": mode_series,
                "health_index": np.round(health, 2),
            }
        )
        asset_ts["rolling_mean_7d"] = asset_ts["health_index"].rolling(7, min_periods=3).mean()
        std7 = asset_ts["health_index"].rolling(7, min_periods=3).std().fillna(1.5).replace(0, 1.5)
        asset_ts["anomaly_score"] = ((asset_ts["rolling_mean_7d"] - asset_ts["health_index"]).abs() / std7).fillna(0)

        current_health = float(asset_ts["health_index"].iloc[-1])
        recent14 = asset_ts.tail(14)
        slope = float(np.polyfit(np.arange(len(recent14)), recent14["health_index"], 1)[0]) if len(recent14) > 1 else 0.0
        anomaly = float(asset_ts["anomaly_score"].iloc[-1])

        threshold = 60
        if slope < -0.05:
            ttf = float(np.clip(max(0.0, (threshold - current_health) / slope), 0, 365))
        elif current_health <= threshold:
            ttf = 0.0
        else:
            ttf = 365.0

        summary.append(
            {
                "asset_id": asset["asset_id"],
                "current_health": round(current_health, 2),
                "slope_14d": round(slope, 3),
                "anomaly_score": round(anomaly, 3),
                "predicted_time_to_threshold": round(ttf, 1),
            }
        )
        all_data.append(asset_ts)

    return pd.concat(all_data, ignore_index=True), pd.DataFrame(summary)


# ------------------------------
# Inference & Scoring Helpers
# ------------------------------

def parse_notification(text: str, selected_asset: pd.Series) -> dict:
    """Simple rule-based parser for free-text maintenance notifications."""
    text_l = (text or "").lower()

    keyword_map = {
        "vibration": {
            "failure": "Rotor imbalance / bearing degradation",
            "missing": ["RMS vibration trend", "Bearing temperature", "Recent alignment report"],
            "inspection": ["Collect FFT spectrum", "Check bearing lubrication", "Perform laser alignment check"],
            "confidence": 0.86,
        },
        "noise": {
            "failure": "Mechanical looseness / cavitation",
            "missing": ["Acoustic recording", "Load condition at event", "Valve position history"],
            "inspection": ["Acoustic inspection", "Casing bolt torque check", "Process upset review"],
            "confidence": 0.74,
        },
        "temperature": {
            "failure": "Thermal stress / insulation degradation",
            "missing": ["Infrared thermography", "Ambient condition log", "Oil temperature trend"],
            "inspection": ["Thermal scan", "Cooling path verification", "Insulation resistance test"],
            "confidence": 0.81,
        },
        "leak": {
            "failure": "Seal/gasket failure or wall thinning",
            "missing": ["Leak rate estimate", "Fluid composition", "Recent thickness measurement"],
            "inspection": ["Visual + UT spot checks", "Seal integrity check", "Corrosion under insulation review"],
            "confidence": 0.78,
        },
        "pressure": {
            "failure": "Flow restriction / control instability",
            "missing": ["Upstream/downstream pressure trend", "Valve travel history", "Recent calibration status"],
            "inspection": ["Pressure transmitter validation", "Control loop tuning review", "Line blockage check"],
            "confidence": 0.76,
        },
    }

    found = [k for k in keyword_map if k in text_l]
    if found:
        picked = found[0]
        rule = keyword_map[picked]
        suspected_failure = rule["failure"]
        confidence = rule["confidence"]
        missing_information = rule["missing"]
        recommended_inspection = rule["inspection"]
    else:
        suspected_failure = "General performance degradation"
        confidence = 0.58
        missing_information = ["Exact event timestamp", "Operating context", "Recent maintenance actions"]
        recommended_inspection = ["Initial site walkdown", "Baseline condition monitoring", "Operator interview"]

    component_hint = selected_asset["asset_name"]
    if "bearing" in text_l:
        component_hint = f"{selected_asset['asset_name']} - Bearing assembly"
    elif "seal" in text_l:
        component_hint = f"{selected_asset['asset_name']} - Mechanical seal"
    elif "winding" in text_l:
        component_hint = f"{selected_asset['asset_name']} - Electrical winding"

    return {
        "suspected_failure_type": suspected_failure,
        "suspected_component": component_hint,
        "confidence": round(float(confidence), 2),
        "missing_information": missing_information,
        "recommended_inspection": recommended_inspection,
    }


def notification_templates(subsystem: str):
    """Common operator phrases to improve key-in quality."""
    base = [
        "High vibration noticed during high load; noise increased near bearing housing.",
        "Temperature trend rising steadily over last shift; check cooling path.",
        "Pressure fluctuation observed during start-stop cycle; possible control instability.",
        "Intermittent leak observed near flange area; leak rate appears to increase under load.",
    ]
    if subsystem == "Electrical":
        base = [
            "Transformer temperature alarm intermittently triggered under peak loading.",
            "Switchgear compartment showing abnormal hot spot and occasional noise.",
            "Winding or insulation degradation suspected after repeated thermal excursions.",
        ] + base[:2]
    elif subsystem == "Rotating":
        base = [
            "Vibration and tonal noise increased during high load operation.",
            "Bearing temperature rise with possible lubrication degradation symptoms.",
            "Start-stop cycles causing unstable vibration baseline and transient spikes.",
        ] + base[:2]
    return base[:5]


def mock_mistral_5w(user_text: str, asset_name: str, subsystem: str) -> dict:
    """Offline mock of Mistral post-processing into standardized 5W maintenance note."""
    txt = (user_text or "").strip()
    text_l = txt.lower()

    what = "Abnormal condition reported"
    if "vibration" in text_l:
        what = "Abnormal vibration trend"
    elif "temperature" in text_l:
        what = "Abnormal temperature increase"
    elif "pressure" in text_l:
        what = "Pressure instability"
    elif "leak" in text_l:
        what = "Leakage observed"

    when = "During latest operating shift"
    if "start" in text_l or "start-stop" in text_l:
        when = "During start-stop transition"
    elif "high load" in text_l:
        when = "During high-load operation"

    where = f"{asset_name} ({subsystem})"
    who = "Field Operator"
    why = "Potential degradation requiring early inspection to avoid cascading impact"

    standardized = (
        f"[WHAT] {what}. [WHEN] {when}. [WHERE] {where}. "
        f"[WHO] {who}. [WHY] {why}. Source note: {txt or 'N/A'}"
    )

    return {
        "what": what,
        "when": when,
        "where": where,
        "who": who,
        "why": why,
        "standardized_5w": standardized,
        "llm_model": "Mistral (mock offline prompt)",
    }


def call_local_mistral_5w(user_text: str, asset_name: str, subsystem: str, endpoint: str = "http://localhost:11434/api/generate", model: str = "mistral"):
    """Call a local Mistral-compatible endpoint (e.g., Ollama) to standardize 5W."""
    prompt = (
        "You are a maintenance assistant. Convert the user note into strict 5W fields. "
        "Return ONLY valid JSON with keys: what, when, where, who, why, standardized_5w. "
        f"Asset: {asset_name} | Subsystem: {subsystem}. User note: {user_text}"
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }

    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=12) as resp:
            raw = resp.read().decode("utf-8")
        outer = json.loads(raw)
        text = outer.get("response", "{}")
        parsed = json.loads(text)

        out = {
            "what": str(parsed.get("what", "N/A")),
            "when": str(parsed.get("when", "N/A")),
            "where": str(parsed.get("where", f"{asset_name} ({subsystem})")),
            "who": str(parsed.get("who", "Field Operator")),
            "why": str(parsed.get("why", "N/A")),
            "standardized_5w": str(parsed.get("standardized_5w", "N/A")),
            "llm_model": f"{model} (local)",
        }
        return True, out, ""
    except (error.URLError, TimeoutError, json.JSONDecodeError, error.HTTPError, ValueError) as ex:
        return False, {}, str(ex)


def call_remote_mistral_5w(user_text: str, asset_name: str, subsystem: str, endpoint: str, model: str, api_key: str = ""):
    """Call remote API endpoint for 5W standardization (cloud deployment mode)."""
    prompt = (
        "You are a maintenance assistant. Convert the user note into strict 5W fields. "
        "Return ONLY valid JSON with keys: what, when, where, who, why, standardized_5w. "
        f"Asset: {asset_name} | Subsystem: {subsystem}. User note: {user_text}"
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=18) as resp:
            raw = resp.read().decode("utf-8")
        outer = json.loads(raw)

        # Accept either direct 5W JSON or wrapped `response` JSON string.
        if all(k in outer for k in ["what", "when", "where", "who", "why", "standardized_5w"]):
            parsed = outer
        else:
            text = outer.get("response", "{}")
            parsed = json.loads(text)

        out = {
            "what": str(parsed.get("what", "N/A")),
            "when": str(parsed.get("when", "N/A")),
            "where": str(parsed.get("where", f"{asset_name} ({subsystem})")),
            "who": str(parsed.get("who", "Field Operator")),
            "why": str(parsed.get("why", "N/A")),
            "standardized_5w": str(parsed.get("standardized_5w", "N/A")),
            "llm_model": f"{model} (remote API)",
        }
        return True, out, ""
    except (error.URLError, TimeoutError, json.JSONDecodeError, error.HTTPError, ValueError) as ex:
        return False, {}, str(ex)


def compute_risk_score(systemic_priority_norm: float, current_health: float, anomaly_score: float):
    """Risk formula required by the demo specification."""
    anomaly_n = float(np.clip(anomaly_score / 5.0, 0, 1))
    risk_score = (
        systemic_priority_norm * 0.5
        + ((100 - current_health) / 100.0) * 0.3
        + anomaly_n * 0.2
    ) * 100
    return float(np.clip(risk_score, 0, 100)), anomaly_n


def evaluate_options(asset: pd.Series, risk_score: float, predicted_ttf: float, defer_weeks: int, planned_window: str):
    """Evaluate four intervention strategies with synthetic economics and risk impacts."""
    base_cost = float(asset["mobilization_cost"])
    crit = asset["criticality"] / 10

    options = [
        {"option": "Immediate Repair", "risk_reduction": np.clip(70 + 20 * crit, 0, 100), "expected_downtime_hours": 16 + 8 * crit, "mobilization_cost": base_cost * 1.15},
        {"option": "Defer to Next Window", "risk_reduction": np.clip(28 + 3 * defer_weeks, 0, 75), "expected_downtime_hours": 8 + 2 * defer_weeks, "mobilization_cost": base_cost * (0.88 + 0.02 * defer_weeks)},
        {"option": "Merge with Planned Maintenance", "risk_reduction": np.clip(55 + 8 * crit, 0, 90), "expected_downtime_hours": 12 + 4 * crit, "mobilization_cost": base_cost * 0.82},
        {"option": "Operational Mitigation", "risk_reduction": np.clip(35 + 10 * (1 - crit), 0, 65), "expected_downtime_hours": 4 + 2 * (1 - crit), "mobilization_cost": base_cost * 0.58},
    ]

    rows = []
    for opt in options:
        residual = np.clip(risk_score - opt["risk_reduction"] * 0.65, 0, 100)
        urgency_bonus = 12 if predicted_ttf < 30 and opt["option"] == "Immediate Repair" else 0
        window_bonus = 6 if opt["option"] == "Merge with Planned Maintenance" else 0
        score = (
            0.55 * (100 - residual)
            + 0.25 * opt["risk_reduction"]
            + 0.10 * (100 - min(opt["expected_downtime_hours"], 100))
            + 0.10 * (100 - min(opt["mobilization_cost"] / 4000, 100))
            + urgency_bonus
            + window_bonus
        )
        rows.append(
            {
                "option": opt["option"],
                "risk_reduction": round(float(opt["risk_reduction"]), 1),
                "expected_downtime_hours": round(float(opt["expected_downtime_hours"]), 1),
                "mobilization_cost": round(float(opt["mobilization_cost"]), 0),
                "residual_risk": round(float(residual), 1),
                "decision_score": round(float(np.clip(score, 0, 100)), 1),
                "planned_window": planned_window if opt["option"] == "Merge with Planned Maintenance" else "-",
                "defer_weeks": defer_weeks if opt["option"] == "Defer to Next Window" else 0,
            }
        )

    return pd.DataFrame(rows).sort_values("decision_score", ascending=False).reset_index(drop=True)


def retrieve_standards_snippets(subsystem: str, suspected_failure_type: str):
    """Retrieve 1-2 guidance snippets based on subsystem and failure hypothesis."""
    library = [
        {"title": "IEEE C57.104 – Transformer DGA interpretation (excerpt)", "tags": ["Electrical", "thermal", "insulation", "temperature"], "excerpt": "Dissolved gas analysis trends should be interpreted with rate-of-change context; rapid acetylene or hydrogen rise indicates urgent diagnostics."},
        {"title": "IEEE C57.91 – Transformer loading guide (excerpt)", "tags": ["Electrical", "loading", "temperature"], "excerpt": "Emergency loading above nameplate may be permissible for limited durations if top-oil and winding hot-spot temperatures remain controlled."},
        {"title": "API 610 – Pump vibration and operation guidance (excerpt)", "tags": ["Rotating", "vibration", "noise", "pump"], "excerpt": "Persistent vibration above acceptable limits warrants verification of alignment, balance, and hydraulic operating range before prolonged operation."},
        {"title": "API 579-1/ASME FFS-1 – Fitness-for-service (excerpt)", "tags": ["Process", "leak", "pressure", "wall thinning"], "excerpt": "Assessment levels should match consequence and uncertainty; local metal-loss findings require remaining life evaluation before deferral."},
        {"title": "IEC 61511 – Functional safety considerations (excerpt)", "tags": ["Process", "Electrical", "safety", "trip", "risk"], "excerpt": "Operational changes used as safeguards shall be validated for independence and reliability within the safety lifecycle."},
    ]

    query = f"{subsystem} {suspected_failure_type}".lower()
    scored = []
    for item in library:
        score = sum(1 for tag in item["tags"] if tag.lower() in query)
        if subsystem in item["tags"]:
            score += 1
        scored.append((score, item))

    ranked = [itm for score, itm in sorted(scored, key=lambda x: x[0], reverse=True) if score > 0]
    return (ranked if ranked else library[:2])[:2]


def build_sap_payload(asset: pd.Series, notification_structured: dict, risk_score: float, traffic_light: str, predicted_ttf: float, options_df: pd.DataFrame, standards_refs):
    """Build mock SAP-ready work order JSON payload."""
    return {
        "asset_id": asset["asset_id"],
        "asset_name": asset["asset_name"],
        "subsystem": asset["subsystem"],
        "notification_structured": notification_structured,
        "risk_score": round(float(risk_score), 2),
        "traffic_light_status": traffic_light,
        "predicted_time_to_threshold": round(float(predicted_ttf), 1),
        "recommended_option": options_df.iloc[0].to_dict(),
        "option_evaluations": options_df.to_dict(orient="records"),
        "standards_references": [s["title"] for s in standards_refs],
        "planner_approval_required": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def build_layout_positions(assets_df: pd.DataFrame) -> pd.DataFrame:
    """Mock facility layout coordinates grouped by subsystem zones."""
    zone_x = {"Electrical": 12, "Rotating": 45, "Process": 78}
    rows = []
    for subsystem in ["Electrical", "Rotating", "Process"]:
        subset = assets_df[assets_df["subsystem"] == subsystem].reset_index(drop=True)
        y_vals = np.linspace(15, 85, len(subset))
        for i, r in subset.iterrows():
            rows.append({"asset_id": r["asset_id"], "asset_name": r["asset_name"], "subsystem": subsystem, "x": zone_x[subsystem], "y": float(y_vals[i])})
    return pd.DataFrame(rows)


def cascade_impact(graph: nx.DiGraph, source: str, cutoff: int = 4) -> dict:
    """Compute max propagated impact strength from selected source to downstream assets."""
    impact = {source: 1.0}
    for node in graph.nodes:
        if node == source:
            continue
        max_strength = 0.0
        for path in nx.all_simple_paths(graph, source=source, target=node, cutoff=cutoff):
            weights = [graph[path[i]][path[i + 1]]["propagation_weight"] for i in range(len(path) - 1)]
            strength = float(np.prod(weights))
            max_strength = max(max_strength, strength)
        if max_strength > 0:
            impact[node] = max_strength
    return impact


def traffic_light_text(value: float, green: float, yellow: float) -> str:
    if value < green:
        return "🟢 Green"
    if value < yellow:
        return "🟡 Yellow"
    return "🔴 Red"


def sanitize_chart_df(df: pd.DataFrame, required_cols=None) -> pd.DataFrame:
    """Remove NaN/inf from chart-bound dataframes to avoid Vega-Lite front-end crashes."""
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    if required_cols:
        cols = [c for c in required_cols if c in out.columns]
        if cols:
            out = out.dropna(subset=cols)
    out = out.dropna(how="all")
    return out


# ------------------------------
# App UI
# ------------------------------

def main():
    st.set_page_config(page_title="ORACLE – Maintenance Decision Intelligence", page_icon="🛰️", layout="wide")

    st.markdown(
        """
        <style>
            .oracle-card {padding: 0.65rem 0.9rem; border-radius: 0.8rem; border: 1px solid rgba(49,51,63,0.2); background: rgba(250,250,252,0.8);}
            .oracle-sub {color: #667085; margin-top: -0.4rem; margin-bottom: 0.6rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## ORACLE · Maintenance Decision Intelligence")
    st.markdown("<div class='oracle-sub'>Synthetic decision intelligence demo: Notification → Structuring → Graph Risk → PdM Signals → Prescriptive Action → Standards Rationale → SAP JSON.</div>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("Controls")
        scenario = st.selectbox("Scenario", ["Offshore expensive", "Onshore cheaper"], help="Changes mobilization cost multipliers.")
        demo_mode = st.toggle("Demo Mode (randomize mock data)", value=True)

        st.subheader("Risk Status Thresholds")
        green_threshold = st.slider("Green upper bound (<)", min_value=20, max_value=50, value=35)
        yellow_threshold = st.slider("Yellow upper bound (<)", min_value=55, max_value=85, value=70)
        if green_threshold >= yellow_threshold:
            st.warning("Threshold conflict detected. Reverting to defaults: Green<35, Yellow<70.")
            green_threshold, yellow_threshold = 35, 70

    seed = int(datetime.now().timestamp()) % 100000 if demo_mode else 42

    assets_df = build_assets(scenario)
    graph, priority_df, adjacency_lines = build_graph(assets_df)
    ts_df, health_df = generate_health_timeseries(assets_df, seed=seed)

    model_df = (
        assets_df
        .merge(priority_df[["asset_id", "systemic_priority"]], on="asset_id", how="left")
        .merge(health_df, on="asset_id", how="left")
    )
    model_df["systemic_priority_normalized"] = model_df["systemic_priority"] / 100

    model_df[["risk_score", "anomaly_n"]] = model_df.apply(
        lambda r: pd.Series(compute_risk_score(r["systemic_priority_normalized"], r["current_health"], r["anomaly_score"])),
        axis=1,
    )

    default_idx = int(model_df["risk_score"].idxmax())
    selected_name = st.sidebar.selectbox("Selected Asset", model_df["asset_name"].tolist(), index=default_idx)
    selected_asset = model_df[model_df["asset_name"] == selected_name].iloc[0]

    risk_score = float(selected_asset["risk_score"])
    selected_status = traffic_light_text(risk_score, green_threshold, yellow_threshold)
    facility_risk = float(model_df["risk_score"].mean())
    facility_status = traffic_light_text(facility_risk, green_threshold, yellow_threshold)

    notification_default = (
        f"Operator reports rising vibration and intermittent noise at {selected_name} during high load. "
        "Observed slight temperature increase and occasional pressure fluctuation."
    )
    notification_text = st.sidebar.text_area("Free-text Notification", value=notification_default, height=140)
    parsed_notification = parse_notification(notification_text, selected_asset)

    defer_weeks = st.sidebar.slider("Weeks to defer (Option B)", 1, 12, 4)
    planned_windows = [(datetime.now().date() + timedelta(days=d)).isoformat() for d in (7, 14, 21, 28, 42)]
    planned_window = st.sidebar.selectbox("Planned Window (Option C)", planned_windows)

    st.sidebar.markdown("---")
    st.sidebar.subheader("LLM Runtime")
    online_mode = bool(st.session_state.get("_online_mode", False))

    if online_mode:
        st.sidebar.caption("Online mode: use remote API endpoint (for Streamlit Cloud).")
        use_local_mistral = False
        use_remote_api = st.sidebar.toggle("Use remote Mistral API", value=True)
        remote_api_model = st.sidebar.text_input("API model", value="mistral")
        remote_api_endpoint = st.sidebar.text_input(
            "API endpoint",
            value=st.secrets.get("MISTRAL_API_ENDPOINT", "https://your-api-endpoint/v1/mistral"),
        )
        remote_api_key = st.sidebar.text_input(
            "API key",
            value=st.secrets.get("MISTRAL_API_KEY", ""),
            type="password",
        )
        local_mistral_model = "mistral"
        local_mistral_endpoint = "http://localhost:11434/api/generate"
    else:
        use_local_mistral = st.sidebar.toggle("Use local Mistral (Ollama)", value=False)
        use_remote_api = st.sidebar.toggle("Use remote Mistral API", value=False)
        local_mistral_model = st.sidebar.text_input("Local model", value="mistral")
        local_mistral_endpoint = st.sidebar.text_input("Local endpoint", value="http://localhost:11434/api/generate")
        remote_api_model = st.sidebar.text_input("API model", value="mistral")
        remote_api_endpoint = st.sidebar.text_input("API endpoint", value="")
        remote_api_key = st.sidebar.text_input("API key", value="", type="password")

    options_df = evaluate_options(
        selected_asset,
        risk_score,
        float(selected_asset["predicted_time_to_threshold"]),
        defer_weeks,
        planned_window,
    )
    standards = retrieve_standards_snippets(selected_asset["subsystem"], parsed_notification["suspected_failure_type"])
    sap_payload = build_sap_payload(
        selected_asset,
        parsed_notification,
        risk_score,
        selected_status,
        float(selected_asset["predicted_time_to_threshold"]),
        options_df,
        standards,
    )

    # Global KPI calculations (rendered in Overview only)
    selected_ts_kpi = ts_df[ts_df["asset_id"] == selected_asset["asset_id"]].sort_values("date").reset_index(drop=True)
    latest_row = selected_ts_kpi.iloc[-1]
    prev_row = selected_ts_kpi.iloc[-2] if len(selected_ts_kpi) > 1 else latest_row
    health_delta = float(latest_row["health_index"] - prev_row["health_index"])
    anomaly_delta = float(latest_row["anomaly_score"] - prev_row["anomaly_score"])
    risk_latest, _ = compute_risk_score(
        float(selected_asset["systemic_priority_normalized"]),
        float(latest_row["health_index"]),
        float(latest_row["anomaly_score"]),
    )
    risk_prev, _ = compute_risk_score(
        float(selected_asset["systemic_priority_normalized"]),
        float(prev_row["health_index"]),
        float(prev_row["anomaly_score"]),
    )
    risk_delta = float(risk_latest - risk_prev)

    tabs = st.tabs([
        "Overview",
        "Notification Assist (5W)",
        "Asset Risk Graph",
        "Health & PdM Signals",
        "Decision Orchestration",
        "Standards (RAG) & Explainability",
        "SAP Proposal Export",
    ])

    with tabs[0]:
        st.subheader("Overview")
        st.info("頁面說明：展示整體風險態勢、通知文字結構化結果，讓決策者先快速掌握目前資產狀態。")

        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.markdown(f"**Selected Asset Status:** {selected_status}")
        c2.markdown(f"**Facility Status:** {facility_status}")

        kpi = st.columns(5)
        kpi[0].metric("Current Health Index", f"{latest_row['health_index']:.1f}", delta=f"{health_delta:+.2f}")
        kpi[1].metric("Risk Score", f"{risk_latest:.1f}", delta=f"{risk_delta:+.2f}")
        kpi[2].metric("Predicted Time-to-Threshold (days)", f"{selected_asset['predicted_time_to_threshold']:.1f}")
        kpi[3].metric("Anomaly Score", f"{latest_row['anomaly_score']:.2f}", delta=f"{anomaly_delta:+.2f}")
        kpi[4].metric("Estimated Mobilization Cost", f"${selected_asset['mobilization_cost']:,.0f}")
        st.caption("Summary 使用最新一筆資料；Delta = 最新值 - 倒數第二筆。Risk formula: risk_score = (systemic_priority_normalized*0.5 + (100-current_health)/100*0.3 + anomaly_score_normalized*0.2) * 100")

        overview_cols = ["asset_id", "asset_name", "subsystem", "criticality", "current_health", "anomaly_score", "systemic_priority", "risk_score"]
        risk_rank = model_df[overview_cols].sort_values("risk_score", ascending=False)
        risk_rank_chart = sanitize_chart_df(risk_rank, ["risk_score", "asset_name"])
        with st.expander("查看明細資料表（Asset risk rank）", expanded=False):
            st.dataframe(risk_rank, use_container_width=True, hide_index=True)

        chart = (
            alt.Chart(risk_rank_chart)
            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
            .encode(
                x=alt.X("risk_score:Q", title="Risk Score"),
                y=alt.Y("asset_name:N", sort="-x", title="Asset"),
                color=alt.Color("subsystem:N", title="Subsystem"),
                tooltip=["asset_name", "risk_score", "current_health", "systemic_priority"],
            )
            .properties(height=420)
        )
        st.altair_chart(chart, use_container_width=True)

        st.markdown("#### Notification Structuring")
        st.markdown(f"系統判斷目前通知最可能是 **{parsed_notification['suspected_failure_type']}**，疑似位置為 **{parsed_notification['suspected_component']}**，置信度 **{parsed_notification['confidence']:.2f}**。")
        with st.expander("查看通知結構化原始輸出", expanded=False):
            left, right = st.columns(2)
            left.json(parsed_notification)
            right.dataframe(
                pd.DataFrame({
                    "field": list(parsed_notification.keys()),
                    "value": [str(", ".join(v) if isinstance(v, list) else v) for v in parsed_notification.values()],
                }),
                use_container_width=True,
                hide_index=True,
            )

    with tabs[1]:
        st.subheader("Notification Assist (5W)")
        st.info("頁面說明：協助工人快速輸入有重點的故障描述。先點常見句型，再交由 Mistral（mock）標準化成 5W。")

        templates = notification_templates(selected_asset["subsystem"])
        st.markdown("**常見故障描述推薦（點選可填入）**")
        temp_cols = st.columns(2)
        if "notif_assist_text" not in st.session_state:
            st.session_state["notif_assist_text"] = ""

        for i, t in enumerate(templates):
            if temp_cols[i % 2].button(f"+ {t}", key=f"tpl_{i}"):
                current = st.session_state.get("notif_assist_text", "")
                st.session_state["notif_assist_text"] = (current + " " + t).strip()

        draft_text = st.text_area(
            "Notification Draft",
            value=st.session_state.get("notif_assist_text", ""),
            height=170,
            key="notif_assist_editor",
            help="可編輯推薦文字，送出後產生標準化 5W。",
        )
        st.session_state["notif_assist_text"] = draft_text

        st.markdown("**語音輸入（Beta）**")
        st.caption("可錄音上傳；在無離線 STT 引擎條件下，請於下方輸入語音轉寫文字（或使用模擬轉寫）。")

        if hasattr(st, "audio_input"):
            audio = st.audio_input("按下開始錄音")
        else:
            st.warning("目前 Streamlit 版本不支援 `st.audio_input`，已切換為檔案上傳模式。建議升級 Streamlit。")
            audio = st.file_uploader("上傳語音檔（wav/mp3/m4a）", type=["wav", "mp3", "m4a"], key="audio_upload_fallback")

        voice_transcript = st.text_input("語音轉寫文字", value="", key="voice_transcript_text")
        c_voice1, c_voice2 = st.columns(2)
        if c_voice1.button("使用語音轉寫覆蓋草稿"):
            if voice_transcript.strip():
                st.session_state["notif_assist_text"] = voice_transcript.strip()
                st.rerun()
            else:
                st.warning("請先輸入語音轉寫文字。")
        if c_voice2.button("使用模擬轉寫"):
            mock_text = f"Operator voice note: vibration increased on {selected_name} during high load, please inspect soon."
            st.session_state["notif_assist_text"] = mock_text
            st.rerun()

        if audio is not None:
            st.success("已收到音訊檔（語音輸入成功）。")

        if st.button("送出進行 5W 標準化", type="primary"):
            user_note = st.session_state.get("notif_assist_text", "")
            if use_local_mistral:
                ok, result_5w, err = call_local_mistral_5w(
                    user_note,
                    selected_name,
                    selected_asset["subsystem"],
                    endpoint=local_mistral_endpoint,
                    model=local_mistral_model,
                )
                if not ok:
                    st.warning(f"本機 Mistral 呼叫失敗，改用 mock 流程。原因: {err}")
                    result_5w = mock_mistral_5w(user_note, selected_name, selected_asset["subsystem"])
            elif use_remote_api and remote_api_endpoint.strip():
                ok, result_5w, err = call_remote_mistral_5w(
                    user_note,
                    selected_name,
                    selected_asset["subsystem"],
                    endpoint=remote_api_endpoint.strip(),
                    model=remote_api_model,
                    api_key=remote_api_key,
                )
                if not ok:
                    st.warning(f"Remote API 呼叫失敗，改用 mock 流程。原因: {err}")
                    result_5w = mock_mistral_5w(user_note, selected_name, selected_asset["subsystem"])
            else:
                result_5w = mock_mistral_5w(user_note, selected_name, selected_asset["subsystem"])

            st.markdown("#### 標準化 5W 結果")
            st.json(result_5w)
            fivew_df = pd.DataFrame(
                {
                    "item": ["WHAT", "WHEN", "WHERE", "WHO", "WHY", "MODEL"],
                    "content": [
                        str(result_5w["what"]),
                        str(result_5w["when"]),
                        str(result_5w["where"]),
                        str(result_5w["who"]),
                        str(result_5w["why"]),
                        str(result_5w.get("llm_model", "mock")),
                    ],
                }
            )
            st.dataframe(fivew_df, use_container_width=True, hide_index=True)
            st.code(result_5w["standardized_5w"], language="text")

    with tabs[2]:
        st.subheader("Asset Risk Graph")
        st.info("頁面說明：先看廠務 layout 與各資產健康分數，再模擬單台健康下滑對下游影響台數的變化。")

        layout_df = build_layout_positions(assets_df).merge(
            model_df[["asset_id", "asset_name", "subsystem", "risk_score", "current_health"]],
            on=["asset_id", "asset_name", "subsystem"],
            how="left",
        )
        layout_df = sanitize_chart_df(layout_df, ["x", "y", "asset_name", "current_health", "subsystem"])

        st.markdown("#### Facility Layout（顯示當前健康分數）")
        if layout_df.empty:
            st.warning("No valid layout data available.")
        else:
            layout_chart = (
                alt.Chart(layout_df)
                .mark_circle(stroke="white", strokeWidth=1)
                .encode(
                    x=alt.X("x:Q", title="Facility Zone X"),
                    y=alt.Y("y:Q", title="Facility Zone Y"),
                    size=alt.Size("current_health:Q", scale=alt.Scale(range=[120, 950]), title="Current Health"),
                    color=alt.Color("current_health:Q", title="Health Index", scale=alt.Scale(scheme="redyellowgreen")),
                    shape=alt.Shape("subsystem:N", title="Subsystem"),
                    tooltip=["asset_name", "asset_id", "subsystem", alt.Tooltip("current_health:Q", title="Health")],
                )
                .properties(height=330)
            )
            labels = (
                alt.Chart(layout_df)
                .mark_text(dy=-12, fontSize=11)
                .encode(x="x:Q", y="y:Q", text=alt.Text("current_health:Q", format=".1f"))
            )
            st.altair_chart((layout_chart + labels), use_container_width=True)

        st.markdown("#### Impact Simulator（點選資產 + 健康拉霸）")
        sim_asset_name = st.selectbox(
            "選擇資產（模擬點擊）",
            model_df["asset_name"].tolist(),
            index=int(model_df[model_df["asset_id"] == selected_asset["asset_id"]].index[0]),
            key="impact_sim_asset",
        )
        sim_asset = model_df.loc[model_df["asset_name"] == sim_asset_name].iloc[0]
        impact_base = cascade_impact(graph, sim_asset["asset_id"], cutoff=4)
        base_impacted_count = max(len(impact_base) - 1, 0)

        current_h = float(sim_asset["current_health"])
        sim_health = st.slider(
            "模擬健康分數（往下拉看影響變化）",
            min_value=0.0,
            max_value=100.0,
            value=float(round(current_h, 1)),
            step=0.1,
            key="impact_health_slider",
        )
        health_ratio = 0.0 if current_h <= 0 else float(np.clip(sim_health / current_h, 0, 1.5))

        impacted_sim = int(round(base_impacted_count * health_ratio))
        impacted_sim = max(0, min(base_impacted_count, impacted_sim))

        c_imp1, c_imp2, c_imp3 = st.columns(3)
        c_imp1.metric("Base impacted assets", f"{base_impacted_count}")
        c_imp2.metric("Simulated impacted assets", f"{impacted_sim}", delta=f"{impacted_sim - base_impacted_count:+d}")
        c_imp3.metric("Simulated health", f"{sim_health:.1f}", delta=f"{sim_health - current_h:+.1f}")

        st.markdown(
            f"資產 **{sim_asset_name}** 目前健康 **{current_h:.1f}**。當健康下滑至 **{sim_health:.1f}** 時，"
            f"預估可影響台數由 **{base_impacted_count}** 下降為 **{impacted_sim}**。"
        )

        impact_table = layout_df[["asset_id", "asset_name", "subsystem"]].copy()
        impact_table["impact_strength_base"] = impact_table["asset_id"].map(impact_base).fillna(0.0)
        impact_table = impact_table[impact_table["impact_strength_base"] > 0].copy()
        impact_table["impact_strength_simulated"] = impact_table["impact_strength_base"] * health_ratio
        impact_table["risk_score"] = impact_table["asset_id"].map(model_df.set_index("asset_id")["risk_score"]).fillna(0.0)
        impact_table = impact_table.sort_values("impact_strength_simulated", ascending=False)

        impact_chart_df = sanitize_chart_df(
            impact_table,
            ["asset_name", "impact_strength_simulated", "impact_strength_base"],
        )

        if impact_chart_df.empty:
            st.warning("No valid cascade-impact data available for this asset.")
        else:
            comp = impact_chart_df.melt(
                id_vars=["asset_name"],
                value_vars=["impact_strength_base", "impact_strength_simulated"],
                var_name="scenario",
                value_name="strength",
            )
            comp["scenario"] = comp["scenario"].map(
                {
                    "impact_strength_base": "Base",
                    "impact_strength_simulated": "Simulated",
                }
            )
            impact_compare = (
                alt.Chart(comp)
                .mark_bar()
                .encode(
                    x=alt.X("strength:Q", title="Cascade Impact Strength"),
                    y=alt.Y("asset_name:N", sort="-x", title="Downstream Asset"),
                    color=alt.Color("scenario:N", title="Scenario"),
                    tooltip=["asset_name", "scenario", "strength"],
                )
                .properties(height=320)
            )
            st.altair_chart(impact_compare, use_container_width=True)

        with st.expander("查看 systemic priority 與 adjacency 詳細資料", expanded=False):
            st.dataframe(
                priority_df[["asset_id", "asset_name", "out_degree", "betweenness", "systemic_priority"]],
                use_container_width=True,
                hide_index=True,
            )
            st.dataframe(
                impact_table[["asset_id", "asset_name", "subsystem", "impact_strength_base", "impact_strength_simulated"]],
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("**Adjacency List (with propagation weights)**")
            st.code("\n".join(adjacency_lines), language="text")

    with tabs[3]:
        st.subheader("Health & PdM Signals")
        st.info("頁面說明：查看 90 天健康趨勢、異常分數和操作模式變化，輔助預估達到門檻的剩餘天數。")
        st.markdown(f"目前 **{selected_name}** 健康值最新為 **{latest_row['health_index']:.1f}**，最近一天變化 **{health_delta:+.2f}**；預估到達門檻剩餘 **{selected_asset['predicted_time_to_threshold']:.1f}** 天。")

        asset_ts = ts_df[ts_df["asset_id"] == selected_asset["asset_id"]].sort_values("date").reset_index(drop=True).copy()
        asset_ts = sanitize_chart_df(asset_ts, ["date", "health_index", "anomaly_score"])
        threshold = 60

        st.caption("可用滑桿模擬時間推進，觀察單一機台健康值下降速度；虛線為二次擬合趨勢。")
        min_sim = 3 if len(asset_ts) >= 3 else 1
        default_sim = len(asset_ts) if len(asset_ts) > 0 else 1
        sim_day = st.slider("Simulation Day (time progression)", min_value=min_sim, max_value=default_sim, value=default_sim, step=1)
        sim_ts = asset_ts.iloc[:sim_day].copy()
        sim_ts = sanitize_chart_df(sim_ts, ["date", "health_index", "anomaly_score"])
        if sim_ts.empty:
            st.warning("No data available for the selected range.")
            sim_ts = asset_ts.copy()
        sim_ts["t_idx"] = np.arange(len(sim_ts))
        if len(sim_ts) >= 3:
            coef = np.polyfit(sim_ts["t_idx"], sim_ts["health_index"], 2)
            sim_ts["health_quad_fit"] = np.polyval(coef, sim_ts["t_idx"])
        else:
            sim_ts["health_quad_fit"] = sim_ts["health_index"]
        sim_ts_clean = sanitize_chart_df(sim_ts, ["date", "health_index", "health_quad_fit", "anomaly_score"])
        if sim_ts_clean.empty:
            st.warning("No chart-ready data available for plotting.")
        else:
            health_line = (
                alt.Chart(sim_ts_clean)
                .mark_line(point=False, strokeWidth=2)
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("health_index:Q", title="Health Index", scale=alt.Scale(domain=[0, 100])),
                    color=alt.value("#1f77b4"),
                    tooltip=["date:T", "health_index:Q", "operating_mode:N"],
                )
            )
            fit_line = (
                alt.Chart(sim_ts_clean)
                .mark_line(strokeDash=[8, 5], strokeWidth=2, color="#2ca02c")
                .encode(
                    x="date:T",
                    y=alt.Y("health_quad_fit:Q", title="Health Index"),
                    tooltip=["date:T", alt.Tooltip("health_quad_fit:Q", title="Quadratic Fit")],
                )
            )
            threshold_line = (
                alt.Chart(sanitize_chart_df(pd.DataFrame({"y": [threshold]}), ["y"]))
                .mark_rule(color="red", strokeDash=[6, 5])
                .encode(y="y:Q")
            )
            st.altair_chart((health_line + fit_line + threshold_line).properties(height=340).interactive(), use_container_width=True)

            anomaly = (
                alt.Chart(sim_ts_clean)
                .mark_area(opacity=0.35, color="#ff7f0e")
                .encode(x="date:T", y=alt.Y("anomaly_score:Q", title="Anomaly Score"), tooltip=["date:T", "anomaly_score:Q"])
                .properties(height=180)
            )
            st.altair_chart(anomaly, use_container_width=True)

        with st.expander("查看最近 10 筆健康訊號資料", expanded=False):
            st.dataframe(
                sim_ts[["date", "operating_mode", "health_index", "health_quad_fit", "anomaly_score"]].tail(10).sort_values("date", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

    with tabs[4]:
        st.subheader("Decision Orchestration")
        st.info("頁面說明：比較 4 種處置策略在風險降低、停機、成本與殘餘風險上的綜合分數，給出推薦方案。")
        st.markdown(f"依目前風險條件，建議策略傾向 **{options_df.iloc[0]['option']}**，可在成本與殘餘風險間取得較佳平衡。")

        with st.expander("查看策略評分明細", expanded=False):
            st.dataframe(options_df, use_container_width=True, hide_index=True)
        options_df_chart = sanitize_chart_df(options_df, ["option", "decision_score", "residual_risk"])

        score_chart = (
            alt.Chart(options_df_chart)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("decision_score:Q", title="Decision Score"),
                y=alt.Y("option:N", sort="-x", title="Intervention Option"),
                color=alt.value("#1f77b4"),
                tooltip=["option", "decision_score", "risk_reduction", "mobilization_cost"],
            )
            .properties(height=280, title="Decision Score")
        )

        residual_chart = (
            alt.Chart(options_df_chart)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("residual_risk:Q", title="Residual Risk"),
                y=alt.Y("option:N", sort="-x", title="Intervention Option"),
                color=alt.value("#ff7f0e"),
                tooltip=["option", "residual_risk", "expected_downtime_hours"],
            )
            .properties(height=280, title="Residual Risk")
        )

        c_score, c_res = st.columns(2)
        c_score.altair_chart(score_chart, use_container_width=True)
        c_res.altair_chart(residual_chart, use_container_width=True)

        best = options_df.iloc[0]
        rec_light = traffic_light_text(float(best["residual_risk"]), green_threshold, yellow_threshold).split()[0]
        st.success(f"Recommended Option: {best['option']} {rec_light} · Decision Score {best['decision_score']}")

    with tabs[5]:
        st.subheader("Standards (RAG) & Explainability")
        st.info("頁面說明：依資產與疑似故障型態擷取標準片段，並說明推薦決策與風險訊號之間的因果關聯。")

        st.markdown("#### Cited Guidance")
        st.markdown("系統已根據子系統與故障假設自動挑選最相關的 1–2 條標準節錄。")
        with st.expander("查看引用標準內容", expanded=False):
            for snip in standards:
                st.markdown(f"**{snip['title']}**")
                st.write(snip["excerpt"])

        st.markdown("#### Explainability")
        explain = (
            f"Recommended strategy **{options_df.iloc[0]['option']}** is selected because {selected_name} has "
            f"systemic priority **{selected_asset['systemic_priority']:.1f}**, current health **{selected_asset['current_health']:.1f}**, "
            f"and anomaly score **{selected_asset['anomaly_score']:.2f}** resulting in risk score **{risk_score:.1f}**. "
            f"Notification parser indicates **{parsed_notification['suspected_failure_type']}** on "
            f"**{parsed_notification['suspected_component']}** (confidence {parsed_notification['confidence']:.2f}). "
            f"Retrieved standards constrain excessive deferral and support options with lower residual risk."
        )
        st.info(explain)

        factors_df = pd.DataFrame(
            {
                "factor": ["Systemic Priority", "Health Degradation", "Anomaly", "Notification Confidence"],
                "value": [
                    round(float(selected_asset["systemic_priority"]), 1),
                    round(float(100 - selected_asset["current_health"]), 1),
                    round(float(selected_asset["anomaly_score"] * 20), 1),
                    round(float(parsed_notification["confidence"] * 100), 1),
                ],
            }
        )
        factors_df_chart = sanitize_chart_df(factors_df, ["factor", "value"])
        factor_chart = (
            alt.Chart(factors_df_chart)
            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
            .encode(x="factor:N", y="value:Q", color="factor:N", tooltip=["factor", "value"])
            .properties(height=260)
        )
        st.altair_chart(factor_chart, use_container_width=True)

    with tabs[6]:
        st.subheader("SAP Proposal Export")
        st.info("頁面說明：輸出可交給 planner 審核的工單提案 JSON，含通知結構化內容、風險與推薦方案。")

        st.json(sap_payload)
        st.download_button(
            label="Download JSON",
            data=json.dumps(sap_payload, indent=2),
            file_name=f"oracle_work_order_{selected_asset['asset_id']}.json",
            mime="application/json",
        )

        preview = pd.DataFrame(
            {
                "key": [
                    "asset_id", "asset_name", "risk_score", "traffic_light_status",
                    "predicted_time_to_threshold", "recommended_option", "planner_approval_required",
                ],
                "value": [
                    str(sap_payload["asset_id"]), str(sap_payload["asset_name"]), str(sap_payload["risk_score"]),
                    str(sap_payload["traffic_light_status"]), str(sap_payload["predicted_time_to_threshold"]),
                    str(sap_payload["recommended_option"]["option"]), str(sap_payload["planner_approval_required"]),
                ],
            }
        )
        with st.expander("查看 proposal key-value 明細", expanded=False):
            st.dataframe(preview, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
