import json
from datetime import datetime, timedelta

import altair as alt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st


# ------------------------------
# Data Builders
# ------------------------------

def build_assets(scenario: str = "Offshore expensive") -> pd.DataFrame:
    """Create a mock asset register with realistic maintenance attributes."""
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
        offshore_multiplier = 0.85
        onshore_multiplier = 0.70
    else:
        offshore_multiplier = 1.25
        onshore_multiplier = 1.00

    multipliers = np.where(df["location"] == "Offshore", offshore_multiplier, onshore_multiplier)
    df["mobilization_cost"] = (df["mobilization_cost"] * multipliers).round(0).astype(int)
    return df



def build_graph(assets_df: pd.DataFrame):
    """Construct a directed dependency graph and derive systemic priority metrics."""
    graph = nx.DiGraph()
    for _, row in assets_df.iterrows():
        graph.add_node(
            row["asset_id"],
            asset_name=row["asset_name"],
            subsystem=row["subsystem"],
            criticality=row["criticality"],
        )

    edges = [
        ("T1", "C1", 0.90),
        ("T1", "C2", 0.65),
        ("C1", "S1", 0.85),
        ("C2", "S1", 0.70),
        ("S1", "HX1", 0.75),
        ("HX1", "S2", 0.55),
        ("S2", "F1", 0.60),
        ("P1", "S2", 0.40),
        ("P2", "S1", 0.45),
        ("TR1", "SW1", 0.85),
        ("TR2", "SW1", 0.55),
        ("SW1", "T1", 0.80),
        ("SW1", "C1", 0.60),
        ("SW2", "P1", 0.50),
        ("SW2", "P2", 0.55),
        ("V1", "C1", 0.50),
        ("V2", "S1", 0.35),
        ("F1", "V2", 0.30),
    ]
    for src, dst, weight in edges:
        graph.add_edge(src, dst, propagation_weight=weight)

    out_degree = dict(graph.out_degree(weight="propagation_weight"))
    betweenness = nx.betweenness_centrality(graph, weight="propagation_weight", normalized=True)

    metrics = assets_df[["asset_id", "asset_name", "criticality"]].copy()
    metrics["out_degree"] = metrics["asset_id"].map(out_degree).fillna(0)
    metrics["betweenness"] = metrics["asset_id"].map(betweenness).fillna(0)

    def normalize(series: pd.Series) -> pd.Series:
        rng = series.max() - series.min()
        if rng < 1e-9:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - series.min()) / rng

    metrics["out_degree_n"] = normalize(metrics["out_degree"])
    metrics["betweenness_n"] = normalize(metrics["betweenness"])
    metrics["criticality_n"] = normalize(metrics["criticality"])
    metrics["systemic_priority"] = (
        0.45 * metrics["out_degree_n"]
        + 0.35 * metrics["betweenness_n"]
        + 0.20 * metrics["criticality_n"]
    ) * 100
    metrics = metrics.sort_values("systemic_priority", ascending=False).reset_index(drop=True)

    adjacency_lines = []
    for node in graph.nodes:
        neighbors = list(graph.successors(node))
        if neighbors:
            weighted = [f"{n}({graph[node][n]['propagation_weight']:.2f})" for n in neighbors]
            adjacency_lines.append(f"{node} -> " + ", ".join(weighted))
        else:
            adjacency_lines.append(f"{node} -> [no downstream dependencies]")

    return graph, metrics, adjacency_lines



def generate_health_timeseries(assets_df: pd.DataFrame, seed: int = 42, days: int = 90):
    """Generate synthetic 90-day health and anomaly trends per asset with mode effects."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(datetime.now().date() - timedelta(days=days - 1), periods=days, freq="D")
    modes = np.array(["Normal", "High Load", "Start-Stop"])
    mode_shift = {"Normal": 0.0, "High Load": -3.5, "Start-Stop": -1.5}

    all_rows = []
    summary_rows = []

    for _, asset in assets_df.iterrows():
        subsystem = asset["subsystem"]
        base = rng.uniform(70, 95)
        noise = rng.normal(0, 1.8, days)
        mode_series = rng.choice(modes, size=days, p=[0.60, 0.25, 0.15])

        if subsystem == "Rotating":
            trend = np.linspace(0, rng.uniform(8, 24), days)
            step = np.zeros(days)
            step_start = rng.integers(50, 75)
            step[step_start:] = rng.uniform(4, 10)
            degradation = trend + step
        elif subsystem == "Electrical":
            degradation = np.linspace(0, rng.uniform(5, 13), days)
        else:  # Process
            cyclical = 2.2 * np.sin(np.linspace(0, 5 * np.pi, days))
            trend = np.linspace(0, rng.uniform(3, 10), days)
            degradation = trend - cyclical

        health = []
        for i in range(days):
            mode_adj = mode_shift[mode_series[i]]
            h = base - degradation[i] + mode_adj + noise[i]
            health.append(np.clip(h, 5, 100))

        df_asset = pd.DataFrame(
            {
                "date": dates,
                "asset_id": asset["asset_id"],
                "asset_name": asset["asset_name"],
                "subsystem": subsystem,
                "operating_mode": mode_series,
                "health_index": np.round(health, 2),
            }
        )
        df_asset["rolling_mean_7d"] = df_asset["health_index"].rolling(7, min_periods=3).mean()
        rolling_std = df_asset["health_index"].rolling(7, min_periods=3).std().fillna(1.5).replace(0, 1.5)
        df_asset["anomaly_score"] = (
            (df_asset["rolling_mean_7d"] - df_asset["health_index"]).abs() / rolling_std
        ).fillna(0)

        current_health = float(df_asset["health_index"].iloc[-1])
        recent = df_asset.tail(14)
        if len(recent) > 1:
            slope = float(np.polyfit(np.arange(len(recent)), recent["health_index"], 1)[0])
        else:
            slope = 0.0
        anomaly_current = float(df_asset["anomaly_score"].iloc[-1])

        threshold = 60
        if slope < -0.05:
            est_days = (threshold - current_health) / slope
            predicted_days = max(0.0, est_days)
            predicted_days = float(np.clip(predicted_days, 0, 365))
        elif current_health <= threshold:
            predicted_days = 0.0
        else:
            predicted_days = 365.0

        summary_rows.append(
            {
                "asset_id": asset["asset_id"],
                "current_health": round(current_health, 2),
                "slope_14d": round(slope, 3),
                "anomaly_score": round(anomaly_current, 3),
                "predicted_time_to_threshold": round(predicted_days, 1),
            }
        )
        all_rows.append(df_asset)

    ts_df = pd.concat(all_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    return ts_df, summary_df


# ------------------------------
# Inference & Scoring Helpers
# ------------------------------

def parse_notification(text: str, selected_asset: pd.Series) -> dict:
    """Rule-based parser for free-text maintenance notifications."""
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



def compute_risk_score(systemic_priority_norm: float, current_health: float, anomaly_score: float):
    """Compute risk score using the transparent formula required for the demo."""
    anomaly_n = float(np.clip(anomaly_score / 5.0, 0, 1))
    risk_score = (
        systemic_priority_norm * 0.5
        + ((100 - current_health) / 100.0) * 0.3
        + anomaly_n * 0.2
    ) * 100
    return float(np.clip(risk_score, 0, 100)), anomaly_n



def evaluate_options(
    asset: pd.Series,
    risk_score: float,
    predicted_ttf: float,
    defer_weeks: int,
    planned_window: str,
):
    """Create four intervention strategy options with synthetic scoring."""
    base_cost = float(asset["mobilization_cost"])
    criticality_factor = asset["criticality"] / 10

    options = [
        {
            "option": "Immediate Repair",
            "risk_reduction": np.clip(70 + 20 * criticality_factor, 0, 100),
            "expected_downtime_hours": 16 + 8 * criticality_factor,
            "mobilization_cost": base_cost * 1.15,
        },
        {
            "option": "Defer to Next Window",
            "risk_reduction": np.clip(28 + 3 * defer_weeks, 0, 75),
            "expected_downtime_hours": 8 + 2 * defer_weeks,
            "mobilization_cost": base_cost * (0.88 + 0.02 * defer_weeks),
        },
        {
            "option": "Merge with Planned Maintenance",
            "risk_reduction": np.clip(55 + 8 * criticality_factor, 0, 90),
            "expected_downtime_hours": 12 + 4 * criticality_factor,
            "mobilization_cost": base_cost * 0.82,
        },
        {
            "option": "Operational Mitigation",
            "risk_reduction": np.clip(35 + 10 * (1 - criticality_factor), 0, 65),
            "expected_downtime_hours": 4 + 2 * (1 - criticality_factor),
            "mobilization_cost": base_cost * 0.58,
        },
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

    options_df = pd.DataFrame(rows).sort_values("decision_score", ascending=False).reset_index(drop=True)
    return options_df



def retrieve_standards_snippets(subsystem: str, suspected_failure_type: str):
    """Simple in-code standards retrieval based on subsystem and failure hints."""
    library = [
        {
            "title": "IEEE C57.104 – Transformer DGA interpretation (excerpt)",
            "tags": ["Electrical", "thermal", "insulation", "temperature"],
            "excerpt": "Dissolved gas analysis trends should be interpreted with rate-of-change context; rapid acetylene or hydrogen rise indicates urgent diagnostics.",
        },
        {
            "title": "IEEE C57.91 – Transformer loading guide (excerpt)",
            "tags": ["Electrical", "loading", "temperature"],
            "excerpt": "Emergency loading above nameplate may be permissible for limited durations if top-oil and winding hot-spot temperatures remain controlled.",
        },
        {
            "title": "API 610 – Pump vibration and operation guidance (excerpt)",
            "tags": ["Rotating", "vibration", "noise", "pump"],
            "excerpt": "Persistent vibration above acceptable limits warrants verification of alignment, balance, and hydraulic operating range before prolonged operation.",
        },
        {
            "title": "API 579-1/ASME FFS-1 – Fitness-for-service (excerpt)",
            "tags": ["Process", "leak", "pressure", "wall thinning"],
            "excerpt": "Assessment levels should match consequence and uncertainty; local metal-loss findings require remaining life evaluation before deferral.",
        },
        {
            "title": "IEC 61511 – Functional safety considerations (excerpt)",
            "tags": ["Process", "Electrical", "safety", "trip", "risk"],
            "excerpt": "Operational changes used as safeguards shall be validated for independence and reliability within the safety lifecycle.",
        },
    ]

    query = f"{subsystem} {suspected_failure_type}".lower()
    scored = []
    for item in library:
        score = 0
        for tag in item["tags"]:
            if tag.lower() in query:
                score += 1
        if subsystem in item["tags"]:
            score += 1
        scored.append((score, item))

    ranked = [item for score, item in sorted(scored, key=lambda x: x[0], reverse=True) if score > 0]
    if len(ranked) == 0:
        ranked = library[:2]
    return ranked[:2]



def build_sap_payload(
    asset: pd.Series,
    notification_structured: dict,
    risk_score: float,
    traffic_light: str,
    predicted_ttf: float,
    options_df: pd.DataFrame,
    standards_refs,
):
    """Build a synthetic SAP work order proposal payload."""
    recommended = options_df.iloc[0].to_dict()
    payload = {
        "asset_id": asset["asset_id"],
        "asset_name": asset["asset_name"],
        "subsystem": asset["subsystem"],
        "notification_structured": notification_structured,
        "risk_score": round(float(risk_score), 2),
        "traffic_light_status": traffic_light,
        "predicted_time_to_threshold": round(float(predicted_ttf), 1),
        "recommended_option": recommended,
        "option_evaluations": options_df.to_dict(orient="records"),
        "standards_references": [s["title"] for s in standards_refs],
        "planner_approval_required": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    return payload


# ------------------------------
# Streamlit App
# ------------------------------

def main():
    st.set_page_config(page_title="ORACLE – Maintenance Decision Intelligence", layout="wide")

    st.markdown("## ORACLE · Maintenance Decision Intelligence")
    st.caption(
        "Research-grade prototype using synthetic data: from free-text notification to risk-aware prescriptive maintenance and SAP-ready proposal export."
    )

    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Go to page",
            [
                "Overview",
                "Asset Risk Graph",
                "Health & PdM Signals",
                "Decision Orchestration",
                "Standards (RAG) & Explainability",
                "SAP Proposal Export",
            ],
        )

        st.header("Scenario Controls")
        scenario = st.selectbox("Scenario", ["Offshore expensive", "Onshore cheaper"])
        demo_mode = st.toggle("Demo Mode (randomize mock data)", value=True)

        st.subheader("Traffic Light Thresholds")
        green_threshold = st.slider("Green upper bound (<)", min_value=20, max_value=50, value=35)
        yellow_threshold = st.slider("Yellow upper bound (<)", min_value=55, max_value=85, value=70)
        if green_threshold >= yellow_threshold:
            st.warning("Green threshold should be lower than yellow threshold. Using defaults 35/70.")
            green_threshold, yellow_threshold = 35, 70

    seed = int(datetime.now().timestamp()) % 100000 if demo_mode else 42

    assets_df = build_assets(scenario=scenario)
    graph, priority_df, adjacency_lines = build_graph(assets_df)
    ts_df, health_summary_df = generate_health_timeseries(assets_df, seed=seed)

    model_df = (
        assets_df.merge(priority_df[["asset_id", "systemic_priority"]], on="asset_id", how="left")
        .merge(health_summary_df, on="asset_id", how="left")
    )
    model_df["systemic_priority_normalized"] = model_df["systemic_priority"] / 100

    default_asset_idx = int(model_df["systemic_priority"].idxmax())
    selected_asset_name = st.sidebar.selectbox("Select Asset", model_df["asset_name"].tolist(), index=default_asset_idx)
    selected_asset = model_df.loc[model_df["asset_name"] == selected_asset_name].iloc[0]

    risk_score, anomaly_n = compute_risk_score(
        selected_asset["systemic_priority_normalized"],
        selected_asset["current_health"],
        selected_asset["anomaly_score"],
    )

    def traffic_light(v: float):
        if v < green_threshold:
            return "🟢 Green"
        if v < yellow_threshold:
            return "🟡 Yellow"
        return "🔴 Red"

    model_df[["risk_score", "anomaly_n"]] = model_df.apply(
        lambda r: pd.Series(
            compute_risk_score(r["systemic_priority"] / 100, r["current_health"], r["anomaly_score"])
        ),
        axis=1,
    )

    facility_risk = float(model_df["risk_score"].mean())
    selected_status = traffic_light(risk_score)
    facility_status = traffic_light(facility_risk)

    st.markdown("---")
    col1, col2 = st.columns([2, 2])
    col1.markdown(f"**Selected Asset Status:** {selected_status}")
    col2.markdown(f"**Facility Status:** {facility_status}")

    metric_cols = st.columns(5)
    metric_cols[0].metric("Current Health Index", f"{selected_asset['current_health']:.1f}")
    metric_cols[1].metric("Risk Score", f"{risk_score:.1f}")
    metric_cols[2].metric("Predicted Time-to-Threshold (days)", f"{selected_asset['predicted_time_to_threshold']:.1f}")
    metric_cols[3].metric("Estimated Mobilization Cost", f"${selected_asset['mobilization_cost']:,.0f}")
    metric_cols[4].metric("Estimated Risk Reduction (mock)", f"{min(95, 35 + selected_asset['criticality'] * 5):.0f}%")

    st.caption(
        "Risk formula: risk_score = (systemic_priority_normalized*0.5 + (100-current_health)/100*0.3 + anomaly_score_normalized*0.2)*100"
    )

    notification_default = (
        f"Operator reports increasing vibration and intermittent noise at {selected_asset_name} during high load. "
        "Observed slight temperature rise and occasional pressure fluctuation."
    )
    notification_text = st.sidebar.text_area("Free-text Notification", value=notification_default, height=150)
    parsed_notification = parse_notification(notification_text, selected_asset)

    defer_weeks = st.sidebar.slider("Weeks to defer (for option B)", min_value=1, max_value=12, value=4)
    planned_windows = [
        (datetime.now().date() + timedelta(days=d)).isoformat() for d in (7, 14, 21, 28, 42)
    ]
    planned_window = st.sidebar.selectbox("Planned maintenance window (for option C)", planned_windows)

    options_df = evaluate_options(
        selected_asset, risk_score, selected_asset["predicted_time_to_threshold"], defer_weeks, planned_window
    )
    standards = retrieve_standards_snippets(selected_asset["subsystem"], parsed_notification["suspected_failure_type"])
    sap_payload = build_sap_payload(
        selected_asset,
        parsed_notification,
        risk_score,
        selected_status,
        selected_asset["predicted_time_to_threshold"],
        options_df,
        standards,
    )

    # ------------------------------
    # Pages
    # ------------------------------
    if page == "Overview":
        st.subheader("Facility Overview")
        st.caption("Synthetic assets, derived risk posture, and parsed notification snapshot.")

        top_table = model_df[
            [
                "asset_id",
                "asset_name",
                "subsystem",
                "criticality",
                "current_health",
                "anomaly_score",
                "systemic_priority",
                "risk_score",
            ]
        ].sort_values("risk_score", ascending=False)
        st.dataframe(top_table, use_container_width=True, hide_index=True)

        risk_chart = (
            alt.Chart(top_table)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("risk_score:Q", title="Risk Score"),
                y=alt.Y("asset_name:N", sort="-x", title="Asset"),
                color=alt.Color("subsystem:N"),
                tooltip=["asset_name", "risk_score", "current_health", "systemic_priority"],
            )
            .properties(height=420)
        )
        st.altair_chart(risk_chart, use_container_width=True)

        st.subheader("Notification Structuring")
        c1, c2 = st.columns([1, 1])
        c1.json(parsed_notification)
        parsed_table = pd.DataFrame(
            {
                "Field": list(parsed_notification.keys()),
                "Value": [
                    ", ".join(v) if isinstance(v, list) else v for v in parsed_notification.values()
                ],
            }
        )
        c2.dataframe(parsed_table, use_container_width=True, hide_index=True)

    elif page == "Asset Risk Graph":
        st.subheader("Dependency Graph & Systemic Priority")
        st.caption("Directed graph captures cascade impact from upstream failures to downstream consequences.")

        st.dataframe(
            priority_df[["asset_id", "asset_name", "out_degree", "betweenness", "systemic_priority"]],
            use_container_width=True,
            hide_index=True,
        )

        top_n = st.slider("Top N assets by systemic priority", min_value=5, max_value=len(priority_df), value=10)
        chart_df = priority_df.head(top_n)
        p_chart = (
            alt.Chart(chart_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("systemic_priority:Q", title="Systemic Priority"),
                y=alt.Y("asset_name:N", sort="-x", title="Asset"),
                color=alt.Color("systemic_priority:Q", scale=alt.Scale(scheme="redyellowgreen", reverse=True)),
                tooltip=["asset_name", "out_degree", "betweenness", "systemic_priority"],
            )
            .properties(height=420)
        )
        st.altair_chart(p_chart, use_container_width=True)

        st.markdown("**Adjacency List (with propagation weights)**")
        st.code("\n".join(adjacency_lines), language="text")

    elif page == "Health & PdM Signals":
        st.subheader(f"Health & Condition Signals · {selected_asset_name}")
        st.caption("90-day synthetic health index with operating context, anomaly signal, and threshold forecast.")

        asset_ts = ts_df[ts_df["asset_id"] == selected_asset["asset_id"]].copy()
        threshold = 60

        health_line = (
            alt.Chart(asset_ts)
            .mark_line(point=False)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("health_index:Q", title="Health Index (0-100)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("operating_mode:N", title="Operating Mode"),
                tooltip=["date:T", "health_index:Q", "operating_mode:N"],
            )
            .properties(height=320)
        )
        threshold_df = pd.DataFrame({"y": [threshold]})
        threshold_rule = alt.Chart(threshold_df).mark_rule(strokeDash=[8, 6], color="red").encode(y="y:Q")
        st.altair_chart((health_line + threshold_rule).interactive(), use_container_width=True)

        anomaly_chart = (
            alt.Chart(asset_ts)
            .mark_area(opacity=0.35)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("anomaly_score:Q", title="Anomaly Score"),
                tooltip=["date:T", "anomaly_score:Q"],
            )
            .properties(height=180)
        )
        st.altair_chart(anomaly_chart, use_container_width=True)

        st.dataframe(
            asset_ts[["date", "operating_mode", "health_index", "anomaly_score"]]
            .tail(10)
            .sort_values("date", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    elif page == "Decision Orchestration":
        st.subheader("Prescriptive Decision Orchestration")
        st.caption("Four intervention paths evaluated with synthetic risk, cost, downtime, and residual risk effects.")

        st.dataframe(options_df, use_container_width=True, hide_index=True)

        comp = options_df.melt(
            id_vars=["option"], value_vars=["decision_score", "residual_risk"],
            var_name="metric", value_name="value"
        )
        bar = (
            alt.Chart(comp)
            .mark_bar()
            .encode(
                x=alt.X("option:N", title="Intervention Option"),
                y=alt.Y("value:Q", title="Score / Risk"),
                color=alt.Color("metric:N", title="Metric"),
                xOffset="metric:N",
                tooltip=["option", "metric", "value"],
            )
            .properties(height=320)
        )
        st.altair_chart(bar, use_container_width=True)

        best = options_df.iloc[0]
        rec_status = "🟢" if best["residual_risk"] < green_threshold else "🟡" if best["residual_risk"] < yellow_threshold else "🔴"
        st.success(f"Recommended Option: {best['option']} {rec_status} · Decision Score {best['decision_score']}")

    elif page == "Standards (RAG) & Explainability":
        st.subheader("Standards Retrieval & Explainability")
        st.caption("Mock retrieval-augmented rationale anchored in subsystem context and failure hypothesis.")

        st.markdown("#### Cited Guidance")
        for snip in standards:
            st.markdown(f"**{snip['title']}**")
            st.write(snip["excerpt"])

        st.markdown("#### Explainability")
        explanation = (
            f"The recommended strategy **{options_df.iloc[0]['option']}** is prioritized because {selected_asset_name} has "
            f"systemic priority **{selected_asset['systemic_priority']:.1f}**, current health **{selected_asset['current_health']:.1f}**, "
            f"and anomaly score **{selected_asset['anomaly_score']:.2f}** resulting in risk score **{risk_score:.1f}**. "
            f"Notification parsing indicates **{parsed_notification['suspected_failure_type']}** on "
            f"**{parsed_notification['suspected_component']}** (confidence {parsed_notification['confidence']:.2f}). "
            f"Retrieved standards emphasize diagnostic urgency and safe operating boundaries, constraining deferral choices "
            f"and favoring options with stronger risk reduction and controlled residual risk."
        )
        st.info(explanation)

        factors_df = pd.DataFrame(
            {
                "factor": ["Systemic Priority", "Health Degradation", "Anomaly", "Notification Confidence"],
                "value": [
                    round(selected_asset["systemic_priority"], 1),
                    round(100 - selected_asset["current_health"], 1),
                    round(selected_asset["anomaly_score"] * 20, 1),
                    round(parsed_notification["confidence"] * 100, 1),
                ],
            }
        )
        fchart = (
            alt.Chart(factors_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(x="factor:N", y="value:Q", color="factor:N", tooltip=["factor", "value"])
            .properties(height=260)
        )
        st.altair_chart(fchart, use_container_width=True)

    elif page == "SAP Proposal Export":
        st.subheader("SAP Work Order Proposal (Mock JSON)")
        st.caption("Final payload combining notification, risk logic, options, and standards rationale.")

        st.json(sap_payload)
        st.download_button(
            label="Download JSON",
            data=json.dumps(sap_payload, indent=2),
            file_name=f"oracle_work_order_{selected_asset['asset_id']}.json",
            mime="application/json",
        )

        st.markdown("#### Proposal Preview Table")
        preview_df = pd.DataFrame(
            {
                "key": [
                    "asset_id",
                    "asset_name",
                    "risk_score",
                    "traffic_light_status",
                    "predicted_time_to_threshold",
                    "recommended_option",
                    "planner_approval_required",
                ],
                "value": [
                    sap_payload["asset_id"],
                    sap_payload["asset_name"],
                    sap_payload["risk_score"],
                    sap_payload["traffic_light_status"],
                    sap_payload["predicted_time_to_threshold"],
                    sap_payload["recommended_option"]["option"],
                    sap_payload["planner_approval_required"],
                ],
            }
        )
        st.dataframe(preview_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
