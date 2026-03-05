import json
import os
from datetime import datetime, timedelta
from urllib import request, error
import time
import re

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
    mode_shift = {"Normal": 0.0, "High Load": -2.2, "Start-Stop": -1.2}

    all_data, summary = [], []

    for _, asset in assets_df.iterrows():
        subsystem = asset["subsystem"]
        base = rng.uniform(80, 98)
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

        threshold = 40
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
    return base[:3]


def notification_keywords(subsystem: str):
    """Short keyword recommendations for operators (compact and unit-friendly)."""
    common = [
        "vib", "noise", "bear", "align", "temp", "press",
        "leak", "oil", "85°C", "3 mm/s"
    ]
    if subsystem == "Electrical":
        return [
            "hot", "insul", "trip", "load", "95°C", "450V", "120A", "THD 8%"
        ]
    if subsystem == "Process":
        return [
            "press", "flow", "valve", "leak", "8 bar", "120 m3/h", "45°C", "DP +0.8"
        ]
    return common


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


def call_openai_chatgpt_5w(user_text: str, asset_name: str, subsystem: str, model: str, api_key: str, endpoint: str = "https://api.openai.com/v1/chat/completions"):
    """Call OpenAI Chat Completions API to standardize operator notes into 5W JSON."""

    def _parse_http_error(ex: error.HTTPError):
        status = getattr(ex, "code", None)
        retry_after = ""
        try:
            retry_after = ex.headers.get("Retry-After", "") if ex.headers else ""
        except Exception:
            retry_after = ""

        body_msg = ""
        try:
            body_raw = ex.read().decode("utf-8")
            body = json.loads(body_raw)
            body_msg = str(body.get("error", {}).get("message", "")).strip()
        except Exception:
            body_msg = ""

        if status == 429:
            hint = "Rate limit or quota reached (HTTP 429)."
            if retry_after:
                hint += f" Retry-After={retry_after}s."
            if body_msg:
                hint += f" {body_msg}"
            return hint
        return f"HTTP {status}: {body_msg or str(ex)}"

    system_prompt = (
        "You are a maintenance assistant. Convert operator note into strict 5W fields. "
        "Return JSON object only with keys: what, when, where, who, why, standardized_5w."
    )
    user_prompt = f"Asset: {asset_name} | Subsystem: {subsystem}. User note: {user_text}"

    payload = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    retries = [0.0, 1.5, 3.0]
    for attempt, backoff_s in enumerate(retries, start=1):
        if backoff_s > 0:
            time.sleep(backoff_s)

        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=22) as resp:
                raw = resp.read().decode("utf-8")
            outer = json.loads(raw)
            content = outer.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            parsed = json.loads(content)

            out = {
                "what": str(parsed.get("what", "N/A")),
                "when": str(parsed.get("when", "N/A")),
                "where": str(parsed.get("where", f"{asset_name} ({subsystem})")),
                "who": str(parsed.get("who", "Field Operator")),
                "why": str(parsed.get("why", "N/A")),
                "standardized_5w": str(parsed.get("standardized_5w", "N/A")),
                "llm_model": f"{model} (OpenAI)",
            }
            return True, out, ""
        except error.HTTPError as ex:
            err_msg = _parse_http_error(ex)
            if ex.code == 429 and attempt < len(retries):
                continue
            return False, {}, err_msg
        except (error.URLError, TimeoutError, json.JSONDecodeError, ValueError, KeyError, IndexError) as ex:
            return False, {}, str(ex)

    return False, {}, "Unknown OpenAI API error"


def call_openai_fault_analysis(question: str, model: str, api_key: str, endpoint: str = "https://api.openai.com/v1/chat/completions"):
    """Generate fixed transformer fault-analysis response via LLM."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a transformer diagnostics assistant. Answer in concise professional English."},
            {"role": "user", "content": question},
        ],
        "temperature": 0.2,
    }
    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=22) as resp:
            raw = resp.read().decode("utf-8")
        outer = json.loads(raw)
        content = str(outer.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
        return (True, content, "") if content else (False, "", "Empty response")
    except Exception as ex:
        return False, "", str(ex)


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
    """Evaluate intervention strategies with a 3C score: DecisionScore = 100 / (C_maintenance + C_production + C_risk)."""
    base_cost = float(asset["mobilization_cost"])
    crit = float(asset["criticality"]) / 10.0

    options = [
        {"option": "Immediate Repair", "risk_reduction": np.clip(70 + 20 * crit, 0, 100), "expected_downtime_hours": 16 + 8 * crit, "mobilization_cost": base_cost * 1.15},
        {"option": "Defer to Next Window", "risk_reduction": np.clip(28 + 3 * defer_weeks, 0, 75), "expected_downtime_hours": 8 + 2 * defer_weeks, "mobilization_cost": base_cost * (0.88 + 0.02 * defer_weeks)},
        {"option": "Merge with Planned Maintenance", "risk_reduction": np.clip(55 + 8 * crit, 0, 90), "expected_downtime_hours": 12 + 4 * crit, "mobilization_cost": base_cost * 0.82},
        {"option": "Operational Mitigation", "risk_reduction": np.clip(35 + 10 * (1 - crit), 0, 65), "expected_downtime_hours": 4 + 2 * (1 - crit), "mobilization_cost": base_cost * 0.58},
    ]

    rows = []
    for opt in options:
        residual = np.clip(risk_score - opt["risk_reduction"] * 0.65, 0, 100)

        # 3C components (rescaled so denominator is usually >1 and ranking is stable/interpretable)
        c_maintenance = max(opt["mobilization_cost"] / 150000.0, 0.10)
        c_production = max(opt["expected_downtime_hours"] / 72.0, 0.05)
        c_risk = max((residual / 100.0) * (0.8 + 0.6 * crit), 0.05)

        c_total = c_maintenance + c_production + c_risk
        decision_score = float(100.0 / max(c_total, 1e-6))

        rows.append(
            {
                "option": opt["option"],
                "risk_reduction": round(float(opt["risk_reduction"]), 1),
                "expected_downtime_hours": round(float(opt["expected_downtime_hours"]), 1),
                "mobilization_cost": round(float(opt["mobilization_cost"]), 0),
                "residual_risk": round(float(residual), 1),
                "C_maintenance": round(float(c_maintenance), 3),
                "C_production": round(float(c_production), 3),
                "C_risk": round(float(c_risk), 3),
                "C_total": round(float(c_total), 3),
                "decision_score": round(decision_score, 1),
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
    """Mock facility layout coordinates with semi-realistic, non-uniform equipment placement."""
    manual_layout = {
        "TR1": (9, 78), "TR2": (17, 70), "SW1": (12, 57), "SW2": (19, 48),
        "T1": (43, 76), "C1": (55, 72), "C2": (50, 58), "P1": (36, 45), "P2": (46, 36),
        "S1": (74, 74), "S2": (82, 64), "V1": (72, 52), "V2": (86, 46), "HX1": (78, 34), "HX2": (88, 28),
    }
    zone_bounds = {
        "Electrical": (8, 24, 40, 84),
        "Rotating": (32, 58, 30, 84),
        "Process": (66, 90, 24, 82),
    }

    rng = np.random.default_rng(2026)
    rows = []
    for _, r in assets_df.iterrows():
        aid = str(r["asset_id"])
        subsystem = str(r["subsystem"])
        if aid in manual_layout:
            x, y = manual_layout[aid]
        else:
            x0, x1, y0, y1 = zone_bounds.get(subsystem, (20, 80, 20, 80))
            x = float(rng.uniform(x0, x1))
            y = float(rng.uniform(y0, y1))
        rows.append({"asset_id": aid, "asset_name": r["asset_name"], "subsystem": subsystem, "x": float(x), "y": float(y)})
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


def build_direct_causal_dot(graph: nx.DiGraph, model_df: pd.DataFrame, source_id: str, max_depth: int = 2) -> str:
    """Build a compact Graphviz DOT for direct/near-direct causal propagation."""
    dist = nx.single_source_shortest_path_length(graph, source=source_id, cutoff=max_depth)
    nodes = set(dist.keys())
    if source_id not in nodes:
        nodes.add(source_id)

    rows = [
        "digraph G {",
        "rankdir=LR;",
        'graph [bgcolor="#ffffff"];',
        'node [shape=box, style="rounded,filled", fillcolor="#F8FAFF", color="#6B7280"];',
        'edge [color="#9CA3AF"];',
    ]
    risk_map = model_df.set_index("asset_id")["risk_score"].to_dict()
    name_map = model_df.set_index("asset_id")["asset_name"].to_dict()

    for n in nodes:
        risk = float(risk_map.get(n, 0.0))
        fill = "#FECACA" if risk >= 70 else ("#FEF3C7" if risk >= 35 else "#DCFCE7")
        pen = "2.2" if n == source_id else "1.2"
        label = f"{name_map.get(n, n)}\nRisk:{risk:.1f}"
        rows.append(f'"{n}" [label="{label}", fillcolor="{fill}", penwidth={pen}];')

    for u, v, data in graph.edges(data=True):
        if u in nodes and v in nodes:
            w = float(data.get("propagation_weight", 0.0))
            rows.append(f'"{u}" -> "{v}" [label="{w:.2f}"];')

    rows.append("}")
    return "\n".join(rows)


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


def _draft_set(value: str):
    st.session_state["notif_assist_editor"] = value


def _draft_append(value: str):
    current = st.session_state.get("notif_assist_editor", "")
    st.session_state["notif_assist_editor"] = (current + " " + value).strip()


def _draft_clear():
    st.session_state["notif_assist_editor"] = ""


def typewriter_render(text: str, speed_ms: int = 36):
    """Stream text with chunked tokens to mimic ChatGPT output."""
    if not text:
        st.markdown("")
        return ""

    speed = max(speed_ms, 1) / 1000.0

    def token_stream():
        tokens = [tok for tok in re.split(r"(\s+)", text) if tok]
        for tok in tokens:
            yield tok
            wait = speed
            if tok.strip().endswith((".", "!", "?", "；", "。")):
                wait = speed * 4
            elif tok.strip().endswith((",", "，", ":", "：")):
                wait = speed * 2
            time.sleep(wait)

    if hasattr(st, "write_stream"):
        return st.write_stream(token_stream)

    placeholder = st.empty()
    buff = ""
    for tok in token_stream():
        buff += tok
        placeholder.markdown(buff + "▌")
    placeholder.markdown(buff)
    return buff


def render_tab_guide(tab_key: str, text: str, speed_ms: int = 42):
    """Render a per-tab one-time typewriter guide, then keep static summary."""
    state_key = f"tab_guide_played_{tab_key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = False

    st.markdown("<div class='cadence-mini-title'>PAGE GUIDE</div>", unsafe_allow_html=True)
    if not st.session_state[state_key]:
        typewriter_render(text, speed_ms=speed_ms)
        st.session_state[state_key] = True
    else:
        st.caption(text)


def get_secret_or_default(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return str(os.getenv(key, default))




def get_first_config(keys: list[str], default: str = "") -> str:
    """Read the first non-empty key from Streamlit secrets or environment variables."""
    for key in keys:
        val = get_secret_or_default(key, "")
        if str(val).strip():
            return str(val)
    return default


def normalize_api_key(raw_key: str) -> str:
    """Trim whitespace and accidental surrounding quotes."""
    key = (raw_key or "").strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in {'"', "'"}:
        key = key[1:-1].strip()
    return key


def normalize_openai_endpoint(raw_endpoint: str) -> str:
    """Accept full chat-completions endpoint or base URL and normalize to /v1/chat/completions."""
    endpoint = (raw_endpoint or "").strip()
    if not endpoint:
        return "https://api.openai.com/v1/chat/completions"
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/chat/completions"):
        return endpoint
    if endpoint.endswith("/v1"):
        return f"{endpoint}/chat/completions"
    if endpoint.endswith("/v1/chat"):
        return f"{endpoint}/completions"
    if endpoint.endswith("api.openai.com"):
        return f"{endpoint}/v1/chat/completions"
    return f"{endpoint}/v1/chat/completions"



# ------------------------------
# App UI
# ------------------------------

def main():
    st.set_page_config(page_title="CADENCE – Coordinated Asset Decision Engine", page_icon="🏢", layout="wide")

    st.markdown(
        """
        <style>
            .stApp {background: linear-gradient(180deg, #F8FAFC 0%, #EEF2F7 100%); color:#0F172A;}
            [data-testid="stSidebar"] {background: #0F172A; border-right: 1px solid rgba(148,163,184,0.25);}
            [data-testid="stSidebar"] * {color: #E2E8F0 !important;}
            [data-testid="stSidebar"] .stSelectbox label,
            [data-testid="stSidebar"] .stSlider label,
            [data-testid="stSidebar"] .stToggle label {font-weight:600;}
            .gen-note {color:#64748B;font-size:0.9rem;margin-top:-0.3rem;}
            .cadence-wrap {padding:1.0rem 1.2rem; margin:0.2rem 0 0.7rem; border:1px solid #CBD5E1; border-radius:14px; background:#FFFFFF;}
            .cadence-main {font-size:2.0rem; font-weight:800; letter-spacing:0.015em; color:#0F172A;}
            .cadence-sub {font-size:1.05rem; color:#334155; margin-top:0.15rem;}
            .cadence-tag {font-size:0.92rem; color:#64748B; margin-top:0.2rem;}
            div[data-testid="stMetric"] {background:#FFFFFF; border:1px solid #CBD5E1; border-radius:12px; padding:0.5rem 0.7rem; min-height: 152px; display:flex; flex-direction:column; justify-content:flex-start;}
            .stTabs [data-baseweb="tab-list"] {gap:6px;}
            .stTabs [data-baseweb="tab"] {border-radius:10px 10px 0 0; background:#E2E8F0; padding:8px 14px;}
            .stTabs [aria-selected="true"] {background:#1D4ED8 !important; color:white !important;}

            /* Make horizontal radio navigation look like tabs */
            div[role="radiogroup"] {
                gap: 0.35rem;
                border-bottom: 1px solid #CBD5E1;
                padding-bottom: 0.08rem;
                margin-bottom: 0.55rem;
            }
            div[role="radiogroup"] > label {
                margin: 0 !important;
                padding: 0 !important;
                background: transparent !important;
            }
            div[role="radiogroup"] > label > div:first-child {
                display: none;
            }
            div[role="radiogroup"] > label > div:last-child {
                background: #E2E8F0;
                border: 1px solid #CBD5E1;
                border-bottom: none;
                border-radius: 10px 10px 0 0;
                padding: 0.42rem 0.62rem;
                transition: all 0.16s ease;
            }
            div[role="radiogroup"] > label p {
                margin: 0 !important;
                font-weight: 600;
                font-size: 0.82rem;
                color: #475569;
            }
            div[role="radiogroup"] > label:has(input:checked) > div:last-child {
                background: #1D4ED8;
                border-color: #1D4ED8;
                box-shadow: 0 2px 8px rgba(29,78,216,0.18);
            }
            div[role="radiogroup"] > label:has(input:checked) p {
                color: #FFFFFF;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    tab_labels = [
        "Overview",
        "Notification (5W)",
        "Risk Graph",
        "Health Signals",
        "RAG & Explainability",
        "Decision",
        "SAP Export",
    ]
    selected_tab = st.radio(
        "Navigation",
        tab_labels,
        horizontal=True,
        label_visibility="collapsed",
        key="main_tab_nav",
    )

    header_l, header_r = st.columns([5.3, 2.0], vertical_alignment="center")
    with header_l:
        st.markdown(
            """
            <div class='cadence-wrap'>
              <div class='cadence-main'>CADENCE</div>
              <div class='cadence-sub'>Coordinated Asset DECision Engine</div>
              <div class='cadence-tag'>Professional demo for risk-aware, standards-guided maintenance orchestration</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with header_r:
        st.markdown("<div class='cadence-mini-title'>PAGE GUIDE</div>", unsafe_allow_html=True)
        guide_text_map = {
            "Overview": "Overview summarizes current health, risk deltas, anomaly movement, and subsystem ranking so leadership can focus on the most exposed assets first.",
            "Notification (5W)": "Notification Assist converts operator notes into structured 5W records—What, When, Where, Who, Why—to improve data quality for downstream diagnosis and action.",
            "Risk Graph": "Asset Risk Graph explains dependency and cascade pathways, helping teams understand how degradation in one node can propagate to downstream systems.",
            "Health Signals": "Health & PdM Signals shows trend trajectory, anomaly evolution, and threshold crossing projection, supporting proactive maintenance planning.",
            "RAG & Explainability": "RAG & Explainability combines standards references with model reasoning to present likely fault hypotheses and transparent decision context.",
            "Decision": "Decision Orchestration compares options under the 3C lens—maintenance cost, production impact, and residual risk—to identify the most balanced plan.",
            "SAP Export": "SAP Proposal Export packages recommendation, risk context, and references into an ERP-ready payload for planner approval and execution handoff.",
        }
        typewriter_render(guide_text_map.get(selected_tab, ""), speed_ms=110)

    with st.sidebar:
        st.header("Executive Controls")
        scenario = st.selectbox("Scenario", ["Offshore expensive", "Onshore cheaper"], help="Changes mobilization cost multipliers.")
        demo_mode = st.toggle("Demo Mode (randomize mock data)", value=True, key="demo_mode_toggle")

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
    asset_options = model_df["asset_name"].tolist()
    if "selected_asset_name" not in st.session_state:
        st.session_state["selected_asset_name"] = asset_options[default_idx]
    if st.session_state["selected_asset_name"] not in asset_options:
        st.session_state["selected_asset_name"] = asset_options[default_idx]

    selected_name = st.sidebar.selectbox(
        "Selected Asset",
        asset_options,
        index=asset_options.index(st.session_state["selected_asset_name"]),
        key="selected_asset_name",
    )
    selected_asset = model_df[model_df["asset_name"] == selected_name].iloc[0]

    risk_score = float(selected_asset["risk_score"])
    selected_status = traffic_light_text(risk_score, green_threshold, yellow_threshold)
    facility_risk = float(model_df["risk_score"].mean())
    facility_status = traffic_light_text(facility_risk, green_threshold, yellow_threshold)

    notification_default = (
        f"Operator reports rising vibration and intermittent noise at {selected_name} during high load. "
        "Observed slight temperature increase and occasional pressure fluctuation."
    )
    if "main_notification_text" not in st.session_state:
        st.session_state["main_notification_text"] = notification_default
    if "notif_assist_editor" not in st.session_state:
        st.session_state["notif_assist_editor"] = st.session_state["main_notification_text"]
    if "fivew_review" not in st.session_state:
        st.session_state["fivew_review"] = {}
    if "fivew_finalized" not in st.session_state:
        st.session_state["fivew_finalized"] = False

    parsed_notification = parse_notification(st.session_state["main_notification_text"], selected_asset)

    defer_weeks = st.sidebar.slider("Weeks to defer (Option B)", 1, 12, 4)
    planned_windows = [(datetime.now().date() + timedelta(days=d)).isoformat() for d in (7, 14, 21, 28, 42)]
    planned_window = st.sidebar.selectbox("Planned Window (Option C)", planned_windows)

    st.sidebar.markdown("---")
    llm_source = str(get_secret_or_default("LLM_SOURCE", "openai_api")).strip().lower()
    use_openai_api = llm_source != "mock"
    openai_model = str(get_secret_or_default("OPENAI_MODEL", "gpt-4o-mini"))
    openai_endpoint = normalize_openai_endpoint(get_secret_or_default("OPENAI_API_ENDPOINT", "https://api.openai.com/v1/chat/completions"))
    openai_api_key = normalize_api_key(get_first_config(
        ["OPENAI_API_KEY", "openai_api_key", "OPEN_API_KEY", "open_api_key"],
        "",
    ))

    stream_render = str(get_secret_or_default("TYPEWRITER_OUTPUT", "true")).strip().lower() not in {"0", "false", "no"}

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

    if selected_tab == "Overview":
        st.subheader("Overview")
        render_tab_guide("overview", "This page gives the executive snapshot: live health, risk deltas, anomaly changes, and subsystem ranking so leaders can align attention before jumping into detailed analysis.")
        st.markdown("""
        This view follows the **3C-based risk-constrained planning storyline**: 
        **(1)** monitor asset health and context, **(2)** estimate system-aware risk, and **(3)** compare feasible intervention plans with transparent trade-offs.
        """)
        st.caption("Risk context here is aligned with HealthScore, Redundancy, and Dependency signals used by the CADENCE workflow.")

        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.markdown(f"**Selected Asset Status:** {selected_status}")
        c2.markdown(f"**Facility Status:** {facility_status}")

        kpi = st.columns(5)
        kpi[0].metric("Current Health Index", f"{latest_row['health_index']:.1f}", delta=f"{health_delta:+.2f}")
        kpi[1].metric("Risk Score", f"{risk_latest:.1f}", delta=f"{risk_delta:+.2f}")
        kpi[2].metric("Predicted Time-to-Threshold (days)", f"{selected_asset['predicted_time_to_threshold']:.1f}", delta=" ")
        kpi[3].metric("Anomaly Score", f"{latest_row['anomaly_score']:.2f}", delta=f"{anomaly_delta:+.2f}")
        kpi[4].metric("Estimated Mobilization Cost", f"${selected_asset['mobilization_cost']:,.0f}", delta=" ")

        overview_cols = ["asset_id", "asset_name", "subsystem", "criticality", "current_health", "anomaly_score", "systemic_priority", "risk_score"]
        risk_rank = model_df[overview_cols].sort_values(["subsystem", "risk_score"], ascending=[True, False]).copy()
        risk_rank["risk_level"] = np.where(risk_rank["risk_score"] > 40, "High (>40)", "Normal (<=40)")
        risk_rank_chart = sanitize_chart_df(risk_rank, ["risk_score", "asset_name", "subsystem"]) 
        with st.expander("View detailed asset risk table", expanded=False):
            st.dataframe(risk_rank, use_container_width=True, hide_index=True)

        st.markdown("#### Subsystem Risk Panels")
        panel_cols = st.columns(3)
        for idx, subsystem in enumerate(["Electrical", "Process", "Rotating"]):
            panel_df = risk_rank_chart[risk_rank_chart["subsystem"] == subsystem].copy()
            panel_df = panel_df.sort_values("risk_score", ascending=False)
            with panel_cols[idx]:
                st.markdown(f"**{subsystem}**")
                if panel_df.empty:
                    st.info("No assets")
                else:
                    panel_chart = (
                        alt.Chart(panel_df)
                        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                        .encode(
                            x=alt.X("risk_score:Q", title="Risk Score"),
                            y=alt.Y("asset_name:N", sort="-x", title=None),
                            color=alt.Color(
                                "risk_level:N",
                                scale=alt.Scale(domain=["Normal (<=40)", "High (>40)"], range=["#93C5FD", "#F9A8D4"]),
                                legend=None,
                            ),
                            tooltip=["asset_name", "risk_score", "current_health", "systemic_priority"],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(panel_chart, use_container_width=True)

        st.markdown("#### Notification Structuring")
        st.markdown(f"Most likely failure type: **{parsed_notification['suspected_failure_type']}** | Suspected component: **{parsed_notification['suspected_component']}** | Confidence: **{parsed_notification['confidence']:.2f}**")
        with st.expander("View structured notification output", expanded=False):
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

    elif selected_tab == "Notification (5W)":
        st.subheader("Notification (5W)")

        top_left, top_right = st.columns([1.3, 1])
        with top_left:
            st.markdown("#### Quick Terms")
            kw_list = notification_keywords(selected_asset["subsystem"])
            kw_cols = st.columns(len(kw_list))
            for i, kw in enumerate(kw_list):
                kw_cols[i].button(kw, key=f"kw_{i}", on_click=_draft_append, args=(kw,))

            st.markdown("#### Quick Sentences")
            for i, t in enumerate(notification_templates(selected_asset["subsystem"])):
                st.button(t, key=f"tpl_{i}", on_click=_draft_append, args=(t,), use_container_width=True)

        with top_right:
            st.markdown("#### Voice Input (Mock Only)")
            st.caption("Real voice transcription uses paid tokens, so this demo uses simulation only.")
            if hasattr(st, "audio_input"):
                st.audio_input("Record audio (demo only)")
            else:
                st.file_uploader("Upload audio (demo only)", type=["wav", "mp3", "m4a"], key="audio_mock_upload")
            if st.button("Use simulated voice transcript", use_container_width=True):
                mock_text = f"Operator voice note: vibration increased on {selected_name} during high load; inspect bearing and alignment."
                _draft_set(mock_text)
                st.success("Applied simulated voice content.")

        st.markdown("#### Draft · 5W Generator")
        st.markdown("<div class='gen-note'>Generate from the Draft text below, edit (human-in-the-loop), then submit to lock.</div>", unsafe_allow_html=True)
        draft_col, action_col = st.columns([16, 3], vertical_alignment="top")
        with draft_col:
            st.text_area("Draft text", height=200, key="notif_assist_editor", label_visibility="collapsed")
        with action_col:
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            generate_clicked = st.button("Generate 5W", key="generate_5w_btn", use_container_width=True)
            st.button("Clear", key="clear_draft_btn", on_click=_draft_clear, use_container_width=True)

        if generate_clicked:
            user_note = st.session_state.get("notif_assist_editor", "")
            if use_openai_api and openai_api_key.strip():
                ok, result_5w, err = call_openai_chatgpt_5w(
                    user_note,
                    selected_name,
                    selected_asset["subsystem"],
                    model=openai_model,
                    api_key=openai_api_key.strip(),
                    endpoint=openai_endpoint.strip(),
                )
                if not ok:
                    st.warning(f"ChatGPT API call failed, switching to mock. Reason: {err}. If 429, retry later or check billing/quota.")
                    result_5w = mock_mistral_5w(user_note, selected_name, selected_asset["subsystem"])
            else:
                if use_openai_api:
                    st.warning("OpenAI API key not configured; switching to mock.")
                result_5w = mock_mistral_5w(user_note, selected_name, selected_asset["subsystem"])

            st.session_state["fivew_review"] = {
                "what": str(result_5w.get("what", "")),
                "when": str(result_5w.get("when", "")),
                "where": str(result_5w.get("where", "")),
                "who": str(result_5w.get("who", "")),
                "why": str(result_5w.get("why", "")),
                "standardized_5w": str(result_5w.get("standardized_5w", "")),
                "llm_model": str(result_5w.get("llm_model", "mock")),
            }
            st.session_state["fivew_finalized"] = False

        review = st.session_state.get("fivew_review", {})
        if review:
            locked = bool(st.session_state.get("fivew_finalized", False))
            st.markdown("#### 5W Review (Human in the Loop)")
            c1, c2 = st.columns(2)
            review["what"] = c1.text_input("WHAT", value=review.get("what", ""), disabled=locked)
            review["when"] = c2.text_input("WHEN", value=review.get("when", ""), disabled=locked)
            c3, c4 = st.columns(2)
            review["where"] = c3.text_input("WHERE", value=review.get("where", ""), disabled=locked)
            review["who"] = c4.text_input("WHO", value=review.get("who", ""), disabled=locked)
            review["why"] = st.text_input("WHY", value=review.get("why", ""), disabled=locked)
            review["standardized_5w"] = st.text_area("Standardized 5W", value=review.get("standardized_5w", ""), height=160, disabled=locked)
            st.caption(f"Model: {review.get('llm_model', 'mock')}")

            st.session_state["fivew_review"] = review

            if stream_render and not locked:
                st.markdown("##### Streaming Preview")
                typewriter_render(review.get("standardized_5w", ""), speed_ms=28)

            if st.button("Submit and Lock", key="finalize_5w_btn", use_container_width=True, disabled=locked):
                st.session_state["fivew_finalized"] = True
                st.success("5W submitted and locked.")

    elif selected_tab == "Risk Graph":
        st.subheader("Risk Graph")
        

        layout_df = build_layout_positions(assets_df).merge(
            model_df[["asset_id", "asset_name", "subsystem", "risk_score", "current_health"]],
            on=["asset_id", "asset_name", "subsystem"],
            how="left",
        )
        layout_df = sanitize_chart_df(layout_df, ["x", "y", "asset_name", "current_health", "subsystem"])

        st.markdown("#### Facility Layout (Current Health)")
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
                .mark_text(dy=-15, fontSize=10, color="#1F2937")
                .encode(x="x:Q", y="y:Q", text=alt.Text("asset_name:N"))
            )
            st.altair_chart((layout_chart + labels), use_container_width=True)

        st.markdown("#### Direct Causal Graph")
        causal_depth = st.slider("Propagation depth", min_value=1, max_value=4, value=2, key="causal_depth_slider")
        dot = build_direct_causal_dot(graph, model_df, selected_asset["asset_id"], max_depth=causal_depth)
        st.graphviz_chart(dot, use_container_width=True)

        st.markdown("#### Impact Simulator")
        sim_asset_name = st.selectbox(
            "Select asset (simulation)",
            model_df["asset_name"].tolist(),
            index=int(model_df[model_df["asset_id"] == selected_asset["asset_id"]].index[0]),
            key="impact_sim_asset",
        )
        sim_asset = model_df.loc[model_df["asset_name"] == sim_asset_name].iloc[0]
        impact_base = cascade_impact(graph, sim_asset["asset_id"], cutoff=4)
        base_impacted_count = max(len(impact_base) - 1, 0)

        current_h = float(sim_asset["current_health"])
        sim_health = st.slider(
            "Simulated health score",
            min_value=0.0,
            max_value=100.0,
            value=float(round(current_h, 1)),
            step=0.1,
            key="impact_health_slider",
        )
        # Map lower health -> higher propagation stress (and vice versa).
        base_stress = max(100.0 - current_h, 1e-6)
        sim_stress = max(100.0 - float(sim_health), 0.0)
        health_ratio = float(np.clip(sim_stress / base_stress, 0.0, 2.0))

        # Count impacted downstream assets using a relative-strength cutoff so
        # improved health can reduce impacted count while deterioration keeps/expands influence.
        downstream_strengths = [
            float(v)
            for aid, v in impact_base.items()
            if aid != sim_asset["asset_id"] and float(v) > 0
        ]
        if downstream_strengths:
            cutoff = min(downstream_strengths) * 0.95
            impacted_sim = int(sum((s * health_ratio) >= cutoff for s in downstream_strengths))
        else:
            impacted_sim = 0

        c_imp1, c_imp2, c_imp3 = st.columns(3)
        c_imp1.metric("Base impacted assets", f"{base_impacted_count}")
        c_imp2.metric("Simulated impacted assets", f"{impacted_sim}", delta=f"{impacted_sim - base_impacted_count:+d}")
        c_imp3.metric("Simulated health", f"{sim_health:.1f}", delta=f"{sim_health - current_h:+.1f}")

        if sim_health < current_h:
            health_trend_txt = "drops"
        elif sim_health > current_h:
            health_trend_txt = "rises"
        else:
            health_trend_txt = "stays"

        st.markdown(
            f"Asset **{sim_asset_name}** current health is **{current_h:.1f}**. "
            f"When health {health_trend_txt} to **{sim_health:.1f}**, estimated impacted downstream assets change from **{base_impacted_count}** to **{impacted_sim}**."
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

        with st.expander("View systemic priority and adjacency details", expanded=False):
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

    elif selected_tab == "Health Signals":
        st.subheader("Health Signals")

        asset_ts = ts_df[ts_df["asset_id"] == selected_asset["asset_id"]].sort_values("date").reset_index(drop=True).copy()
        asset_ts = sanitize_chart_df(asset_ts, ["date", "health_index", "anomaly_score"])
        threshold = 40

        st.caption("Use the slider to simulate timeline progression and observe health decline; dashed line shows linear trend.")
        st.caption("Threshold line uses Health Index = 40.")
        min_sim = 2 if len(asset_ts) >= 2 else 1
        default_sim = len(asset_ts) if len(asset_ts) > 0 else 1
        sim_day = st.slider("Simulation Day (time progression)", min_value=min_sim, max_value=default_sim, value=default_sim, step=1)
        sim_ts = asset_ts.iloc[:sim_day].copy()
        sim_ts = sanitize_chart_df(sim_ts, ["date", "health_index", "anomaly_score"])
        if sim_ts.empty:
            st.warning("No data available for the selected range.")
            sim_ts = asset_ts.copy()
        sim_ts["t_idx"] = np.arange(len(sim_ts))
        if len(sim_ts) >= 2:
            coef = np.polyfit(sim_ts["t_idx"], sim_ts["health_index"], 1)
            sim_ts["health_fit"] = np.polyval(coef, sim_ts["t_idx"])
        else:
            sim_ts["health_fit"] = sim_ts["health_index"]
        sim_ts_clean = sanitize_chart_df(sim_ts, ["date", "health_index", "health_fit", "anomaly_score"])
        if sim_ts_clean.empty:
            st.warning("No chart-ready data available for plotting.")
        else:
            # Extend the existing regression line directly into the future.
            sim_ts_clean = sim_ts_clean.copy()
            sim_ts_clean["date"] = pd.to_datetime(sim_ts_clean["date"])
            last_date = sim_ts_clean["date"].max()
            start_date = sim_ts_clean["date"].min()

            # Use one single polynomial model for both historical fit and future extension.
            hist_idx = np.arange(len(sim_ts_clean), dtype=float)
            if len(sim_ts_clean) >= 2:
                reg_coef = np.polyfit(hist_idx, sim_ts_clean["health_index"], 1)
            else:
                reg_coef = np.array([-0.08, float(sim_ts_clean["health_index"].iloc[-1])])

            hist_fit = np.clip(np.polyval(reg_coef, hist_idx), 0, 100)
            hist_fit_df = sim_ts_clean[["date"]].copy()
            hist_fit_df["health_reg_ext"] = hist_fit

            future_rows = []
            crossing_date = None
            horizon_days = 540  # ~18 months to make year transitions visible
            for d in range(1, horizon_days + 1):
                f_date = last_date + pd.Timedelta(days=d)
                f_idx = len(sim_ts_clean) + d
                f_health = float(np.clip(np.polyval(reg_coef, f_idx), 0, 100))
                future_rows.append({"date": f_date, "health_reg_ext": f_health})
                if crossing_date is None and f_health <= threshold:
                    crossing_date = f_date

            future_df = pd.DataFrame(future_rows)
            ext_fit_df = pd.concat([hist_fit_df, future_df], ignore_index=True)

            health_90_row = future_df.loc[future_df.index == 89, "health_reg_ext"] if len(future_df) >= 90 else pd.Series(dtype=float)
            health_90 = float(health_90_row.iloc[0]) if not health_90_row.empty else (float(future_df["health_reg_ext"].iloc[-1]) if not future_df.empty else float(hist_fit_df["health_reg_ext"].iloc[-1]))
            if crossing_date is not None:
                st.caption(f"Projected to cross Health Index 40 around {crossing_date.strftime('%Y-%m')}.")
            else:
                st.caption("Projected threshold crossing is beyond current 3-month horizon.")
            st.markdown(f"**3-month projection:** Health Index ≈ **{health_90:.1f}**. {'Immediate repair is recommended.' if health_90 <= 40 else 'Prepare and schedule maintenance soon.'}")

            max_date = ext_fit_df["date"].max()

            y_values = np.concatenate([
                sim_ts_clean["health_index"].to_numpy(dtype=float),
                ext_fit_df["health_reg_ext"].to_numpy(dtype=float),
                np.array([float(threshold)]),
            ])
            y_low = float(np.nanmin(y_values))
            y_high = float(np.nanmax(y_values))
            y_span = max(y_high - y_low, 1.0)
            y_pad = max(6.0, y_span * 0.35)
            y_domain = [max(0.0, y_low - y_pad), min(100.0, y_high + y_pad)]

            # Keep one explicit y-channel here to avoid stale-build confusion.
            health_line = (
                alt.Chart(sim_ts_clean)
                .mark_line(point=False, strokeWidth=2)
                .encode(
                    x=alt.X(
                        "date:T",
                        title="Year-Month",
                        axis=alt.Axis(format="%Y-%m", labelAngle=-35, tickCount=12),
                        scale=alt.Scale(domain=[start_date, max_date]),
                    ),
                    y=alt.Y(
                        "health_index:Q",
                        title="Health Index",
                        scale=alt.Scale(domain=y_domain),
                    ),
                    color=alt.value("#1f77b4"),
                    tooltip=["date:T", "health_index:Q", "operating_mode:N"],
                )
            )
            fit_line = (
                alt.Chart(ext_fit_df)
                .mark_line(strokeDash=[8, 5], strokeWidth=2, color="#2ca02c")
                .encode(
                    x="date:T",
                    y=alt.Y("health_reg_ext:Q", title="Health Index"),
                    tooltip=["date:T", alt.Tooltip("health_reg_ext:Q", title="Linear Fit Extension")],
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

        with st.expander("View latest 10 health records", expanded=False):
            st.dataframe(
                sim_ts[["date", "operating_mode", "health_index", "health_fit", "anomaly_score"]].tail(10).sort_values("date", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

    elif selected_tab == "Decision":
        st.subheader("Decision")
        
        st.markdown("""
        **3.5 3C-Based Risk-Constrained Decision Layer**  
        Objective: compare feasible plans using a unified 3C view — maintenance cost, production impact, and operational risk exposure.
        """)
        st.markdown(f"Based on current conditions and the 3-month health projection toward the 40 threshold, the recommended strategy is **{options_df.iloc[0]['option']}**.")

        # Reuse 3C components already computed by the scoring layer.
        options_3c = options_df.copy()

        with st.expander("View strategy scoring details", expanded=False):
            st.dataframe(options_3c, use_container_width=True, hide_index=True)
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

        st.markdown("#### 3C Formula Simulator (click one option)")
        selected_option_name = st.radio(
            "Candidate plan",
            options_3c["option"].tolist(),
            horizontal=True,
            key="decision_formula_option",
        )
        chosen = options_3c.loc[options_3c["option"] == selected_option_name].iloc[0]

        st.latex(r"DecisionScore(P_j)=\frac{100}{C_{maintenance}(P_j)+C_{production}(P_j)+C_{risk}(P_j)}")
        st.markdown(
            f"""
            <div style="font-size:1.75rem; font-weight:700; line-height:1.6;">
            DecisionScore({selected_option_name}) =
            100 / (
            <span style="color:#2563eb;">{chosen['C_maintenance']:.3f}</span>
            + <span style="color:#ea580c;">{chosen['C_production']:.3f}</span>
            + <span style="color:#9333ea;">{chosen['C_risk']:.3f}</span>
            ) = <span style="color:#111827;">{chosen['decision_score']:.1f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Simulation mapping: C_maintenance (normalized maintenance cost), C_production (normalized production impact), C_risk (normalized residual-risk exposure).")


    elif selected_tab == "RAG & Explainability":
        st.subheader("RAG & Explainability")

        tr_case = model_df.loc[model_df["asset_id"] == "TR1"].iloc[0]
        fixed_q = "What is the most likely potential fault code at present?"

        st.markdown("#### Transformer Fault Storyline · For example, a power transformer health goes down. We are going to check the potential faults.")
        st.text_input("Query prompt", value=fixed_q, disabled=True, key="fixed_fault_question")

        default_story = (
            f"Transformer case (TR1): current health index is {float(tr_case['current_health']):.1f}. "
            "Based on IEEE C57.104 (2019) style interpretation, CO2 concentration is elevated. "
            "Estimated fault type: T1 thermal fault (T>300°C). "
            "Recommended action: schedule inspection and maintenance immediately."
        )

        if "rag_generated" not in st.session_state:
            st.session_state["rag_generated"] = False
        if "rag_answer_cached" not in st.session_state:
            st.session_state["rag_answer_cached"] = ""

        c_submit_l, c_submit_r = st.columns([2, 7])
        with c_submit_l:
            submit_rag = st.button("Submit Query", key="submit_fixed_rag", use_container_width=True)
        with c_submit_r:
            st.caption("Click submit to generate the answer and reveal the Duval triangle panel.")

        if submit_rag:
            rag_answer = default_story
            if use_openai_api and openai_api_key.strip():
                llm_q = (
                    f"{fixed_q} Use transformer TR1 with health index {float(tr_case['current_health']):.1f}. "
                    "Assume IEEE C57.104:2019 evidence indicates elevated CO2 and infer fault T1 thermal fault (T>300°C). "
                    "Provide a concise executive answer with recommended action."
                )
                ok_rag, rag_text, rag_err = call_openai_fault_analysis(llm_q, openai_model, openai_api_key.strip(), openai_endpoint.strip())
                if ok_rag:
                    rag_answer = rag_text
                else:
                    st.warning(f"RAG LLM generation failed; using scenario baseline. Reason: {rag_err}")
            else:
                if use_openai_api:
                    st.warning("RAG used baseline because OpenAI API key is missing.")
            st.session_state["rag_answer_cached"] = rag_answer
            st.session_state["rag_generated"] = True

        if st.session_state.get("rag_generated", False):
            c_rag_l, c_rag_r = st.columns([1.3, 1])
            with c_rag_l:
                st.markdown("#### LLM-Generated RAG Answer")
                rag_answer_text = st.session_state.get("rag_answer_cached", default_story)
                if stream_render:
                    st.markdown("##### Streaming Answer")
                    typewriter_render(rag_answer_text, speed_ms=30)
                else:
                    st.info(rag_answer_text)
                st.markdown("**Reference (scenario assumption):** IEEE C57.104:2019, elevated CO2 trend")

            with c_rag_r:
                st.markdown("#### Transformer Duval-style Triangle")
                triangle_candidates = [
                    "assets/duval_triangle_reference.png",
                    "assets/duval_triangle_reference.svg",
                ]
                triangle_path = next((p for p in triangle_candidates if os.path.exists(p)), "")
                if triangle_path:
                    st.image(triangle_path, caption=f"Duval Triangle reference ({os.path.basename(triangle_path)})", use_container_width=True)
                else:
                    st.warning("Duval triangle image file not found. Add `assets/duval_triangle_reference.png` or `.svg`.")
        else:
            st.info("Press **Submit Query** to generate the fault-analysis answer and display the triangle chart.")

        if st.session_state.get("rag_generated", False):
            st.markdown("#### Explainability")
            explain = (
                f"Decision context: TR1 current health **{float(tr_case['current_health']):.1f}**, risk score **{float(tr_case['risk_score']):.1f}**. "
                f"System recommendation from options remains **{options_df.iloc[0]['option']}** with focus on preventing thermal escalation."
            )
            st.info(explain)

    elif selected_tab == "SAP Export":
        st.subheader("SAP Export")
        st.markdown("Export the aligned maintenance proposal payload for downstream ERP integration.")

        export_col_l, export_col_r = st.columns([1.4, 1])
        with export_col_l:
            st.markdown("#### JSON Payload")
            st.json(sap_payload)
        with export_col_r:
            st.markdown("#### Export Action")
            st.download_button(
                label="Download JSON",
                data=json.dumps(sap_payload, indent=2),
                file_name=f"oracle_work_order_{selected_asset['asset_id']}.json",
                mime="application/json",
                use_container_width=True,
            )
            st.caption("Payload structure is synchronized with current risk, recommendation, and standards outputs.")

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
        with st.expander("View proposal key-value details", expanded=False):
            st.dataframe(preview, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
