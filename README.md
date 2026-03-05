# CADENCE — Coordinated Asset Decision Engine

> **Aligning Maintenance with Operational Rhythm**

CADENCE is a **system-aware decision layer** on top of existing enterprise platforms (e.g., SAP/Fiori, IBM Maximo).  
It converts asset signals into coordinated, production-aware maintenance plans by balancing:

- **Maintenance cost**
- **Production impact**
- **Risk exposure**

🔗 **Live Demo**: https://predictive-maintenance-demo-energypitch.streamlit.app/

---

## 1) Executive Summary

In many operations, the bottleneck is no longer fault detection — it is **decision coordination**.
CADENCE introduces a practical framework that links:

- structured maintenance data,
- equipment health trajectory,
- dependency/cascade context,
- compliance explainability,
- and planner-facing orchestration.

The goal is not to replace existing CMMS/EAM systems, but to improve maintenance timing decisions with a consistent, traceable decision layer.

---

## 2) Operational Gap CADENCE Targets

CADENCE addresses three common pain points:

1. **Data quality constraints**: free-text notifications, inconsistent coding, weak structure.
2. **Component-level focus**: local alarms without dependency-aware system impact.
3. **Planning fragmentation**: manual negotiation between maintenance and production under mobilization constraints.

---

## 3) CADENCE Architecture (5 Layers)

### 3.1 Data Structuring Layer
- Standardizes notifications and failure descriptors (5W + structured fields).
- Improves analytical consistency from free text to structured inputs.

### 3.2 Asset Dependency Modeling Layer
- Builds an **Asset Causal Risk Graph**.
- Quantifies dependency, redundancy, and cascade impact.

### 3.3 Asset Health Evaluation Layer
- Fuses PdM indicators into a composite **Health Index (0–100)**.
- Tracks degradation trajectory and remaining useful margin.
- Supports rule-based/ML fault detection integration.

### 3.4 Compliance Validation & Explanation Layer
- Uses standards-grounded retrieval (RAG-style support) for traceable justification.
- Supports human review before execution.

### 3.5 Decision Orchestration Layer (3C)
Compares candidate plans such as:
- Immediate repair
- Merge with planned maintenance
- Operational mitigation
- Defer to next window

with a unified score:

\[
\text{Decision Score}(P_j) =
\frac{\text{Benefit}(P_j)}
{C_{maintenance}(P_j)+C_{production}(P_j)+C_{risk}(P_j)}
\]

where higher score indicates a more favorable trade-off.

---

## 4) Transformer Illustrative Scenario

For a transformer with thermal-aging signatures (e.g., DGA trend), CADENCE compares feasible intervention paths and quantifies trade-offs among:

- direct maintenance cost,
- production disruption,
- and near-term risk exposure under system dependency context.

This supports disciplined coordination rather than reactive scheduling.

---

## 5) Application Pages (Streamlit)

- **Overview**
- **Notification Assist (5W)**
- **Asset Risk Graph**
- **Health & PdM Signals**
- **Standards (RAG) & Explainability**
- **Decision Orchestration**
- **SAP Proposal Export**

---

## 6) Local Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Online/cloud wrapper:

```bash
streamlit run streamlit_app_online.py
```

---

## 7) OpenAI Configuration (Secure)

> Never commit API keys in code or git history.

The app reads config from **Streamlit secrets** and/or environment variables.

Supported key names:
- `OPENAI_API_KEY`
- `openai_api_key`
- `OPEN_API_KEY`
- `open_api_key`

Optional:
- `OPENAI_MODEL` (default: `gpt-4o-mini`)
- `OPENAI_API_ENDPOINT`

For Streamlit Cloud, set these in **App Settings → Secrets**.

---

## 8) Notes

- Duval reference uses text-friendly assets (`.svg`) to keep diffs reviewable.
- CADENCE is a decision-support layer; engineering judgment and approval remain essential.

---

## Proposal Team

**Primary contact:** Yu-Chung Wang

Background highlights include leading predictive-maintenance data/algorithm work for high-voltage power equipment and receiving technical recognition in predictive maintenance.
