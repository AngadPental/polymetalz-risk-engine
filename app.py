import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Polymetalz PRIE Platform (PoC)", layout="wide")

# =========================================================
# REQUIRED DATA COLUMNS
# =========================================================
REQUIRED_BUYERS_COLS = [
    "buyer_id",
    "buyer_name",
    "annual_turnover_cr",
    "years_in_business",
    "agreed_dso",
    "avg_dso",
    "dso_std",
    "existing_exposure_cr",
    "order_value_cr",
    "orders_per_quarter",
    "polymer_volatility_pct",
    "cibil_score",
]

REQUIRED_DEALS_COLS = [
    "deal_id",
    "buyer_id",
    "transaction_amount_cr",
    "requested_terms_dso",
]

# =========================================================
# PRESETS (INSTANT DEMO)
# =========================================================
PRESETS = {
    "Good Payer – Regular Buyer": {
        "annual_turnover_cr": 20.0,
        "years_in_business": 8,
        "agreed_dso": 30,
        "avg_dso": 32,
        "dso_std": 6,
        "existing_exposure_cr": 0.8,
        "order_value_cr": 0.7,
        "orders_per_quarter": 10,
        "polymer_volatility_pct": 3.0,
        "cibil_score": 780,
    },
    "Average Buyer – Some Delays": {
        "annual_turnover_cr": 12.0,
        "years_in_business": 4,
        "agreed_dso": 30,
        "avg_dso": 45,
        "dso_std": 12,
        "existing_exposure_cr": 1.0,
        "order_value_cr": 0.9,
        "orders_per_quarter": 6,
        "polymer_volatility_pct": 5.0,
        "cibil_score": 720,
    },
    "Risky Buyer – High Exposure + Irregular": {
        "annual_turnover_cr": 8.0,
        "years_in_business": 2,
        "agreed_dso": 30,
        "avg_dso": 75,
        "dso_std": 22,
        "existing_exposure_cr": 1.6,
        "order_value_cr": 1.2,
        "orders_per_quarter": 2,
        "polymer_volatility_pct": 8.0,
        "cibil_score": 640,
    },
}

# =========================================================
# UTILITIES (UI)
# =========================================================
def band_badge(band: str) -> str:
    # Colored tags using emoji + text (works reliably on Streamlit)
    mapping = {
        "Low": "🟢 Low",
        "Moderate": "🟠 Moderate",
        "High": "🔴 High",
        "Very High": "🟣 Very High",
    }
    return mapping.get(band, band)

def decision_badge(dec: str) -> str:
    mapping = {
        "APPROVE": "✅ APPROVE",
        "REVIEW": "⚠️ REVIEW",
        "REJECT": "⛔ REJECT",
        "ERROR": "❌ ERROR",
    }
    return mapping.get(dec, dec)

def style_band_cell(val: str):
    # Pandas Styler for Portfolio/CRM tables
    if "Low" in val:
        return "background-color: #e9f8ee; color: #0f5132; font-weight: 600;"
    if "Moderate" in val:
        return "background-color: #fff4e5; color: #7a4b00; font-weight: 600;"
    if "High" in val:
        return "background-color: #fdecec; color: #842029; font-weight: 600;"
    if "Very High" in val:
        return "background-color: #f0e8ff; color: #3a1a7a; font-weight: 600;"
    return ""

def style_decision_cell(val: str):
    if "APPROVE" in val:
        return "background-color: #e9f8ee; color: #0f5132; font-weight: 700;"
    if "REVIEW" in val:
        return "background-color: #fff4e5; color: #7a4b00; font-weight: 700;"
    if "REJECT" in val:
        return "background-color: #fdecec; color: #842029; font-weight: 700;"
    if "ERROR" in val:
        return "background-color: #f8d7da; color: #842029; font-weight: 700;"
    return ""

# =========================================================
# SCORING FUNCTIONS (CORE LOGIC UNCHANGED)
# =========================================================
def score_dso_deviation(avg_dso: float, agreed_dso: float) -> int:
    dev = avg_dso - agreed_dso
    if dev <= 5:
        return 95
    if dev <= 15:
        return 75
    if dev <= 30:
        return 50
    return 25

def score_exposure_ratio(existing_exposure_cr: float, annual_turnover_cr: float) -> int:
    ratio_pct = (existing_exposure_cr / max(annual_turnover_cr, 0.01)) * 100
    if ratio_pct < 5:
        return 95
    if ratio_pct < 10:
        return 75
    if ratio_pct < 20:
        return 50
    return 25

def score_order_ratio(order_value_cr: float, annual_turnover_cr: float) -> int:
    ratio_pct = (order_value_cr / max(annual_turnover_cr, 0.01)) * 100
    if ratio_pct < 2:
        return 95
    if ratio_pct < 5:
        return 75
    if ratio_pct < 10:
        return 50
    return 25

def score_years_in_business(y: int) -> int:
    if y > 10:
        return 95
    if y >= 5:
        return 75
    if y >= 2:
        return 50
    return 25

def score_payment_variability(dso_std: float) -> int:
    if dso_std <= 7:
        return 95
    if dso_std <= 15:
        return 70
    return 40

def score_order_frequency(orders_per_quarter: int) -> int:
    if orders_per_quarter >= 9:
        return 95
    if orders_per_quarter >= 4:
        return 75
    return 40

def score_cibil(cibil_score: float) -> int:
    if cibil_score >= 750:
        return 95
    if cibil_score >= 700:
        return 75
    if cibil_score >= 650:
        return 50
    return 25

def risk_band(score: float) -> str:
    if score >= 80:
        return "Low"
    if score >= 60:
        return "Moderate"
    if score >= 40:
        return "High"
    return "Very High"

def apply_volatility_modifier(weighted_score: float, volatility_pct: float) -> float:
    modifier = max(0.85, 1.0 - (volatility_pct / 100.0))
    return weighted_score * modifier

def max_credit_capacity_pct(band: str) -> float:
    return {"Low": 0.15, "Moderate": 0.10, "High": 0.05, "Very High": 0.02}[band]

def margin_buffer_pp(band: str) -> float:
    return {"Low": 0.2, "Moderate": 0.5, "High": 1.0, "Very High": 2.0}[band]

def default_proxy(band: str) -> float:
    return {"Low": 0.01, "Moderate": 0.03, "High": 0.07, "Very High": 0.15}[band]

def policy_decision(band: str, available_limit_cr: float) -> str:
    if available_limit_cr <= 0:
        return "REJECT"
    if band == "Very High":
        return "REJECT"
    if band == "High":
        return "REVIEW"
    return "APPROVE"

def compute_engine(profile: dict, weights: dict) -> dict:
    annual_turnover_cr = float(profile["annual_turnover_cr"])
    years_in_business = int(profile["years_in_business"])
    agreed_dso = float(profile["agreed_dso"])
    avg_dso = float(profile["avg_dso"])
    dso_std = float(profile["dso_std"])
    existing_exposure_cr = float(profile["existing_exposure_cr"])
    order_value_cr = float(profile["order_value_cr"])
    orders_per_quarter = int(profile["orders_per_quarter"])
    vol = float(profile["polymer_volatility_pct"])
    cibil = float(profile["cibil_score"])

    s_payment = score_dso_deviation(avg_dso, agreed_dso)
    s_exposure = score_exposure_ratio(existing_exposure_cr, annual_turnover_cr)
    s_order = score_order_ratio(order_value_cr, annual_turnover_cr)
    s_years = score_years_in_business(years_in_business)
    s_variability = score_payment_variability(dso_std)
    s_frequency = score_order_frequency(orders_per_quarter)
    s_cibil = score_cibil(cibil)

    total_w = sum(weights.values()) or 1

    def wnorm(w):
        return w / total_w

    weighted_pre_mod = (
        s_payment * wnorm(weights["payment"])
        + s_exposure * wnorm(weights["exposure"])
        + s_order * wnorm(weights["order"])
        + s_variability * wnorm(weights["variability"])
        + s_years * wnorm(weights["years"])
        + s_frequency * wnorm(weights["frequency"])
        + s_cibil * wnorm(weights["cibil"])
        + 95 * wnorm(weights["volatility"])
    )

    final_score = apply_volatility_modifier(weighted_pre_mod, vol)
    band = risk_band(final_score)

    cap_pct = max_credit_capacity_pct(band)
    max_credit_cr = annual_turnover_cr * cap_pct
    available_limit_cr = max(0.0, max_credit_cr - existing_exposure_cr)

    base_margin = 1.0
    rec_margin = base_margin + margin_buffer_pp(band)

    wc_exposure_cr = available_limit_cr * (avg_dso / 365.0)
    pd_proxy = default_proxy(band)
    expected_loss_cr = pd_proxy * wc_exposure_cr

    decision = policy_decision(band, available_limit_cr)

    return {
        "component_scores": {
            "Payment history score": s_payment,
            "Exposure ratio score": s_exposure,
            "Order size ratio score": s_order,
            "Payment variability score": s_variability,
            "Years in business score": s_years,
            "Order frequency score": s_frequency,
            "CIBIL score": s_cibil,
            "Volatility (%)": vol,
        },
        "weighted_pre_mod": round(weighted_pre_mod, 1),
        "final_score": round(final_score, 1),
        "risk_band": band,
        "capacity_pct": cap_pct,
        "max_credit_cr": round(max_credit_cr, 2),
        "available_limit_cr": round(available_limit_cr, 2),
        "recommended_margin_pct": round(rec_margin, 2),
        "wc_exposure_proxy_cr": round(wc_exposure_cr, 3),
        "expected_loss_proxy_cr": round(expected_loss_cr, 4),
        "decision": decision,
        "pd_proxy": pd_proxy,
    }

# =========================================================
# SESSION STATE
# =========================================================
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

def log_event(event_type: str, details: str):
    st.session_state.audit_log.insert(
        0,
        {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "event": event_type, "details": details},
    )

# =========================================================
# HEADER (MORE PRODUCT-LIKE)
# =========================================================
st.markdown("## Polymetalz PRIE Platform (PoC)")
st.caption(
    "Enterprise pre-trade risk + credit orchestration: buyer scoring • deal-level policy gate • portfolio analytics • CRM ingestion • governance log"
)

# =========================================================
# TOP BAR: ROLE SELECTOR (PLATFORM FEEL)
# =========================================================
top_l, top_r = st.columns([2, 1])
with top_r:
    role = st.selectbox("User role (simulation)", ["Sales Manager", "Risk Analyst", "CFO / Finance Head", "Admin"], index=1)

with top_l:
    if role == "Sales Manager":
        st.info("Sales view: focus on decision, available limit, and recommended margin for the deal.")
    elif role == "Risk Analyst":
        st.info("Risk view: focus on score drivers, risk band rationale, and policy thresholds.")
    elif role == "CFO / Finance Head":
        st.info("Finance view: focus on portfolio exposure, expected loss proxy, and working capital impact.")
    else:
        st.info("Admin view: focus on data ingestion, governance logs, and future security controls.")

# =========================================================
# SIDEBAR: DATA MODE + POLICY CONTROL + INPUTS
# =========================================================
with st.sidebar:
    st.header("Data ingestion")
    mode = st.radio("Choose mode", ["Preset (single buyer)", "Upload buyers CSV (platform)"], index=1)

    buyers_df = None
    selected_row = None

    if mode == "Upload buyers CSV (platform)":
        buyers_up = st.file_uploader("Upload buyers CSV (with CIBIL)", type=["csv"], key="buyers_upload")
        if buyers_up is not None:
            buyers_df = pd.read_csv(buyers_up)
            missing = [c for c in REQUIRED_BUYERS_COLS if c not in buyers_df.columns]
            if missing:
                st.error(f"Missing columns in buyers CSV: {missing}")
                st.stop()

            buyers_df = buyers_df.copy()
            buyers_df["display"] = buyers_df["buyer_id"].astype(str) + " — " + buyers_df["buyer_name"].astype(str)
            choice = st.selectbox("Single buyer view", buyers_df["display"].tolist())
            selected_row = buyers_df.loc[buyers_df["display"] == choice].iloc[0].to_dict()
            log_event("DATA", "Uploaded buyers CSV and selected buyer")

    st.divider()
    st.header("Policy Control Panel")
    st.caption("Baseline weights are research-informed; adjust for managerial judgement (auto-normalized).")

    w_payment = st.slider("Payment history", 0, 40, 23)
    w_exposure = st.slider("Exposure ratio", 0, 40, 18)
    w_order = st.slider("Order size ratio", 0, 30, 14)
    w_variability = st.slider("Payment variability", 0, 30, 12)
    w_years = st.slider("Years in business", 0, 25, 9)
    w_frequency = st.slider("Order frequency", 0, 25, 9)
    w_cibil = st.slider("CIBIL score", 0, 30, 12)
    w_vol = st.slider("Volatility modifier", 0, 20, 3)

    weights = {
        "payment": w_payment,
        "exposure": w_exposure,
        "order": w_order,
        "variability": w_variability,
        "years": w_years,
        "frequency": w_frequency,
        "cibil": w_cibil,
        "volatility": w_vol,
    }
    st.success(f"Total weight = {sum(weights.values())} (normalized automatically)")

    st.divider()
    st.header("Buyer inputs")
    if mode == "Preset (single buyer)":
        preset_name = st.selectbox("Preset", list(PRESETS.keys()))
        profile = PRESETS[preset_name].copy()
    else:
        if selected_row is None:
            st.warning("Upload buyers CSV to enable platform features. Using preset fallback.")
            profile = PRESETS["Average Buyer – Some Delays"].copy()
        else:
            profile = {
                "annual_turnover_cr": float(selected_row["annual_turnover_cr"]),
                "years_in_business": int(selected_row["years_in_business"]),
                "agreed_dso": int(selected_row["agreed_dso"]),
                "avg_dso": int(selected_row["avg_dso"]),
                "dso_std": float(selected_row["dso_std"]),
                "existing_exposure_cr": float(selected_row["existing_exposure_cr"]),
                "order_value_cr": float(selected_row["order_value_cr"]),
                "orders_per_quarter": int(selected_row["orders_per_quarter"]),
                "polymer_volatility_pct": float(selected_row["polymer_volatility_pct"]),
                "cibil_score": float(selected_row["cibil_score"]),
                "buyer_id": selected_row.get("buyer_id", ""),
                "buyer_name": selected_row.get("buyer_name", ""),
            }

    # Editable inputs
    profile["annual_turnover_cr"] = st.number_input("Annual turnover (₹ Cr)", 0.5, 500.0, float(profile["annual_turnover_cr"]), 0.5)
    profile["years_in_business"] = st.number_input("Years in business", 0, 50, int(profile["years_in_business"]), 1)
    profile["agreed_dso"] = st.number_input("Agreed DSO (days)", 0, 180, int(profile["agreed_dso"]), 1)
    profile["avg_dso"] = st.number_input("Avg actual DSO (days)", 0, 180, int(profile["avg_dso"]), 1)
    profile["dso_std"] = st.number_input("DSO variability (std dev, days)", 0.0, 60.0, float(profile["dso_std"]), 1.0)
    profile["existing_exposure_cr"] = st.number_input("Existing exposure (₹ Cr)", 0.0, 100.0, float(profile["existing_exposure_cr"]), 0.1)
    profile["order_value_cr"] = st.number_input("Transaction amount (₹ Cr)", 0.0, 100.0, float(profile["order_value_cr"]), 0.1)
    profile["orders_per_quarter"] = st.number_input("Orders per quarter", 0, 50, int(profile["orders_per_quarter"]), 1)
    profile["polymer_volatility_pct"] = st.slider("Polymer price volatility (%)", 0.0, 15.0, float(profile["polymer_volatility_pct"]), 0.5)
    profile["cibil_score"] = st.number_input("CIBIL score", 300, 900, int(profile["cibil_score"]), 1)

# =========================================================
# SINGLE BUYER: RUN ENGINE + TRANSACTION GATE
# =========================================================
res = compute_engine(profile, weights)
txn_amount = float(profile["order_value_cr"])
within_limit = txn_amount <= float(res["available_limit_cr"])

policy_dec = res["decision"]
final_status = policy_dec
if policy_dec == "APPROVE" and not within_limit:
    final_status = "REVIEW"

# Top decision banner
if final_status == "APPROVE":
    st.success("✅ DECISION: APPROVE — within policy thresholds and within available limit")
elif final_status == "REVIEW":
    st.warning("⚠️ DECISION: REVIEW — requires human check (risk band and/or transaction exceeds available limit)")
else:
    st.error("⛔ DECISION: REJECT — outside policy or no available limit")

# =========================================================
# TABS (RENAMED)
# =========================================================
tab_decision, tab_insights, tab_trans, tab_port, tab_crm, tab_flow, tab_audit = st.tabs(
    [
        "Buyer Decision",
        "Insights",
        "Transparency",
        "Risk Portfolio",
        "CRM Integration",
        "Platform Flow",
        "Governance",
    ]
)

# =========================================================
# TAB 1: BUYER DECISION
# =========================================================
with tab_decision:
    st.subheader("Decision summary (single buyer)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Score (0–100)", f"{res['final_score']:.1f}")
    c2.metric("Risk Band", band_badge(res["risk_band"]))
    c3.metric("Available Limit (₹ Cr)", f"{res['available_limit_cr']:.2f}")
    c4.metric("Recommended Margin (%)", f"{res['recommended_margin_pct']:.2f}")

    st.divider()
    st.markdown("### Deal-level check (transaction gate)")

    left, right = st.columns([1, 1])
    with left:
        st.metric("Transaction amount (₹ Cr)", f"{txn_amount:.2f}")
        st.metric("Within available limit?", "Yes ✅" if within_limit else "No ⚠️")
    with right:
        st.metric("WC Exposure Proxy (₹ Cr)", f"{res['wc_exposure_proxy_cr']:.3f}")
        st.metric("Expected Loss Proxy (₹ Cr)", f"{res['expected_loss_proxy_cr']:.4f}")

    if within_limit:
        st.success("✅ The deal fits within the available limit → proceed as per policy.")
    else:
        st.warning("⚠️ Deal exceeds available limit → force review (split shipment / advance / renegotiate terms).")

    st.markdown("### Supporting external signal")
    st.metric("CIBIL (raw)", f"{int(profile['cibil_score'])}")

# =========================================================
# TAB 2: INSIGHTS (VISUALS + SCENARIO)
# =========================================================
with tab_insights:
    st.subheader("Insights: what is driving risk + working capital impact")

    # Score bar
    df_bar = pd.DataFrame({"Metric": ["Risk Score"], "Score": [res["final_score"]]})
    chart = (
        alt.Chart(df_bar)
        .mark_bar()
        .encode(
            x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("Metric:N"),
            tooltip=["Score:Q"],
        )
        .properties(height=120)
    )
    st.altair_chart(chart, use_container_width=True)

    # Weight driver table
    w_total = sum(weights.values()) or 1
    contrib = pd.DataFrame(
        {
            "Factor": [
                "Payment history",
                "Exposure ratio",
                "Order size ratio",
                "Payment variability",
                "Years in business",
                "Order frequency",
                "CIBIL score",
                "Volatility modifier",
            ],
            "Weight (%)": [
                round(100 * weights["payment"] / w_total, 1),
                round(100 * weights["exposure"] / w_total, 1),
                round(100 * weights["order"] / w_total, 1),
                round(100 * weights["variability"] / w_total, 1),
                round(100 * weights["years"] / w_total, 1),
                round(100 * weights["frequency"] / w_total, 1),
                round(100 * weights["cibil"] / w_total, 1),
                round(100 * weights["volatility"] / w_total, 1),
            ],
        }
    ).sort_values("Weight (%)", ascending=False)

    st.markdown("#### Driver weights (policy control)")
    st.dataframe(contrib, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Scenario simulator: DSO stress test (F2 – Working Capital)")
    st.caption("Change DSO scenarios while keeping everything else constant → see WC exposure and expected loss move.")

    scenario_mode = st.radio("Scenario set", ["Default (30/45/60 days)", "Custom"], horizontal=True)
    if scenario_mode == "Default (30/45/60 days)":
        scenarios = [30, 45, 60]
    else:
        s1 = st.number_input("Scenario DSO #1", 0, 180, 30, 1)
        s2 = st.number_input("Scenario DSO #2", 0, 180, 45, 1)
        s3 = st.number_input("Scenario DSO #3", 0, 180, 60, 1)
        scenarios = sorted(list(set([int(s1), int(s2), int(s3)])))

    sim_rows = []
    for dso in scenarios:
        wc_exposure = float(res["available_limit_cr"]) * (dso / 365.0)
        exp_loss = res["pd_proxy"] * wc_exposure
        sim_rows.append(
            {
                "Scenario DSO (days)": dso,
                "WC Exposure Proxy (₹ Cr)": round(wc_exposure, 3),
                "Expected Loss Proxy (₹ Cr)": round(exp_loss, 4),
            }
        )
    sim_df = pd.DataFrame(sim_rows).sort_values("Scenario DSO (days)")

    cA, cB = st.columns([1, 1])
    with cA:
        st.dataframe(sim_df, use_container_width=True, hide_index=True)
    with cB:
        sim_chart = (
            alt.Chart(sim_df)
            .mark_bar()
            .encode(
                x=alt.X("Scenario DSO (days):O", title="DSO Scenario (days)"),
                y=alt.Y("WC Exposure Proxy (₹ Cr):Q", title="Working Capital Exposure (₹ Cr)"),
                tooltip=["Scenario DSO (days):O", "WC Exposure Proxy (₹ Cr):Q"],
            )
            .properties(height=260)
        )
        st.altair_chart(sim_chart, use_container_width=True)

# =========================================================
# TAB 3: TRANSPARENCY
# =========================================================
with tab_trans:
    st.subheader("Transparency: explainable scoring + policy computation")

    breakdown = pd.DataFrame(
        {
            "Item": list(res["component_scores"].keys()) + ["Weighted score (pre-mod)", "Final score (post-mod)"],
            "Value": list(res["component_scores"].values()) + [res["weighted_pre_mod"], res["final_score"]],
        }
    )
    st.dataframe(breakdown, use_container_width=True, hide_index=True)

    st.markdown("#### Policy computation (trace)")
    policy = pd.DataFrame(
        {
            "Item": [
                "Risk band",
                "Capacity % of turnover",
                "Max credit capacity (₹ Cr)",
                "Existing exposure (₹ Cr)",
                "Available limit (₹ Cr)",
                "Policy decision",
                "Transaction amount (₹ Cr)",
                "Transaction gate result",
                "Final status",
            ],
            "Value": [
                band_badge(res["risk_band"]),
                f"{res['capacity_pct']*100:.1f}%",
                f"{res['max_credit_cr']:.2f}",
                f"{profile['existing_exposure_cr']:.2f}",
                f"{res['available_limit_cr']:.2f}",
                decision_badge(policy_dec),
                f"{txn_amount:.2f}",
                "Within limit ✅" if within_limit else "Exceeds limit ⚠️",
                decision_badge(final_status),
            ],
        }
    )
    st.dataframe(policy, use_container_width=True, hide_index=True)

# =========================================================
# TAB 4: RISK PORTFOLIO (PLATFORM)
# =========================================================
with tab_port:
    st.subheader("Risk Portfolio (platform view)")

    if buyers_df is None:
        st.info("Upload buyers CSV in the sidebar to activate portfolio analytics.")
    else:
        rows = []
        for _, r in buyers_df.iterrows():
            prof = {k: r[k] for k in REQUIRED_BUYERS_COLS}
            out = compute_engine(prof, weights)
            rows.append(
                {
                    "buyer_id": r["buyer_id"],
                    "buyer_name": r["buyer_name"],
                    "cibil": int(r["cibil_score"]),
                    "risk_score": out["final_score"],
                    "risk_band": band_badge(out["risk_band"]),
                    "available_limit_cr": out["available_limit_cr"],
                    "recommended_margin_pct": out["recommended_margin_pct"],
                    "wc_exposure_proxy_cr": out["wc_exposure_proxy_cr"],
                    "expected_loss_proxy_cr": out["expected_loss_proxy_cr"],
                    "policy_decision": decision_badge(out["decision"]),
                }
            )
        port = pd.DataFrame(rows)

        # KPIs (more board-like)
        approvals = port["policy_decision"].str.contains("APPROVE").sum()
        reviews = port["policy_decision"].str.contains("REVIEW").sum()
        rejects = port["policy_decision"].str.contains("REJECT").sum()

        total = len(port)
        approval_rate = (approvals / total) if total else 0
        high_risk_pct = (port["risk_band"].str.contains("High").sum() / total) if total else 0

        total_avail = port["available_limit_cr"].sum()
        total_wc = port["wc_exposure_proxy_cr"].sum()
        total_el = port["expected_loss_proxy_cr"].sum()
        capital_at_risk_pct = (total_el / total_wc) if total_wc else 0  # proxy ratio

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Buyers scored", f"{total}")
        k2.metric("Approval rate", f"{approval_rate*100:.0f}%")
        k3.metric("% High/Very High", f"{high_risk_pct*100:.0f}%")
        k4.metric("Capital-at-risk proxy (EL/WC)", f"{capital_at_risk_pct*100:.1f}%")

        st.divider()

        st.markdown("#### Risk band distribution")
        band_counts = port["risk_band"].value_counts().reset_index()
        band_counts.columns = ["risk_band", "count"]
        pie = (
            alt.Chart(band_counts)
            .mark_arc()
            .encode(theta="count:Q", color="risk_band:N", tooltip=["risk_band:N", "count:Q"])
            .properties(height=260)
        )
        st.altair_chart(pie, use_container_width=True)

        st.markdown("#### Portfolio table (exportable)")
        styled = port.style.applymap(style_band_cell, subset=["risk_band"]).applymap(style_decision_cell, subset=["policy_decision"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.download_button(
            "Download portfolio scoring (CSV)",
            data=port.to_csv(index=False).encode("utf-8"),
            file_name="polymetalz_portfolio_scoring.csv",
            mime="text/csv",
        )

# =========================================================
# TAB 5: CRM INTEGRATION (DEAL INBOX)
# =========================================================
with tab_crm:
    st.subheader("CRM Integration (deal ingestion simulation)")
    st.caption("Simulates CRM → API/Webhook → PRIE Engine → Decision → Pushback to CRM status")

    if buyers_df is None:
        st.info("Upload buyers CSV first (needed to map deals to buyer profiles).")
    else:
        deals_up = st.file_uploader("Upload CRM deals CSV", type=["csv"], key="deals_upload")
        if deals_up is not None:
            deals_df = pd.read_csv(deals_up)
            missing = [c for c in REQUIRED_DEALS_COLS if c not in deals_df.columns]
            if missing:
                st.error(f"Missing columns in deals CSV: {missing}")
                st.stop()

            buyer_lookup = buyers_df.set_index("buyer_id").to_dict(orient="index")

            decisions = []
            for _, d in deals_df.iterrows():
                deal_id = str(d["deal_id"])
                buyer_id = str(d["buyer_id"])
                txn_amt = float(d["transaction_amount_cr"])
                req_terms = float(d["requested_terms_dso"]) if "requested_terms_dso" in deals_df.columns else None

                if buyer_id not in buyer_lookup:
                    decisions.append(
                        {"deal_id": deal_id, "buyer_id": buyer_id, "final_status": decision_badge("ERROR"), "reason": "Buyer ID not found"}
                    )
                    continue

                prof = buyer_lookup[buyer_id].copy()
                prof["order_value_cr"] = txn_amt  # deal-level transaction amount override
                if req_terms is not None and req_terms > 0:
                    prof["agreed_dso"] = req_terms  # optional override (terms request)

                out = compute_engine(prof, weights)

                within = txn_amt <= float(out["available_limit_cr"])
                status = out["decision"]
                if status == "APPROVE" and not within:
                    status = "REVIEW"

                decisions.append(
                    {
                        "deal_id": deal_id,
                        "buyer_id": buyer_id,
                        "buyer_name": prof["buyer_name"],
                        "transaction_amount_cr": txn_amt,
                        "risk_score": out["final_score"],
                        "risk_band": band_badge(out["risk_band"]),
                        "available_limit_cr": out["available_limit_cr"],
                        "within_limit": "Yes ✅" if within else "No ⚠️",
                        "final_status": decision_badge(status),
                        "recommended_margin_pct": out["recommended_margin_pct"],
                        "wc_exposure_proxy_cr": out["wc_exposure_proxy_cr"],
                    }
                )

            decision_df = pd.DataFrame(decisions)
            log_event("CRM", "Ingested CRM deals and generated decisions")

            st.markdown("#### Deal inbox (what would be pushed back to CRM)")
            styled_deals = decision_df.style.applymap(style_band_cell, subset=["risk_band"]).applymap(style_decision_cell, subset=["final_status"])
            st.dataframe(styled_deals, use_container_width=True, hide_index=True)

            st.download_button(
                "Download deal decisions (CSV)",
                data=decision_df.to_csv(index=False).encode("utf-8"),
                file_name="polymetalz_deal_decisions.csv",
                mime="text/csv",
            )

            st.divider()
            st.markdown("### API/Integration diagram (how this becomes real)")
            st.code(
                "CRM (Zoho/Salesforce/HubSpot)\n"
                "   ↓  (API/Webhook: new deal event)\n"
                "Integration Layer (mapping + validation)\n"
                "   ↓\n"
                "PRIE Scoring Service (risk + policy + gate)\n"
                "   ↓\n"
                "Decision Service (Approve/Review/Reject + limit + margin)\n"
                "   ↓\n"
                "Pushback to CRM (deal status + recommendations)\n",
                language="text",
            )

# =========================================================
# TAB 6: PLATFORM FLOW (SYSTEM VIEW)
# =========================================================
with tab_flow:
    st.subheader("Platform Flow (end-to-end)")
    st.caption("This is the platformization story in one slide.")

    st.markdown(
        """
**Core system flow (orchestration):**

1) **Data ingestion**  
   - Buyer master data (CSV → future: ERP/CRM sync)  
   - Deal requests (CRM export/API → deal inbox)

2) **Risk intelligence (PRIE Engine)**  
   - Compute explainable component scores (0–100)  
   - Combine weights (policy control panel)  
   - Apply volatility modifier  
   - Assign risk band (Low/Moderate/High/Very High)

3) **Credit policy + limits**  
   - Capacity % of turnover by risk band  
   - Compute available limit = max capacity − existing exposure

4) **Deal-level policy gate**  
   - Check: transaction amount ≤ available limit  
   - Convert APPROVE → REVIEW when the deal is too large

5) **Outputs + auditability**  
   - Recommendations: approve/review/reject, limit, margin  
   - Portfolio analytics: exposure + expected loss proxies  
   - Governance: audit log (who/what/when)
"""
    )

    st.divider()
    st.markdown("### What makes this a platform (not a model)")
    st.markdown(
        """
- **Reusable decision service** (same engine applies to every buyer + deal)  
- **Workflow-ready** (deal inbox + pushback concept to CRM)  
- **Portfolio view** (risk aggregation, not only single decisions)  
- **Governance** (audit trail + future security controls)  
"""
    )

# =========================================================
# TAB 7: GOVERNANCE (AUDIT + SECURITY TALKING POINTS)
# =========================================================
with tab_audit:
    st.subheader("Governance & auditability (simulation)")
    st.caption("In production this would sit on a DB with RBAC/SSO and immutable logs.")

    if len(st.session_state.audit_log) == 0:
        st.info("No audit events yet.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.audit_log), use_container_width=True, hide_index=True)

    with st.expander("Security controls (future slide talking points)", expanded=False):
        st.markdown(
            """
**PoC now:** minimal, demo-grade security (Streamlit hosted over HTTPS).  
**Production plan:**  
- SSO / RBAC (Sales vs Risk vs Admin)  
- Encryption at rest + TLS in transit  
- Secrets management for bureau/CRM keys  
- Audit logging + retention policy  
- Consent + compliance for bureau data (CIBIL)  
"""
        )
