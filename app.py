import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime

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
    "cibil_score",  # NEW
]

REQUIRED_DEALS_COLS = [
    "deal_id",
    "buyer_id",
    "transaction_amount_cr",  # NEW (deal-level)
    "requested_terms_dso",    # optional override for scenario (kept in template)
]

# =========================================================
# PRESETS (SINGLE-BUYER INSTANT DEMO)
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
# SCORING FUNCTIONS
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
    # Simple, defensible mapping for PoC
    # (In production, this could be calibrated to default outcomes)
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
    # cap downside so macro volatility doesn't dominate buyer behavior
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
# SESSION STATE: AUDIT + CRM INBOX (PLATFORM FEEL)
# =========================================================
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

if "crm_inbox" not in st.session_state:
    st.session_state.crm_inbox = []  # list of deal decisions


def log_event(event_type: str, details: str):
    st.session_state.audit_log.insert(
        0,
        {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "event": event_type, "details": details},
    )


# =========================================================
# UI HEADER
# =========================================================
st.title("Polymetalz PRIE Platform (PoC)")
st.caption("Platformised PoC: batch buyer scoring + CRM deal ingestion + deal-level policy gate (transaction amount) + transparency")

# =========================================================
# SIDEBAR: DATA + WEIGHTS + SINGLE VIEW INPUTS
# =========================================================
with st.sidebar:
    st.header("Data mode")
    mode = st.radio("Choose mode", ["Preset (single buyer)", "Upload buyers CSV (platform)"], index=0)

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
    st.header("Weights (adjustable)")
    st.caption("Normalized automatically. Baseline weights are research-informed; adjust for judgement.")

    w_payment = st.slider("Payment history", 0, 40, 23)
    w_exposure = st.slider("Exposure ratio", 0, 40, 18)
    w_order = st.slider("Order size ratio", 0, 30, 14)
    w_variability = st.slider("Payment variability", 0, 30, 12)
    w_years = st.slider("Years in business", 0, 25, 9)
    w_frequency = st.slider("Order frequency", 0, 25, 9)
    w_cibil = st.slider("CIBIL score", 0, 30, 12)  # NEW
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
    st.header("Buyer inputs (single view)")
    if mode == "Preset (single buyer)":
        preset_name = st.selectbox("Preset", list(PRESETS.keys()))
        profile = PRESETS[preset_name].copy()
    else:
        if selected_row is None:
            st.info("Upload buyers CSV to enable platform features. Using a preset fallback.")
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

    # editable inputs
    profile["annual_turnover_cr"] = st.number_input("Annual turnover (₹ Cr)", 0.5, 500.0, float(profile["annual_turnover_cr"]), 0.5)
    profile["years_in_business"] = st.number_input("Years in business", 0, 50, int(profile["years_in_business"]), 1)
    profile["agreed_dso"] = st.number_input("Agreed DSO (days)", 0, 180, int(profile["agreed_dso"]), 1)
    profile["avg_dso"] = st.number_input("Avg actual DSO (days)", 0, 180, int(profile["avg_dso"]), 1)
    profile["dso_std"] = st.number_input("DSO variability (std dev, days)", 0.0, 60.0, float(profile["dso_std"]), 1.0)
    profile["existing_exposure_cr"] = st.number_input("Existing exposure (₹ Cr)", 0.0, 100.0, float(profile["existing_exposure_cr"]), 0.1)
    profile["order_value_cr"] = st.number_input("Transaction amount / Order value (₹ Cr)", 0.0, 100.0, float(profile["order_value_cr"]), 0.1)
    profile["orders_per_quarter"] = st.number_input("Orders per quarter", 0, 50, int(profile["orders_per_quarter"]), 1)
    profile["polymer_volatility_pct"] = st.slider("Polymer price volatility (%)", 0.0, 15.0, float(profile["polymer_volatility_pct"]), 0.5)
    profile["cibil_score"] = st.number_input("CIBIL score", 300, 900, int(profile["cibil_score"]), 1)

# =========================================================
# SINGLE BUYER OUTPUTS + TRANSACTION AMOUNT GATE (A)
# =========================================================
res = compute_engine(profile, weights)

# Deal-level gate: transaction amount should fit within available limit
txn_amount = float(profile["order_value_cr"])
within_limit = txn_amount <= float(res["available_limit_cr"])

# Decision logic: combine policy decision + transaction gate
decision = res["decision"]
status = decision

if decision == "APPROVE" and not within_limit:
    status = "REVIEW"  # not auto-reject, but forces manual review
if decision == "REVIEW" and not within_limit:
    status = "REVIEW"
if decision == "REJECT":
    status = "REJECT"

if status == "APPROVE":
    st.success("✅ DECISION: APPROVE — within policy thresholds and within available limit")
elif status == "REVIEW":
    st.warning("⚠️ DECISION: REVIEW — requires human check (risk band and/or transaction exceeds available limit)")
else:
    st.error("⛔ DECISION: REJECT — outside policy or no available limit")

# =========================================================
# TABS
# =========================================================
tab_out, tab_vis, tab_trans, tab_port, tab_crm, tab_audit = st.tabs(
    ["Outputs (Single)", "Visual Insights", "Model Transparency", "Portfolio (Platform)", "CRM Deals (No Manual Feed)", "Audit Log"]
)

# -------------------------
# Outputs (Single)
# -------------------------
with tab_out:
    st.subheader("Key outputs (single buyer)")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Risk Score (0–100)", f"{res['final_score']:.1f}")
        st.metric("Risk Band", res["risk_band"])

    with c2:
        st.metric("Available Limit (₹ Cr)", f"{res['available_limit_cr']:.2f}")
        st.metric("Transaction Amount (₹ Cr)", f"{txn_amount:.2f}")

    with c3:
        st.metric("Recommended Margin (%)", f"{res['recommended_margin_pct']:.2f}")
        st.metric("WC Exposure Proxy (₹ Cr)", f"{res['wc_exposure_proxy_cr']:.3f}")

    st.divider()
    st.markdown("### Deal check (transaction gate)")
    if within_limit:
        st.success("✅ Transaction fits within available limit → can proceed as per risk policy.")
    else:
        st.warning("⚠️ Transaction exceeds available limit → force manual review / renegotiate (advance / smaller shipment / split orders).")

    st.markdown("### Supporting risk metrics")
    c4, c5 = st.columns(2)
    with c4:
        st.metric("Expected Loss Proxy (₹ Cr)", f"{res['expected_loss_proxy_cr']:.4f}")
    with c5:
        st.metric("CIBIL (raw)", f"{int(profile['cibil_score'])}")

# -------------------------
# Visual Insights + Scenario Simulator
# -------------------------
with tab_vis:
    st.subheader("Visual insights")
    left, right = st.columns([1, 1])

    with left:
        df_bar = pd.DataFrame({"Metric": ["Risk Score"], "Score": [res["final_score"]]})
        chart = (
            alt.Chart(df_bar)
            .mark_bar()
            .encode(
                x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("Metric:N"),
                tooltip=["Score:Q"],
            )
            .properties(height=110)
        )
        st.altair_chart(chart, use_container_width=True)

        df_wc = pd.DataFrame(
            {
                "Available limit (₹ Cr)": [res["available_limit_cr"]],
                "Avg DSO (days)": [profile["avg_dso"]],
                "WC exposure proxy (₹ Cr)": [res["wc_exposure_proxy_cr"]],
                "Expected loss proxy (₹ Cr)": [res["expected_loss_proxy_cr"]],
            }
        )
        st.markdown("#### Working capital exposure (proxy)")
        st.dataframe(df_wc, use_container_width=True, hide_index=True)

    with right:
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
        st.markdown("#### What drives the score most? (weights)")
        st.dataframe(contrib, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Scenario simulator: DSO stress test (F2)")
    st.caption("Stress-test working capital impact by changing DSO while keeping all other inputs constant.")

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
        sim_rows.append({"Scenario DSO (days)": dso, "WC Exposure Proxy (₹ Cr)": round(wc_exposure, 3), "Expected Loss Proxy (₹ Cr)": round(exp_loss, 4)})

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
            .properties(height=250)
        )
        st.altair_chart(sim_chart, use_container_width=True)

# -------------------------
# Model Transparency
# -------------------------
with tab_trans:
    st.subheader("Model transparency (single buyer)")
    breakdown = pd.DataFrame(
        {
            "Item": list(res["component_scores"].keys()) + ["Weighted score (pre-mod)", "Final score (post-mod)"],
            "Value": list(res["component_scores"].values()) + [res["weighted_pre_mod"], res["final_score"]],
        }
    )
    st.dataframe(breakdown, use_container_width=True, hide_index=True)

    st.markdown("#### Credit policy computation")
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
                res["risk_band"],
                f"{res['capacity_pct']*100:.1f}%",
                f"{res['max_credit_cr']:.2f}",
                f"{profile['existing_exposure_cr']:.2f}",
                f"{res['available_limit_cr']:.2f}",
                decision,
                f"{txn_amount:.2f}",
                "Within limit ✅" if within_limit else "Exceeds limit ⚠️",
                status,
            ],
        }
    )
    st.dataframe(policy, use_container_width=True, hide_index=True)

# -------------------------
# Portfolio (Platform): batch scoring
# -------------------------
with tab_port:
    st.subheader("Portfolio dashboard (platform view)")
    if buyers_df is None:
        st.info("Upload buyers CSV (platform mode) to activate portfolio analytics.")
    else:
        rows = []
        for _, r in buyers_df.iterrows():
            prof = {k: r[k] for k in REQUIRED_BUYERS_COLS}
            out = compute_engine(prof, weights)
            rows.append(
                {
                    "buyer_id": r["buyer_id"],
                    "buyer_name": r["buyer_name"],
                    "cibil": r["cibil_score"],
                    "risk_score": out["final_score"],
                    "risk_band": out["risk_band"],
                    "available_limit_cr": out["available_limit_cr"],
                    "recommended_margin_pct": out["recommended_margin_pct"],
                    "wc_exposure_proxy_cr": out["wc_exposure_proxy_cr"],
                    "expected_loss_proxy_cr": out["expected_loss_proxy_cr"],
                    "policy_decision": out["decision"],
                }
            )
        port = pd.DataFrame(rows)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Buyers scored", f"{len(port)}")
        with c2:
            st.metric("Total Available Limit (₹ Cr)", f"{port['available_limit_cr'].sum():.2f}")
        with c3:
            st.metric("Total WC Exposure Proxy (₹ Cr)", f"{port['wc_exposure_proxy_cr'].sum():.2f}")
        with c4:
            st.metric("Total Expected Loss Proxy (₹ Cr)", f"{port['expected_loss_proxy_cr'].sum():.3f}")

        st.markdown("### Risk band distribution")
        band_counts = port["risk_band"].value_counts().reset_index()
        band_counts.columns = ["risk_band", "count"]
        pie = alt.Chart(band_counts).mark_arc().encode(theta="count:Q", color="risk_band:N", tooltip=["risk_band:N", "count:Q"]).properties(height=260)
        st.altair_chart(pie, use_container_width=True)

        st.markdown("### Portfolio table")
        st.dataframe(port.sort_values(["risk_band", "risk_score"], ascending=[True, False]), use_container_width=True, hide_index=True)

        st.download_button(
            "Download portfolio scoring (CSV)",
            data=port.to_csv(index=False).encode("utf-8"),
            file_name="polymetalz_portfolio_scoring.csv",
            mime="text/csv",
        )

# -------------------------
# CRM Deals: no manual feeding (CSV import simulates CRM sync)
# -------------------------
with tab_crm:
    st.subheader("CRM Deals ingestion (simulation)")
    st.caption("Simulates CRM integration: ingest deals export → auto-score → apply transaction amount gate → decision queue.")

    if buyers_df is None:
        st.info("Upload buyers CSV (platform mode) first (needed to map deal buyer_id to buyer profile).")
    else:
        deals_up = st.file_uploader("Upload CRM deals CSV", type=["csv"], key="deals_upload")
        if deals_up is not None:
            deals_df = pd.read_csv(deals_up)
            missing = [c for c in REQUIRED_DEALS_COLS if c not in deals_df.columns]
            if missing:
                st.error(f"Missing columns in deals CSV: {missing}")
                st.stop()

            # Build lookup for buyers
            buyer_lookup = buyers_df.set_index("buyer_id").to_dict(orient="index")

            decisions = []
            for _, d in deals_df.iterrows():
                deal_id = str(d["deal_id"])
                buyer_id = str(d["buyer_id"])
                txn_amt = float(d["transaction_amount_cr"])
                req_terms = float(d["requested_terms_dso"]) if "requested_terms_dso" in deals_df.columns else None

                if buyer_id not in buyer_lookup:
                    decisions.append(
                        {
                            "deal_id": deal_id,
                            "buyer_id": buyer_id,
                            "status": "ERROR",
                            "reason": "Buyer ID not found in buyers dataset",
                        }
                    )
                    continue

                prof = buyer_lookup[buyer_id].copy()
                # Use deal transaction amount (overrides buyer default order_value_cr)
                prof["order_value_cr"] = txn_amt

                # Optional: requested terms DSO could be used as a scenario assumption
                if req_terms is not None and req_terms > 0:
                    prof["agreed_dso"] = req_terms

                out = compute_engine(prof, weights)

                within = txn_amt <= float(out["available_limit_cr"])
                final_status = out["decision"]
                if final_status == "APPROVE" and not within:
                    final_status = "REVIEW"

                decisions.append(
                    {
                        "deal_id": deal_id,
                        "buyer_id": buyer_id,
                        "buyer_name": prof["buyer_name"],
                        "transaction_amount_cr": txn_amt,
                        "risk_score": out["final_score"],
                        "risk_band": out["risk_band"],
                        "available_limit_cr": out["available_limit_cr"],
                        "within_limit": "Yes" if within else "No",
                        "final_status": final_status,
                        "recommended_margin_pct": out["recommended_margin_pct"],
                        "wc_exposure_proxy_cr": out["wc_exposure_proxy_cr"],
                    }
                )

            decision_df = pd.DataFrame(decisions)
            st.dataframe(decision_df, use_container_width=True, hide_index=True)
            log_event("CRM", "Ingested CRM deals and generated decisions")

            st.download_button(
                "Download deal decisions (CSV)",
                data=decision_df.to_csv(index=False).encode("utf-8"),
                file_name="polymetalz_deal_decisions.csv",
                mime="text/csv",
            )

            st.markdown("### How to explain this as 'CRM integration'")
            st.markdown(
                """
In the PoC, we ingest CRM deals via CSV export.  
In production, the same interface becomes an API/webhook from the CRM (Zoho/Salesforce/HubSpot) that:
- pulls deal + buyer_id automatically  
- runs PRIE scoring + transaction gate  
- pushes the decision back to CRM as a deal status (Approved/Review/Rejected) with recommended limit and margin.
"""
            )

# -------------------------
# Audit Log
# -------------------------
with tab_audit:
    st.subheader("Audit log (governance simulation)")
    st.caption("In production this would be stored in a database with RBAC/SSO and immutable logs.")
    if len(st.session_state.audit_log) == 0:
        st.info("No audit events yet.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.audit_log), use_container_width=True, hide_index=True)

    with st.expander("Security (future slide talking points)", expanded=False):
        st.markdown(
            """
**Future security controls (not built in PoC):**
- SSO / RBAC (role-based views for Sales vs Finance vs Admin)
- Encryption at rest + TLS in transit
- Secrets management for bureau/CRM API keys
- Audit logging + retention policy
- Consent & compliance for bureau data (CIBIL)
"""
        )
