import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Polymetalz Risk Engine (PoC)", layout="wide")

# -----------------------------
# Preset profiles (editable)
# -----------------------------
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
    },
}

# -----------------------------
# Band scoring helpers (stepwise)
# -----------------------------
def score_dso_deviation(avg_dso: float, agreed_dso: float) -> int:
    dev = avg_dso - agreed_dso
    if dev <= 5:
        return 95
    if dev <= 15:
        return 75
    if dev <= 30:
        return 50
    return 20


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
    return 30


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


# -----------------------------
# Credit limit + margin buffer logic
# -----------------------------
def max_credit_capacity_pct(band: str) -> float:
    return {"Low": 0.15, "Moderate": 0.10, "High": 0.05, "Very High": 0.02}[band]


def margin_buffer_pp(band: str) -> float:
    return {"Low": 0.2, "Moderate": 0.5, "High": 1.0, "Very High": 2.0}[band]


def default_proxy(band: str) -> float:
    return {"Low": 0.01, "Moderate": 0.03, "High": 0.07, "Very High": 0.15}[band]


def decision_label(band: str, available_limit_cr: float) -> str:
    if available_limit_cr <= 0.0:
        return "REJECT"
    if band == "Very High":
        return "REJECT"
    if band == "High":
        return "REVIEW"
    return "APPROVE"


# -----------------------------
# Header
# -----------------------------
st.title("Polymetalz Risk Intelligence Engine")
st.caption("Proof of Concept: Pre-trade credit + margin decision support (rule-based, manager-adjustable)")

# -----------------------------
# Sidebar inputs
# -----------------------------
with st.sidebar:
    st.header("Buyer Profile")
    preset_name = st.selectbox("Select a preset", list(PRESETS.keys()))
    preset = PRESETS[preset_name]

    st.divider()
    st.subheader("Inputs (editable)")

    annual_turnover_cr = st.number_input(
        "Annual turnover (₹ Cr)", min_value=0.5, max_value=500.0, value=float(preset["annual_turnover_cr"]), step=0.5
    )
    years_in_business = st.number_input("Years in business", min_value=0, max_value=50, value=int(preset["years_in_business"]), step=1)

    agreed_dso = st.number_input("Agreed DSO (days)", min_value=0, max_value=180, value=int(preset["agreed_dso"]), step=1)
    avg_dso = st.number_input("Avg actual DSO (days)", min_value=0, max_value=180, value=int(preset["avg_dso"]), step=1)
    dso_std = st.number_input("DSO variability (std dev, days)", min_value=0.0, max_value=60.0, value=float(preset["dso_std"]), step=1.0)

    existing_exposure_cr = st.number_input(
        "Existing outstanding exposure (₹ Cr)", min_value=0.0, max_value=100.0, value=float(preset["existing_exposure_cr"]), step=0.1
    )
    order_value_cr = st.number_input(
        "Current order value (₹ Cr)", min_value=0.0, max_value=100.0, value=float(preset["order_value_cr"]), step=0.1
    )

    orders_per_quarter = st.number_input("Orders per quarter", min_value=0, max_value=50, value=int(preset["orders_per_quarter"]), step=1)
    polymer_volatility_pct = st.slider("Polymer price volatility (%)", 0.0, 15.0, float(preset["polymer_volatility_pct"]), 0.5)

    st.divider()
    with st.expander("Weights (baseline + adjustable)", expanded=False):
        st.caption("Baseline weights are research-informed; adjust for managerial judgement.")

        w_payment = st.slider("Payment history weight", 0, 40, 25)
        w_exposure = st.slider("Exposure ratio weight", 0, 40, 20)
        w_order = st.slider("Order size ratio weight", 0, 30, 15)
        w_variability = st.slider("Payment variability weight", 0, 30, 15)
        w_years = st.slider("Years in business weight", 0, 25, 10)
        w_frequency = st.slider("Order frequency weight", 0, 25, 10)
        w_vol = st.slider("Volatility modifier weight", 0, 20, 5)

        total_w = w_payment + w_exposure + w_order + w_variability + w_years + w_frequency + w_vol
        if total_w == 0:
            st.error("Weights sum to 0. Increase at least one weight.")
        else:
            st.success(f"Total weight = {total_w} (normalized automatically)")

# Fallback if weights expander isn't opened (still safe)
try:
    total_w
except NameError:
    w_payment, w_exposure, w_order, w_variability, w_years, w_frequency, w_vol = 25, 20, 15, 15, 10, 10, 5
    total_w = w_payment + w_exposure + w_order + w_variability + w_years + w_frequency + w_vol

# -----------------------------
# Compute component scores
# -----------------------------
s_payment = score_dso_deviation(avg_dso, agreed_dso)
s_exposure = score_exposure_ratio(existing_exposure_cr, annual_turnover_cr)
s_order = score_order_ratio(order_value_cr, annual_turnover_cr)
s_years = score_years_in_business(years_in_business)
s_variability = score_payment_variability(dso_std)
s_frequency = score_order_frequency(orders_per_quarter)

def wnorm(w: float) -> float:
    return (w / total_w) if total_w > 0 else 0.0

weighted_pre_mod = (
    s_payment * wnorm(w_payment)
    + s_exposure * wnorm(w_exposure)
    + s_order * wnorm(w_order)
    + s_variability * wnorm(w_variability)
    + s_years * wnorm(w_years)
    + s_frequency * wnorm(w_frequency)
    + 95 * wnorm(w_vol)
)

final_score = apply_volatility_modifier(weighted_pre_mod, polymer_volatility_pct)
band = risk_band(final_score)

# Credit limit
cap_pct = max_credit_capacity_pct(band)
max_credit_cr = annual_turnover_cr * cap_pct
available_limit_cr = max(0.0, max_credit_cr - existing_exposure_cr)

# Margin recommendation
base_margin_pp = 1.0
recommended_margin_pp = base_margin_pp + margin_buffer_pp(band)

# Working capital exposure proxy
expected_exposure_cr = available_limit_cr * (avg_dso / 365.0)

# Expected loss proxy (illustrative)
dp = default_proxy(band)
expected_loss_cr = dp * expected_exposure_cr

# Decision banner
decision = decision_label(band, available_limit_cr)
if decision == "APPROVE":
    st.success(f"✅ DECISION: {decision} — within policy thresholds")
elif decision == "REVIEW":
    st.warning(f"⚠️ DECISION: {decision} — needs manual override / additional safeguards")
else:
    st.error(f"⛔ DECISION: {decision} — outside risk policy or no available limit")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Outputs", "Visual Insights", "Model Transparency"])

with tab1:
    st.markdown("### Key outputs")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Risk Score (0–100)", f"{final_score:.1f}")
        st.metric("Risk Band", band)

    with c2:
        st.metric("Suggested Credit Limit (₹ Cr)", f"{available_limit_cr:.2f}")
        st.metric("Recommended Margin (%)", f"{recommended_margin_pp:.2f}")

    with c3:
        st.metric("WC Exposure Proxy (₹ Cr)", f"{expected_exposure_cr:.2f}")
        st.metric("Expected Loss Proxy (₹ Cr)", f"{expected_loss_cr:.3f}")

    with st.expander("Assumptions & policy notes (for Q&A)", expanded=False):
        st.markdown(
            """
- **Rule-based by design**: early-stage deployment + explainability.  
- **Manager-adjustable weights**: supports judgement in a relationship-driven market.  
- **Not ML (yet)**: ML needs large labeled history (defaults/DSO outcomes). Once sufficient data exists, ML calibration can be layered.
"""
        )

with tab2:
    st.markdown("### Visual insights")

    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### Risk score bar")
        df_bar = pd.DataFrame({"Metric": ["Risk Score"], "Score": [final_score]})
        chart = (
            alt.Chart(df_bar)
            .mark_bar()
            .encode(
                x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("Metric:N"),
                tooltip=["Score:Q"],
            )
            .properties(height=100)
        )
        st.altair_chart(chart, use_container_width=True)

        st.markdown("#### Working capital exposure (proxy)")
        df_wc = pd.DataFrame(
            {
                "Available limit (₹ Cr)": [available_limit_cr],
                "Avg DSO (days)": [avg_dso],
                "WC exposure proxy (₹ Cr)": [expected_exposure_cr],
                "Expected loss proxy (₹ Cr)": [expected_loss_cr],
            }
        )
        st.dataframe(df_wc, use_container_width=True, hide_index=True)

    with right:
        st.markdown("#### What drives the score most? (weights)")
        contrib = pd.DataFrame(
            {
                "Factor": [
                    "Payment history",
                    "Exposure ratio",
                    "Order size ratio",
                    "Payment variability",
                    "Years in business",
                    "Order frequency",
                    "Volatility modifier",
                ],
                "Weight (%)": [
                    round(100 * wnorm(w_payment), 1),
                    round(100 * wnorm(w_exposure), 1),
                    round(100 * wnorm(w_order), 1),
                    round(100 * wnorm(w_variability), 1),
                    round(100 * wnorm(w_years), 1),
                    round(100 * wnorm(w_frequency), 1),
                    round(100 * wnorm(w_vol), 1),
                ],
            }
        ).sort_values("Weight (%)", ascending=False)
        st.dataframe(contrib, use_container_width=True, hide_index=True)

        st.markdown("#### Component scores (0–100)")
        comp = pd.DataFrame(
            {
                "Component": [
                    "Payment history",
                    "Exposure ratio",
                    "Order size ratio",
                    "Payment variability",
                    "Years in business",
                    "Order frequency",
                ],
                "Score": [s_payment, s_exposure, s_order, s_variability, s_years, s_frequency],
            }
        )
        st.dataframe(comp, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Scenario simulator: DSO stress test (F2 – Working Capital)")

    st.caption("Stress-test working capital impact by changing DSO while keeping all other inputs constant.")

    sim_col1, sim_col2 = st.columns([1, 1])

    with sim_col1:
        scenario_mode = st.radio("Choose scenario set", ["Default (30/45/60 days)", "Custom"], horizontal=True)

        if scenario_mode == "Default (30/45/60 days)":
            scenarios = [30, 45, 60]
        else:
            s1 = st.number_input("Scenario DSO #1", min_value=0, max_value=180, value=30, step=1)
            s2 = st.number_input("Scenario DSO #2", min_value=0, max_value=180, value=45, step=1)
            s3 = st.number_input("Scenario DSO #3", min_value=0, max_value=180, value=60, step=1)
            scenarios = sorted(list(set([int(s1), int(s2), int(s3)])))

    # Build scenario table
    sim_rows = []
    for dso in scenarios:
        wc_exposure = available_limit_cr * (dso / 365.0)
        exp_loss = dp * wc_exposure
        sim_rows.append(
            {
                "Scenario DSO (days)": dso,
                "WC Exposure Proxy (₹ Cr)": round(wc_exposure, 3),
                "Expected Loss Proxy (₹ Cr)": round(exp_loss, 4),
            }
        )

    sim_df = pd.DataFrame(sim_rows).sort_values("Scenario DSO (days)")

    with sim_col2:
        st.dataframe(sim_df, use_container_width=True, hide_index=True)

    # Bar chart for WC exposure across scenarios
    sim_chart = (
        alt.Chart(sim_df)
        .mark_bar()
        .encode(
            x=alt.X("Scenario DSO (days):O", title="DSO Scenario (days)"),
            y=alt.Y("WC Exposure Proxy (₹ Cr):Q", title="Working Capital Exposure (₹ Cr)"),
            tooltip=["Scenario DSO (days):O", "WC Exposure Proxy (₹ Cr):Q"],
        )
        .properties(height=240)
    )
    st.altair_chart(sim_chart, use_container_width=True)

with tab3:
    st.markdown("### Model transparency")

    breakdown = pd.DataFrame(
        {
            "Item": [
                "Payment history score",
                "Exposure ratio score",
                "Order size ratio score",
                "Payment variability score",
                "Years in business score",
                "Order frequency score",
                "Volatility (%)",
                "Weighted score (pre-mod)",
                "Final score (post-mod)",
            ],
            "Value": [
                s_payment,
                s_exposure,
                s_order,
                s_variability,
                s_years,
                s_frequency,
                polymer_volatility_pct,
                round(weighted_pre_mod, 1),
                round(final_score, 1),
            ],
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
                "Decision",
            ],
            "Value": [
                band,
                f"{cap_pct*100:.1f}%",
                f"{max_credit_cr:.2f}",
                f"{existing_exposure_cr:.2f}",
                f"{available_limit_cr:.2f}",
                decision,
            ],
        }
    )
    st.dataframe(policy, use_container_width=True, hide_index=True)

    with st.expander("Band logic reference (quick)", expanded=False):
        st.markdown(
            """
**Risk bands**  
- 80–100: Low  
- 60–79: Moderate  
- 40–59: High  
- <40: Very High  

**Decision rule (PoC policy)**  
- Approve: Low/Moderate and available limit > 0  
- Review: High and available limit > 0  
- Reject: Very High or available limit = 0  
"""
        )