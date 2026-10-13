from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib

st.set_page_config(page_title="MEP Project Risk Intelligence", page_icon="🏗️", layout="wide")
DATA = Path("data")
MODELS = Path("models")
FEATURES_NUM = ["budget_egp", "planned_duration_months", "complexity_1_5", "design_maturity_pct",
"procurement_risk_1_5", "subcontractor_risk_1_5", "site_readiness_pct", "change_order_rate_pct",
"payment_delay_days", "resource_gap_pct", "approval_delay_days", "quality_ncr_rate", "safety_incidents",
"long_lead_material_months"]
FEATURES_CAT = ["project_type", "discipline"]
FEATURES = FEATURES_NUM + FEATURES_CAT

st.title("🏗️ MEP Construction Project Risk Intelligence")
st.caption("Historical monitoring • Active-project risk analysis • New-project scenario simulation")
st.warning("Prototype uses synthetic illustrative data. Do not use its predictions for contractual, financial, or safety decisions without real-data validation and domain review.")

@st.cache_data
def load_data():
    p = pd.read_csv(DATA/"mep_projects.csv")
    h = pd.read_csv(DATA/"mep_monthly_history.csv")
    return p, h

if not (DATA/"mep_projects.csv").exists() or not (DATA/"mep_monthly_history.csv").exists():
    st.error("Data files are missing. Run `python generate_data.py` first.")
    st.stop()

projects, history = load_data()
tab1, tab2, tab3 = st.tabs(["Portfolio Monitor", "Project Deep Dive", "New Project: Monte Carlo"])

with tab1:
    st.subheader("Portfolio overview")
    active = projects[projects.status == "Active"]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total projects", f"{len(projects):,}")
    c2.metric("Active projects", f"{len(active):,}")
    c3.metric("Active avg. risk score", f"{active.overall_risk_score.mean():.1f}/100")
    c4.metric("Active portfolio budget", f"EGP {active.budget_egp.sum()/1e6:,.1f}M")
    f1,f2 = st.columns(2)
    with f1:
        status_filter = st.multiselect("Status", ["Active","Completed"], default=["Active","Completed"])
    with f2:
        disc_filter = st.multiselect("MEP discipline", sorted(projects.discipline.unique()), default=sorted(projects.discipline.unique()))
    view = projects[projects.status.isin(status_filter) & projects.discipline.isin(disc_filter)]
    fig = px.histogram(view, x="overall_risk_score", color="status", nbins=20, title="Overall risk score distribution")
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.scatter(view, x="budget_egp", y="overall_risk_score", color="status", size="planned_duration_months",
                      hover_name="project_name", hover_data=["discipline","predicted_delay_probability"], title="Budget vs risk")
    st.plotly_chart(fig2, use_container_width=True)
    st.subheader("Projects requiring attention")
    cols = ["project_id","project_name","status","discipline","budget_egp","overall_risk_score",
            "predicted_delay_probability","approval_delay_days","payment_delay_days","resource_gap_pct"]
    st.dataframe(view.sort_values("overall_risk_score", ascending=False)[cols], use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Active/completed project history")
    pid = st.selectbox("Select project", projects.project_id.tolist())
    p = projects.loc[projects.project_id == pid].iloc[0]
    hist = history[history.project_id == pid].sort_values("month_index")
    a,b,c = st.columns(3)
    a.metric("Risk score", f"{p.overall_risk_score:.1f}/100")
    b.metric("Project status", p.status)
    c.metric("Budget", f"EGP {p.budget_egp/1e6:.2f}M")
    st.plotly_chart(px.line(hist, x="month_index", y="progress_pct", markers=True, title="Progress trend"), use_container_width=True)
    st.plotly_chart(px.line(hist, x="month_index", y=["schedule_variance_days","procurement_delay_days"],
                            markers=True, title="Schedule and procurement indicators"), use_container_width=True)
    st.plotly_chart(px.line(hist, x="month_index", y=["open_rfis","pending_material_approvals","open_ncrs"],
                            markers=True, title="Technical, material approval and quality backlog"), use_container_width=True)
    st.dataframe(hist, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("New project risk estimation")
    st.write("Enter a preliminary MEP project profile. Monte Carlo varies uncertain schedule and cost drivers around your assumptions.")
    with st.form("new_project_form"):
        c1,c2,c3 = st.columns(3)
        with c1:
            budget = st.number_input("Budget (EGP)", min_value=100000.0, value=80000000.0, step=1000000.0)
            duration = st.number_input("Planned duration (months)", min_value=1, max_value=120, value=18)
            project_type = st.selectbox("Project type", ["Residential","Commercial","Hospital","Industrial","Infrastructure"])
            discipline = st.selectbox("Primary MEP discipline", ["HVAC","Firefighting","Plumbing","Electrical","ELV/BMS","Multi-discipline"])
            complexity = st.slider("Complexity (1–5)", 1.0, 5.0, 3.0, .1)
        with c2:
            design = st.slider("Design maturity (%)", 0, 100, 70)
            procurement = st.slider("Procurement risk (1–5)", 1.0, 5.0, 3.0, .1)
            subcontractor = st.slider("Subcontractor risk (1–5)", 1.0, 5.0, 3.0, .1)
            site = st.slider("Site readiness (%)", 0, 100, 70)
            change = st.slider("Expected change-order rate (%)", 0.0, 30.0, 5.0, .5)
        with c3:
            payment = st.slider("Expected payment delay (days)", 0, 180, 30)
            resource = st.slider("Expected resource gap (%)", 0.0, 60.0, 10.0, .5)
            approval = st.slider("Expected approval delay (days)", 0, 90, 15)
            ncr = st.slider("Expected quality NCR rate", 0.0, 20.0, 3.0, .5)
            safety = st.slider("Expected safety incidents (count)", 0, 20, 1)
            lead = st.slider("Long-lead material duration (months)", 1.0, 36.0, 8.0, .5)
        n_sims = st.select_slider("Monte Carlo simulations", options=[1000,5000,10000,25000,50000], value=10000)
        submitted = st.form_submit_button("Run risk assessment")

    if submitted:
        x = pd.DataFrame([{
            "budget_egp":budget, "planned_duration_months":duration, "complexity_1_5":complexity,
            "design_maturity_pct":design, "procurement_risk_1_5":procurement,
            "subcontractor_risk_1_5":subcontractor, "site_readiness_pct":site,
            "change_order_rate_pct":change, "payment_delay_days":payment, "resource_gap_pct":resource,
            "approval_delay_days":approval, "quality_ncr_rate":ncr, "safety_incidents":safety,
            "long_lead_material_months":lead, "project_type":project_type, "discipline":discipline
        }], columns=FEATURES)
        delay_model_path = MODELS/"delay_days_model.joblib"
        cost_model_path = MODELS/"cost_overrun_model.joblib"
        model_delay = joblib.load(delay_model_path) if delay_model_path.exists() else None
        model_cost = joblib.load(cost_model_path) if cost_model_path.exists() else None
        if model_delay and model_cost:
            base_delay = max(0, float(model_delay.predict(x)[0]))
            base_cost_pct = float(model_cost.predict(x)[0])
        else:
            # Transparent fallback for Monte Carlo prototype if models have not been trained.
            risk_signal = (complexity*.28 + procurement*.25 + subcontractor*.2 + change*.035
                           + payment*.008 + resource*.025 + approval*.012
                           - (design-50)*.008 - (site-50)*.008 + ncr*.04)
            base_delay = max(0, 6 + 2.0*risk_signal)
            base_cost_pct = max(-2, 1.2 + 1.8*risk_signal + change*.12 + payment*.015)
            st.info("Models not found; using a transparent illustrative baseline. Run `python train_model.py` to train baseline models.")
        rng = np.random.default_rng(2026)
        # Uncertainty distributions are explicit assumptions and should be elicited from project experts.
        delay_noise = rng.triangular(-0.25, 0.0, 0.9, n_sims) * max(base_delay, 5)
        delay_sims = np.maximum(0, base_delay + delay_noise + rng.normal(0, max(2, duration*.25), n_sims))
        cost_sims_pct = base_cost_pct + rng.triangular(-2.0, 0.0, 8.0, n_sims) + rng.normal(0, 2.5, n_sims)
        cost_sims_egp = budget * cost_sims_pct / 100
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Expected delay", f"{np.mean(delay_sims):.1f} days")
        c2.metric("P90 delay", f"{np.percentile(delay_sims,90):.1f} days")
        c3.metric("Expected cost overrun", f"{np.mean(cost_sims_pct):.1f}%")
        c4.metric("P90 cost exposure", f"EGP {np.percentile(cost_sims_egp,90)/1e6:.2f}M")
        st.plotly_chart(px.histogram(pd.DataFrame({"Delay days":delay_sims}), x="Delay days", nbins=45,
                                     title="Monte Carlo schedule-delay distribution"), use_container_width=True)
        st.plotly_chart(px.histogram(pd.DataFrame({"Cost overrun (%)":cost_sims_pct}), x="Cost overrun (%)", nbins=45,
                                     title="Monte Carlo cost-overrun distribution"), use_container_width=True)
        summary = pd.DataFrame({
            "Metric":["Delay days","Cost overrun (%)","Cost exposure (EGP)"],
            "P10":[np.percentile(delay_sims,10),np.percentile(cost_sims_pct,10),np.percentile(cost_sims_egp,10)],
            "P50":[np.percentile(delay_sims,50),np.percentile(cost_sims_pct,50),np.percentile(cost_sims_egp,50)],
            "P80":[np.percentile(delay_sims,80),np.percentile(cost_sims_pct,80),np.percentile(cost_sims_egp,80)],
            "P90":[np.percentile(delay_sims,90),np.percentile(cost_sims_pct,90),np.percentile(cost_sims_egp,90)],
            "P95":[np.percentile(delay_sims,95),np.percentile(cost_sims_pct,95),np.percentile(cost_sims_egp,95)]
        })
        st.dataframe(summary.round(2), use_container_width=True, hide_index=True)
        st.download_button("Download Monte Carlo summary CSV", summary.to_csv(index=False).encode(), "mep_monte_carlo_summary.csv", "text/csv")
