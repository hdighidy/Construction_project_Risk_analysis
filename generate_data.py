"""Generate illustrative synthetic MEP construction project risk data.
IMPORTANT: synthetic data is for prototyping only, not calibrated to real projects.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
OUT = Path("data")
OUT.mkdir(exist_ok=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main(n_projects=350, months=12):
    rng = np.random.default_rng(SEED)
    rows, monthly = [], []
    disciplines = ["HVAC", "Firefighting", "Plumbing", "Electrical", "ELV/BMS", "Multi-discipline"]
    project_types = ["Residential", "Commercial", "Hospital", "Industrial", "Infrastructure"]
    for i in range(n_projects):
        pid = f"MEP-{i+1:04d}"
        status = rng.choice(["Completed", "Active"], p=[0.42, 0.58])
        budget = float(rng.uniform(8e6, 450e6))
        duration = int(rng.integers(6, 37))
        complexity = float(rng.uniform(1, 5))
        design_maturity = float(rng.uniform(35, 100))
        procurement_risk = float(rng.uniform(1, 5))
        subcontractor_risk = float(rng.uniform(1, 5))
        site_readiness = float(rng.uniform(35, 100))
        change_rate = float(rng.uniform(0, 18))
        payment_delay = float(rng.uniform(0, 120))
        resource_gap = float(rng.uniform(0, 35))
        approval_delay = float(rng.uniform(0, 45))
        quality_ncr_rate = float(rng.uniform(0, 12))
        safety_incidents = int(rng.poisson(1.0))
        lead_time = float(rng.uniform(2, 24))
        # Hidden synthetic risk mechanism. Values are not a real-world calibrated model.
        latent = (-2.2 + .34*complexity + .28*procurement_risk + .23*subcontractor_risk
                  + .018*change_rate + .013*payment_delay + .025*resource_gap
                  + .018*approval_delay + .06*quality_ncr_rate
                  - .012*(design_maturity-60) - .01*(site_readiness-60))
        p_delay = float(sigmoid(latent))
        delay_days = int(max(0, rng.normal(8 + 42*p_delay + .35*approval_delay, 12)))
        cost_overrun_pct = float(max(-3, rng.normal(1 + 12*p_delay + .07*change_rate + .05*payment_delay/10, 5)))
        quality_risk = float(np.clip(.10*quality_ncr_rate + .10*complexity + rng.normal(0, .5), 0, 5))
        safety_risk = float(np.clip(.55*safety_incidents + .3*complexity + rng.normal(0, .6), 0, 5))
        overall = float(np.clip(20*p_delay + 2.1*cost_overrun_pct + 4*quality_risk + 3*safety_risk, 0, 100))
        final_delay = delay_days if status == "Completed" else np.nan
        final_cost_overrun = cost_overrun_pct if status == "Completed" else np.nan
        rows.append({
            "project_id": pid, "project_name": f"MEP Project {i+1:03d}",
            "project_type": rng.choice(project_types), "discipline": rng.choice(disciplines),
            "status": status, "budget_egp": round(budget, 2), "planned_duration_months": duration,
            "complexity_1_5": round(complexity, 2), "design_maturity_pct": round(design_maturity, 1),
            "procurement_risk_1_5": round(procurement_risk, 2),
            "subcontractor_risk_1_5": round(subcontractor_risk, 2),
            "site_readiness_pct": round(site_readiness, 1), "change_order_rate_pct": round(change_rate, 2),
            "payment_delay_days": round(payment_delay, 1), "resource_gap_pct": round(resource_gap, 2),
            "approval_delay_days": round(approval_delay, 1), "quality_ncr_rate": round(quality_ncr_rate, 2),
            "safety_incidents": safety_incidents, "long_lead_material_months": round(lead_time, 1),
            "predicted_delay_probability": round(p_delay, 4), "delay_days_final": final_delay,
            "cost_overrun_pct_final": round(final_cost_overrun, 2) if pd.notna(final_cost_overrun) else np.nan,
            "overall_risk_score": round(overall, 1)
        })
        for m in range(1, months+1):
            # Historical monthly snapshots; completed projects have a complete history too.
            progress = min(100, 100*m/max(duration, 1) + rng.normal(0, 7))
            schedule_var = max(-20, rng.normal(2 + 0.22*approval_delay + 0.18*resource_gap, 8))
            cost_var = rng.normal(cost_overrun_pct*.55, 3)
            open_rfis = int(max(0, rng.poisson(2 + complexity + approval_delay/15)))
            pending_materials = int(max(0, rng.poisson(2 + procurement_risk + lead_time/5)))
            ncr_open = int(max(0, rng.poisson(quality_ncr_rate/2)))
            monthly.append({
                "project_id": pid, "month_index": m, "progress_pct": round(float(np.clip(progress, 0, 100)), 1),
                "schedule_variance_days": round(float(schedule_var), 1),
                "cost_variance_pct": round(float(cost_var), 2), "open_rfis": open_rfis,
                "pending_material_approvals": pending_materials, "open_ncrs": ncr_open,
                "manpower_gap_pct": round(float(np.clip(resource_gap+rng.normal(0, 5), 0, 100)), 1),
                "procurement_delay_days": round(float(max(0, lead_time*2+rng.normal(0, 5))), 1),
            })
    pd.DataFrame(rows).to_csv(OUT/"mep_projects.csv", index=False)
    pd.DataFrame(monthly).to_csv(OUT/"mep_monthly_history.csv", index=False)
    print(f"Created {len(rows)} projects and {len(monthly)} monthly records in {OUT.resolve()}")

if __name__ == "__main__":
    main()
