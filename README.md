# MEP Construction Project Risk Intelligence

A Python/Streamlit prototype for MEP construction project portfolio monitoring, active-project history review, baseline ML estimates, and Monte Carlo uncertainty analysis for new projects.

## Scope
- Synthetic portfolio data for HVAC, firefighting, plumbing, electrical, ELV/BMS and multi-discipline projects.
- Portfolio monitoring: active/completed projects, risk score distribution, budget exposure and attention list.
- Project deep dive: monthly progress, schedule/procurement indicators, RFIs, material approvals and NCRs.
- New-project assessment: user-entered assumptions, baseline ML estimates for delay days/cost overrun, Monte Carlo distributions and P10/P50/P80/P90/P95 outputs.

## Quick start (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python generate_data.py
python train_model.py
streamlit run app.py
```

## Files
- `generate_data.py`: creates `data/mep_projects.csv` and `data/mep_monthly_history.csv`.
- `train_model.py`: trains baseline Random Forest regression pipelines and saves them under `models/`.
- `app.py`: Streamlit dashboards and Monte Carlo simulation.
- `data/`: generated at runtime; do not treat generated data as actual project records.

## Important methodological limitations
1. All data generated here is synthetic and illustrative. The generated target mechanism is not calibrated to Egyptian MEP projects or any specific contractor.
2. ML performance on synthetic data is not evidence of generalization. Replace with completed projects and actual final outcomes before business use.
3. Monte Carlo distributions are provisional assumptions, not learned empirical distributions. Have planning, commercial, procurement, technical-office, QA/QC and project-control experts define distributions and correlations using historical evidence.
4. Do not train on features unavailable at the prediction date. For a real model, define a clear prediction cut-off, exclude leakage, use time-based and preferably project-level validation, and report MAE/RMSE for regression plus calibration/coverage for risk estimates.
5. Treat schedule, cost, quality, procurement, technical, resource and safety risks as separate dimensions. A single overall score is a screening aid, not a substitute for a risk register, quantitative risk analysis or expert review.
6. Real project data should be access-controlled, validated, versioned and auditable. Include data dictionary, units, source owner, update frequency, missingness policy and outcome definitions.

## Recommended production next steps
- Agree definitions: delay days, cost overrun %, risk event probability/impact, project completion and prediction horizon.
- Build a project/activity/material/vendor/approval/RFI/NCR/resource/payment data model.
- Backtest using historical project snapshots with no future information.
- Fit distributions from residuals and historical risk events; model correlations between schedule, procurement, resource and cost.
- Add scenario comparison, risk register ownership, mitigation action, due date and post-mitigation residual risk.
- Add authentication, role-based access, logging and automated data refresh before deployment.
