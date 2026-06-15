## 📌 Project Overview

This project analyzes loyalty behavior of **16,737 Canadian airline members** (activity data 2017–2018) to help marketing teams **proactively retain customers before they disengage**.

Most loyalty programs are reactive — teams learn a high-value member has stopped flying only after it's too late to act. This project makes retention **proactive**: every member receives a specific action, a trigger condition, and a timeline based on their behavior.

### Three Core Deliverables

| # | Deliverable | Description |
|---|-------------|-------------|
| 1 | **Churn Prediction** | 3-model ensemble (XGBoost + LightGBM + Random Forest) assigns each member a churn probability with **0.998 ROC-AUC** |
| 2 | **Customer Segmentation** | Behavioral clustering into 4 actionable segments, each with a distinct retention strategy |
| 3 | **Retention Playbook** | Segment-specific interventions with triggers, timelines, KPIs — packaged into an interactive Streamlit dashboard |

---

## 📊 Key Metrics at a Glance

| Metric | Value |
|--------|-------|
| Total Members Analyzed | 16,737 |
| Flight Activity Records | 392,936 monthly rows |
| Overall Churn Rate | 13.0% |
| Critical Risk Members | 668 (4.0%) |
| High Risk Members | 236 (1.4%) |
| ML Models in Ensemble | 3 (XGBoost · LightGBM · Random Forest) |
| Ensemble ROC-AUC | **0.998** |
| Ensemble F1 Score | **0.93** |
| Ensemble Precision | **0.925** |
| Ensemble Recall | **0.936** |

---

## 🧩 Customer Segments

| Segment | Members | % of Base | Avg CLV | Churn Rate | Strategy |
|---------|---------|-----------|---------|------------|----------|
| 🏆 **Core Flyers** | 10,349 | 61.8% | $7,967 | 4.6% | Loyalty deepening + tier fast-track |
| 🟡 **Drifting Loyalists** | 3,467 | 20.7% | $7,860 | 16.7% | 2× points re-engagement within 30 days |
| 🔴 **Silent Attritors** | 1,351 | 8.1% | $8,057 | 12.9% | Last-mile save — direct call + 3,000 bonus points |
| 💤 **Never Active** | 1,570 | 9.4% | $8,361 | 60.6% | 500 bonus points on first booking + onboarding series |

> 💡 **Key finding:** Drifting Loyalists have a churn rate **3.6× higher** than Core Flyers yet are still reachable — this is the highest-ROI segment for retention spend.

---

## 🤖 ML Model Performance

Three models were tuned via **5-fold cross-validation** with SMOTE (to handle class imbalance at 8.1% churn rate), then combined into a **soft-voting ensemble**:

| Model | ROC-AUC | F1 Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| XGBoost | 0.979 | 0.803 | 0.829 | 0.779 |
| LightGBM | 0.997 | 0.943 | 0.966 | 0.921 |
| Random Forest | 0.999 | 0.963 | 0.949 | 0.977 |
| **Ensemble (final)** | **0.998** | **0.930** | **0.925** | **0.936** |

Each member's ensemble probability maps to a **Risk Tier**:
- 🔴 **Critical** — churn probability ≥ 70% → 668 members
- 🟠 **High** — 50–70% → 236 members
- 🟡 **Medium** — 30–50% → 344 members
- 🟢 **Low** — < 30% → 15,489 members

---

## 🔬 Feature Engineering (29 Features)

Features were built **strictly from the pre-prediction window (Jan 2017 – Jun 2018)** to avoid hindsight bias — the model can be deployed to score future members, not just explain history.

| Feature Category | Examples |
|-----------------|----------|
| **Flight Volume** | `total_flights`, `total_distance`, `pts_earned`, `active_months`, `zero_months` |
| **Recency & Engagement** | `recency_months`, `engagement_ratio`, `redemption_rate`, `avg_dist_per_flight` |
| **Trend** | `activity_trend` (H1 2018 vs. pro-rated 2017 baseline) |
| **Seasonality** | `flights_fall`, `flights_spring`, `flights_summer`, `flights_winter` |
| **Demographics** | `CLV`, `salary`, `tenure_months`, `card_rank`, `loyalty_card` |

> **Top churn signal:** Engagement ratio dropping below 40% precedes formal cancellation by ~6 months.

---

## 🗂️ Repo Structure

```
airline-loyalty-dashboard/
├── app.py                    ← Streamlit dashboard (main entry point)
├── master_final.csv          ← Processed dataset with predictions (47 columns)
├── airline_project.ipynb     ← Full ML pipeline: EDA → cleaning → features → models → output
├── requirements.txt          ← Python dependencies
└── README.md                 ← This file
```

---

## ⚙️ Setup & Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/Raj-RR2004/airline-loyalty-dashboard.git
cd airline-loyalty-dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the dashboard
```bash
streamlit run app.py
```
Opens at `https://airline-loyalty-dashboard.streamlit.app/` automatically.

---

## 📦 Requirements

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
```

---

## 🖥️ Dashboard Features

The dashboard has **4 tabs** with sidebar filters (Segment · Risk Tier · Card Tier · CLV range):

| Tab | What It Shows |
|-----|---------------|
| 📊 **Overview** | Segment donut chart · Churn distribution histogram · CLV vs. engagement scatter |
| ⚠️ **At-Risk** | Filterable table of Critical/High risk customers with downloadable CSV |
| 👥 **Segments** | Deep dive into each segment — CLV distributions, recency, flight activity |
| 📋 **Retention Playbook** | Segment-wise action cards + individual customer lookup by Loyalty ID |

---

## 🔍 Data Pipeline (Notebook)

`airline_project.ipynb` covers the full pipeline:

1. **Data loading** — merge Customer Loyalty History + Customer Flight Activity + Calendar CSV (3 sources, 409,673 total rows)
2. **Cleaning** — salary imputation by loyalty tier, negative salary correction (abs value), cancellation date handling, single-month SD → 0
3. **Churn definition** — formal cancellation OR behavioral inactivity: year-over-year drop-off (flew 2017, zero in 2018) OR within-2018 drop-off (flew H1, zero H2)
4. **Feature engineering** — 29 features across volume, recency, trend, seasonality, demographics; feature window = Jan 2017–Jun 2018
5. **Modeling** — XGBoost, LightGBM, Random Forest → 5-fold CV with SMOTE → soft-voting ensemble
6. **Segmentation** — K-Means (K=3, chosen for business clarity over K=2 which scored marginally higher on silhouette) + Never Active as separate group
7. **Output** — `master_final.csv` with 47 columns: `segment`, `churn_prob`, `risk_tier`, `action`, `what_to_do`, `when_to_act`, and all engineered features

---

## 📂 master_final.csv — Column Reference (47 cols)

Key columns output by the pipeline:

```
Loyalty Number · Gender · Education · Salary · Marital Status
Loyalty Card (Star/Aurora/Nova) · CLV · Enrollment Type
tenure_months · card_rank · churn · is_cancelled · behavioral_churn
never_flew · total_flights · total_distance · total_pts_earned
total_pts_redeemed · total_dollar_redeem · active_months · zero_months
max_flights_month · avg_flights_month · std_flights_month
recency_months · engagement_ratio · redemption_rate · avg_dist_per_flight
flights_2017 · flights_h1_2018 · activity_trend
fall · spring · summer · winter
gender_enc · edu_enc · marital_enc · promo_enc
segment · churn_prob · churn_predicted · risk_tier
priority · action · what_to_do · when_to_act
```

---

## 💡 Key Findings

1. **Behavior, not demographics, drives churn** — recency, engagement ratio, and activity trend are the dominant signals, far outweighing salary, education, or marital status.

2. **Formal cancellation understates true churn** — of 2,175 churned members, only 2,067 formally cancelled; 553+ disengaged silently with zero formal notice.

3. **Star card members have the highest churn risk** despite the lowest CLV — tier design may be inadvertently discouraging loyalty.

4. **CLV is misleading for the Never Active segment** — these 1,570 members show an avg CLV of $8,361 (highest of all segments) yet a 60.6% churn rate, indicating CLV overstates the realized value of dormant accounts.

5. **Silent Attritors show an unusual redemption spike** before disengaging — consistent with a "redeem large reward → leave" pattern that can serve as an early warning signal.

6. **Engagement ratio < 40%** is the strongest early churn indicator — it precedes formal cancellation by approximately 6 months.

---

## 🎯 Retention Recommendations

| # | Segment | Action | Target |
|---|---------|--------|--------|
| 1 | Drifting Loyalists | Automated trigger-based 2× points campaign for 537 elevated-risk members | 25% reactivation |
| 2 | Silent Attritors | Single-touch outreach to 58 highest-CLV members — direct call + 3,000 bonus points | 8% win-back |
| 3 | Never Active | 500 bonus points on first booking + structured onboarding series | 20% first-flight conversion |

---


## 📄 License

This project was developed as an academic analytics case study. Dataset source: Canadian airline loyalty program (2012–2018). For educational and research use only.

---


