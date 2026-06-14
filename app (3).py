
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Airline Loyalty Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── DARK THEME CSS ────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0f0f1a; }
[data-testid="stSidebar"]          { background-color: #1a1a2e; }
[data-testid="metric-container"]   { background-color: #1a1a2e; border-radius: 10px; padding: 10px; }
h1, h2, h3, p, label              { color: white !important; }
.stDataFrame                       { background-color: #1a1a2e; }
div[data-testid="stMetricValue"]   { color: white; font-size: 28px; }
div[data-testid="stMetricLabel"]   { color: #aaaaaa; }
.stSelectbox label, .stMultiSelect label { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("master_final.csv")

df = load_data()

# ── SIDEBAR FILTERS ───────────────────────────────────────────
st.sidebar.markdown("## ✈️ Loyalty Intelligence")
st.sidebar.markdown("---")

seg_filter  = st.sidebar.multiselect(
    "Segment", df["segment"].unique(),
    default=df["segment"].unique()
)
risk_filter = st.sidebar.multiselect(
    "Risk Tier", ["Critical", "High", "Medium", "Low"],
    default=["Critical", "High", "Medium", "Low"]
)
card_filter = st.sidebar.multiselect(
    "Card Tier", df["Loyalty Card"].unique(),
    default=df["Loyalty Card"].unique()
)
clv_min, clv_max = st.sidebar.slider(
    "CLV Range ($)",
    int(df["CLV"].min()), int(df["CLV"].max()),
    (int(df["CLV"].min()), int(df["CLV"].max()))
)

filtered = df[
    df["segment"].isin(seg_filter) &
    df["risk_tier"].isin(risk_filter) &
    df["Loyalty Card"].isin(card_filter) &
    df["CLV"].between(clv_min, clv_max)
]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Showing:** {len(filtered):,} of {len(df):,} customers")

# ── HEADER ────────────────────────────────────────────────────
st.markdown("# ✈️ Airline Loyalty Intelligence Dashboard")
st.markdown("*Real-time churn prediction and retention action engine*")
st.markdown("---")

# ── TOP METRICS ───────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Members",    f"{len(filtered):,}")
c2.metric("Churn Rate",       f"{filtered['churn'].mean()*100:.1f}%")
c3.metric("Critical Risk",    f"{(filtered['risk_tier']=='Critical').sum():,}",
          delta="Needs immediate action", delta_color="inverse")
c4.metric("High Risk",        f"{(filtered['risk_tier']=='High').sum():,}")
c5.metric("Avg CLV",          f"${filtered['CLV'].mean():,.0f}")

st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "⚠️ At-Risk Customers",
    "👥 Segments",
    "📋 Retention Playbook"
])

SEG_COLORS = {
    "Core Flyers"       : "#4CAF50",
    "Drifting Loyalists": "#FF9800",
    "Silent Attritors"  : "#F44336",
    "Never Active"      : "#9E9E9E"
}

RISK_COLORS = {
    "Critical": "#F44336",
    "High"    : "#FF9800",
    "Medium"  : "#FFC107",
    "Low"     : "#4CAF50"
}

# ════════════════════════════════════════
# TAB 1: OVERVIEW
# ════════════════════════════════════════
with tab1:
    col1, col2 = st.columns(2)

    # Donut Chart
    seg_dist = filtered["segment"].value_counts().reset_index()
    seg_dist.columns = ["segment", "count"]
    fig_donut = go.Figure(go.Pie(
        labels=seg_dist["segment"],
        values=seg_dist["count"],
        hole=0.6,
        marker_colors=[SEG_COLORS.get(s, "gray") for s in seg_dist["segment"]]
    ))
    fig_donut.update_layout(
        title="Segment Distribution",
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        font_color="white", height=380,
        legend=dict(font=dict(color="white"))
    )
    col1.plotly_chart(fig_donut, use_container_width=True)

    # Churn Risk Distribution
    fig_hist = px.histogram(
        filtered, x="churn_prob", color="segment",
        nbins=25, title="Churn Risk Distribution",
        color_discrete_map=SEG_COLORS,
        barmode="overlay", opacity=0.75,
        labels={"churn_prob": "Churn Probability", "count": "Customers"}
    )
    fig_hist.update_layout(
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        font_color="white", height=380
    )
    col2.plotly_chart(fig_hist, use_container_width=True)

    col3, col4 = st.columns(2)

    # Churn Rate by Segment Bar
    seg_churn = filtered.groupby("segment").agg(
        churn_rate=("churn", "mean"),
        count=("churn", "count")
    ).reset_index()
    seg_churn["churn_pct"] = (seg_churn["churn_rate"] * 100).round(1)

    fig_bar = px.bar(
        seg_churn, x="segment", y="churn_pct",
        color="segment", color_discrete_map=SEG_COLORS,
        title="Churn Rate by Segment (%)",
        text="churn_pct",
        labels={"churn_pct": "Churn %", "segment": "Segment"}
    )
    fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_bar.update_layout(
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        font_color="white", height=350, showlegend=False
    )
    col3.plotly_chart(fig_bar, use_container_width=True)

    # CLV vs Churn Prob Scatter
    sample = filtered.sample(min(2000, len(filtered)), random_state=42)
    fig_scatter = px.scatter(
        sample, x="engagement_ratio", y="CLV",
        color="risk_tier",
        color_discrete_map=RISK_COLORS,
        title="CLV vs Engagement (by Risk Tier)",
        opacity=0.6, size_max=5,
        labels={"engagement_ratio": "Engagement Ratio", "CLV": "CLV ($)"}
    )
    fig_scatter.update_layout(
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        font_color="white", height=350
    )
    col4.plotly_chart(fig_scatter, use_container_width=True)

# ════════════════════════════════════════
# TAB 2: AT-RISK CUSTOMERS
# ════════════════════════════════════════
with tab2:
    st.markdown("### ⚠️ High Priority Customers - Immediate Action Required")

    risk_level = st.radio(
        "Filter by Risk",
        ["All", "Critical Only", "Critical + High"],
        horizontal=True
    )

    if risk_level == "Critical Only":
        at_risk = filtered[filtered["risk_tier"] == "Critical"]
    elif risk_level == "Critical + High":
        at_risk = filtered[filtered["risk_tier"].isin(["Critical", "High"])]
    else:
        at_risk = filtered[filtered["churn_prob"] > 0.3]

    at_risk = at_risk.sort_values("churn_prob", ascending=False)

    st.markdown(f"**{len(at_risk):,} customers need attention**")

    display_cols = {
        "Loyalty Number" : "ID",
        "Loyalty Card"   : "Card",
        "CLV"            : "CLV ($)",
        "segment"        : "Segment",
        "churn_prob"     : "Churn Risk",
        "risk_tier"      : "Risk Tier",
        "action"         : "Action",
        "what_to_do"     : "What To Do",
        "when_to_act"    : "When"
    }

    show_df = at_risk[list(display_cols.keys())].copy()
    show_df["churn_prob"] = (show_df["churn_prob"] * 100).round(1).astype(str) + "%"
    show_df["CLV ($)"]    = show_df["CLV"].apply(lambda x: f"${x:,.0f}")
    show_df = show_df.drop(columns=["CLV"]).rename(columns=display_cols)

    st.dataframe(show_df, use_container_width=True, height=450)

    # Download button
    csv = at_risk.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download At-Risk List",
        data=csv,
        file_name="at_risk_customers.csv",
        mime="text/csv"
    )

# ════════════════════════════════════════
# TAB 3: SEGMENTS
# ════════════════════════════════════════
with tab3:
    st.markdown("### 👥 Segment Deep Dive")

    seg_cols = st.columns(4)
    segs = ["Core Flyers", "Drifting Loyalists", "Silent Attritors", "Never Active"]
    emojis = ["🏆", "🟡", "🔴", "💤"]

    for col, seg, emoji in zip(seg_cols, segs, emojis):
        grp = filtered[filtered["segment"] == seg]
        if len(grp) == 0:
            continue
        color = SEG_COLORS[seg]
        col.markdown(f"""
        <div style="background:#1a1a2e; border-left:4px solid {color};
                    border-radius:10px; padding:15px; margin:5px">
            <h4 style="color:{color}; margin:0">{emoji} {seg}</h4>
            <h2 style="margin:5px 0; color:white">{len(grp):,}</h2>
            <p style="color:#aaa; margin:0">
                {len(grp)/len(df)*100:.1f}% of base<br>
                {grp["churn"].mean()*100:.1f}% churn rate
            </p>
            <hr style="border-color:#333; margin:10px 0">
            <small style="color:#ddd">
                Avg flights: {grp["total_flights"].mean():.1f}<br>
                Engagement: {grp["engagement_ratio"].mean()*100:.1f}%<br>
                Avg CLV: ${grp["CLV"].mean():,.0f}<br>
                Avg recency: {grp["recency_months"].mean():.1f} mo
            </small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Segment selected for deep dive
    selected_seg = st.selectbox("Select Segment for Deep Dive", segs)
    seg_data = filtered[filtered["segment"] == selected_seg]

    sc1, sc2 = st.columns(2)

    # CLV Distribution for segment
    fig_clv = px.histogram(
        seg_data, x="CLV", nbins=30,
        title=f"CLV Distribution - {selected_seg}",
        color_discrete_sequence=[SEG_COLORS[selected_seg]]
    )
    fig_clv.update_layout(
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        font_color="white", height=300
    )
    sc1.plotly_chart(fig_clv, use_container_width=True)

    # Recency Distribution
    fig_rec = px.histogram(
        seg_data, x="recency_months", nbins=20,
        title=f"Recency Distribution - {selected_seg}",
        color_discrete_sequence=[SEG_COLORS[selected_seg]]
    )
    fig_rec.update_layout(
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        font_color="white", height=300
    )
    sc2.plotly_chart(fig_rec, use_container_width=True)

# ════════════════════════════════════════
# TAB 4: RETENTION PLAYBOOK
# ════════════════════════════════════════
with tab4:
    st.markdown("### 📋 Retention Playbook - Segment Wise Actions")

    playbooks = {
        "Core Flyers": {
            "color"   : "#4CAF50",
            "risk"    : "Low Risk",
            "strategy": "Loyalty Deepening",
            "trigger" : "Engagement ratio drops below 40% for 2 consecutive months",
            "action"  : "Tier upgrade fast-track + companion ticket on milestone flight",
            "timeline": "Quarterly check; act within 7 days of trigger",
            "kpi"     : "Tier upgrade rate + Avg flights per quarter",
            "expected": "Maintain churn < 5%"
        },
        "Drifting Loyalists": {
            "color"   : "#FF9800",
            "risk"    : "Medium Risk",
            "strategy": "Re-engagement Campaign",
            "trigger" : "Churn prob > 40% AND no flights in 3+ months",
            "action"  : "2x points on next booking + personalized route email based on past travel",
            "timeline": "Within 30 days of inactivity signal",
            "kpi"     : "% flying within 60 days of outreach",
            "expected": "25% re-activation rate"
        },
        "Silent Attritors": {
            "color"   : "#F44336",
            "risk"    : "Critical",
            "strategy": "Last-Mile Save or Graceful Exit",
            "trigger" : "Recency > 24 months AND churn prob > 70%",
            "action"  : "Aurora/Nova: Direct call + 3000 points + 15% off. Star: Email + 1500 points",
            "timeline": "One attempt in 7 days; archive after 30 days no response",
            "kpi"     : "Win-back rate vs cost per recovered customer vs CLV",
            "expected": "8% win-back; pause spend on non-responders"
        },
        "Never Active": {
            "color"   : "#9E9E9E",
            "risk"    : "Activation Needed",
            "strategy": "Welcome Activation",
            "trigger" : "Joined but never flew",
            "action"  : "500 bonus points on first booking + onboarding email series",
            "timeline": "Month 1, 2, 3 after joining",
            "kpi"     : "First flight conversion rate",
            "expected": "20% first flight activation"
        }
    }

    p_cols = st.columns(2)
    for i, (seg, pb) in enumerate(playbooks.items()):
        col = p_cols[i % 2]
        grp = filtered[filtered["segment"] == seg]
        col.markdown(f"""
        <div style="background:#1a1a2e; border-radius:12px;
                    border-top:4px solid {pb['color']};
                    padding:20px; margin:10px 0">
            <div style="display:flex; justify-content:space-between; align-items:center">
                <h3 style="color:{pb['color']}; margin:0">{seg}</h3>
                <span style="background:{pb['color']}33; color:{pb['color']};
                             padding:4px 12px; border-radius:20px; font-size:12px">
                    {pb['risk']}
                </span>
            </div>
            <p style="color:#888; margin:5px 0">{len(grp):,} customers · {grp["churn"].mean()*100:.1f}% churn</p>
            <hr style="border-color:#333">
            <b style="color:white">📌 Strategy:</b>
            <p style="color:#ddd">{pb['strategy']}</p>
            <b style="color:white">🎯 Trigger:</b>
            <p style="color:#ddd">{pb['trigger']}</p>
            <b style="color:white">⚡ Action:</b>
            <p style="color:#ddd">{pb['action']}</p>
            <b style="color:white">⏰ Timeline:</b>
            <p style="color:#ddd">{pb['timeline']}</p>
            <b style="color:white">📊 KPI:</b>
            <p style="color:#ddd">{pb['kpi']}</p>
            <div style="background:{pb['color']}22; border-radius:8px; padding:8px; margin-top:10px">
                <b style="color:{pb['color']}">Expected: </b>
                <span style="color:#ddd">{pb['expected']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Customer Lookup
    st.markdown("---")
    st.markdown("### 🔍 Customer Lookup")
    cust_id = st.number_input("Enter Customer ID (Loyalty Number)",
                               min_value=int(df["Loyalty Number"].min()),
                               max_value=int(df["Loyalty Number"].max()),
                               step=1)

    if st.button("🔍 Search Customer"):
        cust = df[df["Loyalty Number"] == cust_id]
        if len(cust) == 0:
            st.error("Customer not found!")
        else:
            cust = cust.iloc[0]
            risk_color = RISK_COLORS.get(cust["risk_tier"], "gray")
            st.markdown(f"""
            <div style="background:#1a1a2e; border-radius:12px;
                        border-left:5px solid {risk_color}; padding:20px">
                <h3 style="color:white">Customer #{int(cust['Loyalty Number'])}</h3>
                <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:15px">
                    <div>
                        <p style="color:#aaa; margin:0">Card Tier</p>
                        <h4 style="color:white; margin:0">{cust['Loyalty Card']}</h4>
                    </div>
                    <div>
                        <p style="color:#aaa; margin:0">CLV</p>
                        <h4 style="color:white; margin:0">${cust['CLV']:,.0f}</h4>
                    </div>
                    <div>
                        <p style="color:#aaa; margin:0">Segment</p>
                        <h4 style="color:{SEG_COLORS.get(cust['segment'],'white')}; margin:0">{cust['segment']}</h4>
                    </div>
                    <div>
                        <p style="color:#aaa; margin:0">Churn Risk</p>
                        <h4 style="color:{risk_color}; margin:0">{cust['churn_prob']*100:.1f}% ({cust['risk_tier']})</h4>
                    </div>
                    <div>
                        <p style="color:#aaa; margin:0">Last Flight</p>
                        <h4 style="color:white; margin:0">{cust['recency_months']:.0f} months ago</h4>
                    </div>
                    <div>
                        <p style="color:#aaa; margin:0">Engagement</p>
                        <h4 style="color:white; margin:0">{cust['engagement_ratio']*100:.1f}%</h4>
                    </div>
                </div>
                <hr style="border-color:#333; margin:15px 0">
                <b style="color:{risk_color}">Recommended Action: {cust['action']}</b>
                <p style="color:#ddd; margin:5px 0">{cust['what_to_do']}</p>
                <p style="color:#aaa; margin:0">⏰ {cust['when_to_act']}</p>
            </div>
            """, unsafe_allow_html=True)
