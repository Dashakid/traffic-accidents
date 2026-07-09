"""
Streamlit Application: Motorcycle Safety — A Story in the Data
NHTSA FARS (US) + WHO Global Status Report on Road Safety

Rewritten for clarity: guided narrative, real state names, plain-language
insights, and clean, digestible visuals.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import gc
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Motorcycle Safety Analysis",
    page_icon="🏍️",
    layout="wide",
    # Collapsed by default so phone users see the content first, not the sidebar
    initial_sidebar_state="collapsed"
)

# ---- Mobile-friendly styling (works on iPhone / small screens) --------------
st.markdown(
    """
    <style>
      /* Tighten page padding on small screens so charts use full width */
      @media (max-width: 640px) {
        .block-container { padding: 1rem 0.75rem 3rem 0.75rem !important; }
        h1 { font-size: 1.5rem !important; }
        h2, h3 { font-size: 1.15rem !important; }
        /* Let metric numbers shrink instead of overflowing */
        [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
        [data-testid="stMetricLabel"] { font-size: 0.75rem !important; }
        /* Make tab labels wrap/scroll nicely */
        .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; flex-wrap: wrap; }
      }
      /* Prevent horizontal scrolling of the whole page on phones */
      .main .block-container { max-width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# REFERENCE DATA: FARS STATE CODES → STATE NAMES
# Source: NHTSA FARS Analytical User's Manual (GSA / FIPS state codes)
# ============================================================================
STATE_CODE_TO_NAME = {
    1: "Alabama", 2: "Alaska", 4: "Arizona", 5: "Arkansas", 6: "California",
    8: "Colorado", 9: "Connecticut", 10: "Delaware", 11: "District of Columbia",
    12: "Florida", 13: "Georgia", 15: "Hawaii", 16: "Idaho", 17: "Illinois",
    18: "Indiana", 19: "Iowa", 20: "Kansas", 21: "Kentucky", 22: "Louisiana",
    23: "Maine", 24: "Maryland", 25: "Massachusetts", 26: "Michigan",
    27: "Minnesota", 28: "Mississippi", 29: "Missouri", 30: "Montana",
    31: "Nebraska", 32: "Nevada", 33: "New Hampshire", 34: "New Jersey",
    35: "New Mexico", 36: "New York", 37: "North Carolina", 38: "North Dakota",
    39: "Ohio", 40: "Oklahoma", 41: "Oregon", 42: "Pennsylvania",
    44: "Rhode Island", 45: "South Carolina", 46: "South Dakota",
    47: "Tennessee", 48: "Texas", 49: "Utah", 50: "Vermont", 51: "Virginia",
    53: "Washington", 54: "West Virginia", 55: "Wisconsin", 56: "Wyoming",
    72: "Puerto Rico"
}

# US Postal abbreviations for the choropleth map (mainland + DC)
STATE_NAME_TO_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
}

DAY_NAMES = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday",
             5: "Thursday", 6: "Friday", 7: "Saturday"}

# State population weights (for realistic fatality distribution)
# Approximate motorcycle accident share by state (larger/more urban states weighted higher)
STATE_WEIGHTS = {
    1: 1.5, 2: 0.2, 4: 1.2, 5: 0.6, 6: 3.5,  # AL, AK, AZ, AR, CA
    8: 1.0, 9: 0.5, 10: 0.3, 11: 0.4, 12: 2.8,  # CO, CT, DE, DC, FL
    13: 1.8, 15: 0.3, 16: 0.4, 17: 1.6, 18: 1.2,  # GA, HI, ID, IL, IN
    19: 0.7, 20: 0.8, 21: 0.9, 22: 0.8, 23: 0.3,  # IA, KS, KY, LA, ME
    24: 0.8, 25: 0.7, 26: 1.3, 27: 0.9, 28: 0.5,  # MD, MA, MI, MN, MS
    29: 0.9, 30: 0.4, 31: 0.5, 32: 0.6, 33: 0.3,  # MO, MT, NE, NV, NH
    34: 0.9, 35: 0.5, 36: 1.7, 37: 1.3, 38: 0.2,  # NJ, NM, NY, NC, ND
    39: 1.4, 40: 0.9, 41: 0.7, 42: 1.1, 44: 0.2,  # OH, OK, OR, PA, RI
    45: 0.8, 46: 0.3, 47: 1.0, 48: 2.5, 49: 0.5,  # SC, SD, TN, TX, UT
    50: 0.2, 51: 1.1, 53: 0.9, 54: 0.4, 55: 0.6,  # VT, VA, WA, WV, WI
    56: 0.3, 72: 0.4  # WY, PR
}


# ============================================================================
# GLOBAL BASELINE DATA (WHO Global Status Report on Road Safety)
# ============================================================================
@st.cache_data
def get_global_baseline():
    """WHO baseline: motorcycle fleet share vs. share of road deaths."""
    return pd.DataFrame({
        'Country': ['Thailand', 'Vietnam', 'Indonesia', 'Puerto Rico', 'United States'],
        'motorcycle_fleet_pct': [80.0, 75.0, 68.0, 12.0, 3.5],
        'fatality_share_pct': [80.0, 72.0, 65.0, 15.0, 8.2],
        'region': ['Southeast Asia', 'Southeast Asia', 'Southeast Asia',
                   'Caribbean', 'North America']
    })


# ============================================================================
# REAL DATA DETECTION (checks for NHTSA FARS CSV files)
# ============================================================================
import os

def has_real_data() -> bool:
    """Check if real NHTSA FARS data files exist locally."""
    data_dir = os.path.join(os.getcwd(), "traffic-accidents")
    acc_exists = os.path.exists(os.path.join(data_dir, "accident.csv"))
    per_exists = os.path.exists(os.path.join(data_dir, "person.csv"))
    veh_exists = os.path.exists(os.path.join(data_dir, "vehicle.csv"))
    return acc_exists and per_exists and veh_exists


@st.cache_data
def load_real_fars_data(year: int, severity_filter: str = 'fatalities') -> pd.DataFrame:
    """Load real NHTSA FARS data from local CSV files.
    
    Motorcycles identified by BODY_TYP codes in vehicle.csv (80, 82, 83, 85, 87, 88),
    not by PER_TYP which is unreliable in FARS.
    """
    data_dir = os.path.join(os.getcwd(), "traffic-accidents")
    acc_df = pd.read_csv(os.path.join(data_dir, "accident.csv"), low_memory=False)
    per_df = pd.read_csv(os.path.join(data_dir, "person.csv"), low_memory=False)
    
    # Standardize column names
    acc_df.columns = [c.upper() for c in acc_df.columns]
    per_df.columns = [c.upper() for c in per_df.columns]
    
    # Filter by year if available
    if 'YEAR' in acc_df.columns:
        acc_df = acc_df[acc_df['YEAR'] == year]
    
    # Merge person and accident data on case ID
    merged = per_df.merge(acc_df[['ST_CASE', 'STATE', 'DAY_WEEK']], 
                          on='ST_CASE', how='inner', suffixes=('', '_acc'))
    
    # CRITICAL: Filter for motorcycles by BODY_TYP from vehicle.csv
    # Motorcycle body types: 80, 82, 83, 85, 87, 88 (5,999 vehicles)
    # Note: PER_TYP==4 only yields ~50 records and is unreliable
    veh_path = os.path.join(data_dir, "vehicle.csv")
    if os.path.exists(veh_path):
        veh_df = pd.read_csv(veh_path, low_memory=False)
        veh_df.columns = [c.upper() for c in veh_df.columns]
        # Filter vehicle data for motorcycles
        motorcycle_body_types = [80, 82, 83, 85, 87, 88]
        veh_df = veh_df[veh_df['BODY_TYP'].isin(motorcycle_body_types)]
        # Merge with person data to get motorcycle riders only
        merged = merged.merge(veh_df[['ST_CASE', 'VEH_NO']], 
                             on=['ST_CASE', 'VEH_NO'], how='inner')
    else:
        # Fallback (less accurate): Use person type
        merged = merged[merged['PER_TYP'] == 4]
    
    # Filter by severity
    if severity_filter == 'fatalities':
        merged = merged[merged['INJ_SEV'] == 4]
    
    # Map state codes to names
    if 'STATE' in merged.columns:
        merged['State'] = merged['STATE'].map(STATE_CODE_TO_NAME)
    
    # Add region indicator
    merged['Region'] = merged['STATE'].apply(
        lambda x: 'Puerto Rico' if x == 72 else 'US Mainland'
    )
    
    # Add time window (weekday vs weekend)
    if 'DAY_WEEK' in merged.columns:
        merged['Time Window'] = merged['DAY_WEEK'].apply(
            lambda x: 'Weekend (Fri–Sun)' if x in [1, 6, 7] else 'Weekday (Mon–Thu)'
        )
    
    return merged.reset_index(drop=True)


# ============================================================================
# DATA INGESTION ENGINE (cached, memory-optimized)
# ============================================================================
@st.cache_data
def load_fars_data_standardized(year: int, severity_filter: str = 'fatalities') -> pd.DataFrame:
    """Standardized: fixed 2000 records, weighted state distribution."""
    np.random.seed(42 + year)
    n_records = 2000
    states = list(STATE_WEIGHTS.keys())
    weights = np.array([STATE_WEIGHTS[s] for s in states])
    weights = weights / weights.sum()
    df = pd.DataFrame({
        'ST_CASE': np.random.randint(100000, 999999, n_records),
        'STATE': np.random.choice(states, n_records, p=weights),
        'DAY_WEEK': np.random.randint(1, 8, n_records),
        'PER_TYP': np.random.choice([1, 2, 3, 4], n_records, p=[0.5, 0.2, 0.2, 0.1]),
        'INJ_SEV': np.random.choice([1, 2, 3, 4], n_records, p=[0.3, 0.3, 0.2, 0.2]),
    })
    df['State'] = df['STATE'].map(STATE_CODE_TO_NAME)
    df['Day'] = df['DAY_WEEK'].map(DAY_NAMES)
    df['Region'] = np.where(df['STATE'] == 72, 'Puerto Rico', 'US Mainland')
    df['Time Window'] = np.where(df['DAY_WEEK'].isin([1, 6, 7]), 'Weekend (Fri–Sun)', 'Weekday (Mon–Thu)')
    df = df[df['PER_TYP'] == 4].copy()
    if severity_filter == 'fatalities':
        df = df[df['INJ_SEV'] == 4]
    return df.reset_index(drop=True)


@st.cache_data
def load_fars_data_random(year: int, severity_filter: str = 'fatalities') -> pd.DataFrame:
    """Non-standardized: variable 1500-3000 records, random state distribution."""
    np.random.seed(42 + year)
    n_records = np.random.randint(1500, 3000)
    df = pd.DataFrame({
        'ST_CASE': np.random.randint(100000, 999999, n_records),
        'STATE': np.random.choice(list(STATE_CODE_TO_NAME.keys()), n_records),
        'DAY_WEEK': np.random.randint(1, 8, n_records),
        'PER_TYP': np.random.choice([1, 2, 3, 4], n_records, p=[0.5, 0.2, 0.2, 0.1]),
        'INJ_SEV': np.random.choice([1, 2, 3, 4], n_records, p=[0.3, 0.3, 0.2, 0.2]),
    })
    df['State'] = df['STATE'].map(STATE_CODE_TO_NAME)
    df['Day'] = df['DAY_WEEK'].map(DAY_NAMES)
    df['Region'] = np.where(df['STATE'] == 72, 'Puerto Rico', 'US Mainland')
    df['Time Window'] = np.where(df['DAY_WEEK'].isin([1, 6, 7]), 'Weekend (Fri–Sun)', 'Weekday (Mon–Thu)')
    df = df[df['PER_TYP'] == 4].copy()
    if severity_filter == 'fatalities':
        df = df[df['INJ_SEV'] == 4]
    return df.reset_index(drop=True)


def load_year_with_gc(year: int, data_type: str = 'standardized', severity_filter: str = 'fatalities') -> tuple:
    """
    Load data (real if available, simulated otherwise).
    Returns (dataframe, data_source_label).
    """
    gc.collect()
    
    # Try real data first
    if has_real_data():
        try:
            df = load_real_fars_data(year, severity_filter=severity_filter)
            return df, "📊 Real NHTSA FARS Data"
        except Exception as e:
            st.warning(f"Could not load real data: {e}. Falling back to simulated.")
    
    # Fall back to simulated
    if data_type == 'standardized':
        df = load_fars_data_standardized(year, severity_filter=severity_filter)
    else:
        df = load_fars_data_random(year, severity_filter=severity_filter)
    
    return df, "🎲 Simulated Data (for demo)"


global_baseline_df = get_global_baseline()

# ============================================================================
# SESSION STATE — track control changes for cache invalidation
# ============================================================================
if 'prev_data_type' not in st.session_state:
    st.session_state.prev_data_type = None
if 'prev_severity' not in st.session_state:
    st.session_state.prev_severity = None

# ============================================================================
# SIDEBAR — controls + guide
# ============================================================================
with st.sidebar:
    st.header("🏍️ Controls")
    
    data_type = st.radio(
        "Data view",
        options=['Standardized', 'Non-standardized'],
        index=0,
        help="Standardized: fixed 2000 records, realistic state weights."
    ).lower()
    
    selected_year = st.select_slider(
        "Data year",
        options=[2021, 2022, 2023, 2024],
        value=2024,
        help="Choose the FARS reporting year."
    )
    
    injury_type = st.radio(
        "Injury severity",
        options=['Fatalities only', 'All accidents'],
        index=0,
        help="Fatalities: fatal injuries only. All accidents: includes non-fatal injuries too."
    )
    severity_filter = 'fatalities' if injury_type == 'Fatalities only' else 'all'

    st.markdown("---")
    st.markdown("### 📖 How to read this")
    st.markdown(
        "- **Where** – fatalities by state\n"
        "- **When** – weekday vs. weekend\n"
        "- **Global context** – how the US compares worldwide\n"
        "- **Explore data** – filter and download records"
    )

# Detect mode/severity changes for cache invalidation
data_mode_changed = data_type != st.session_state.prev_data_type
severity_changed = severity_filter != st.session_state.prev_severity

# Update session state
st.session_state.prev_data_type = data_type
st.session_state.prev_severity = severity_filter

# Load data after controls defined
fars_data, data_source = load_year_with_gc(selected_year, data_type=data_type, severity_filter=severity_filter)

with st.sidebar:
    st.markdown("---")
    st.caption(f"Data mode: **{data_type.title()}**")
    st.caption(f"Injury type: **{injury_type}**")
    st.caption(f"Year loaded: **{selected_year}**")
    st.caption(f"Sample size: **{len(fars_data):,}** records")
    if data_mode_changed or severity_changed:
        st.success("✨ Data refreshed", icon="🔄")
    st.caption(data_source)  # Show data source badge
    st.caption(f"Refreshed: {datetime.now():%Y-%m-%d %H:%M}")

# Precompute common aggregates
total_deaths = len(fars_data)
pr_deaths = int((fars_data['Region'] == 'Puerto Rico').sum())
mainland_deaths = int((fars_data['Region'] == 'US Mainland').sum())
weekend_deaths = int((fars_data['Time Window'] == 'Weekend (Fri–Sun)').sum())
weekend_pct = (weekend_deaths / total_deaths * 100) if total_deaths else 0

state_counts = (
    fars_data.groupby('State').size()
    .reset_index(name='Fatalities')
    .sort_values('Fatalities', ascending=False)
)
top_state = state_counts.iloc[0]['State'] if not state_counts.empty else "—"

# ============================================================================
# HEADER + INTRO
# ============================================================================
st.title("🏍️ Motorcycle Safety Analysis")
st.caption("Made by **Gian-Carlo Javier**")
st.markdown(data_source)  # Display data source badge at top
st.markdown(
    f"Exploring **where** and **when** motorcyclist fatalities happen across the "
    f"United States and Puerto Rico in **{selected_year}**, and how the US fits "
    f"into the **global** road-safety picture."
)

# Clarify that this is filtered motorcycle data from NHTSA
st.info(
    "**Data note:** This analysis filters NHTSA FARS (a comprehensive database of ALL traffic fatalities) "
    "to show **motorcyclists only** (PER_TYP = 4). Totals represent fatal motorcycle crashes, not all road fatalities."
)

# Headline metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total motorcyclist fatalities", f"{total_deaths:,}")
c2.metric("Highest-count state", top_state)
c3.metric("Puerto Rico", f"{pr_deaths:,}")
c4.metric("Share on weekends", f"{weekend_pct:.0f}%")

st.markdown("---")

# ============================================================================
# TABS — guided narrative
# ============================================================================
tab_where, tab_when, tab_global, tab_data = st.tabs(
    ["📍 Where", "🕒 When", "🌍 Global context", "🔎 Explore data"]
)

# ----------------------------------------------------------------------------
# WHERE — by state (names, not codes)
# ----------------------------------------------------------------------------
with tab_where:
    st.subheader(f"Where do motorcyclist fatalities happen? ({selected_year})")

    left, right = st.columns([3, 2])

    with left:
        st.markdown("**Fatalities by state — map view**")
        mainland = state_counts[state_counts['State'] != 'Puerto Rico'].copy()
        mainland['abbr'] = mainland['State'].map(STATE_NAME_TO_ABBR)
        fig_map = px.choropleth(
            mainland,
            locations='abbr',
            locationmode="USA-states",
            color='Fatalities',
            scope="usa",
            color_continuous_scale="Reds",
            hover_name='State',
            labels={'Fatalities': 'Fatalities'}
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption("Puerto Rico is shown separately in the ranking on the right.")

    with right:
        st.markdown("**Top 10 states by fatalities**")
        top10 = state_counts.head(10).sort_values('Fatalities')
        fig_top = px.bar(
            top10,
            x='Fatalities',
            y='State',
            orientation='h',
            color='Fatalities',
            color_continuous_scale='Reds',
            text='Fatalities'
        )
        fig_top.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_showscale=False,
            yaxis_title=None
        )
        st.plotly_chart(fig_top, use_container_width=True)

    st.info(
        f"**Takeaway:** In {selected_year}, **{top_state}** recorded the most "
        f"motorcyclist fatalities. Puerto Rico accounted for **{pr_deaths:,}** "
        f"and the US mainland for **{mainland_deaths:,}**."
    )

# ----------------------------------------------------------------------------
# WHEN — weekday vs weekend + by day
# ----------------------------------------------------------------------------
with tab_when:
    st.subheader(f"When do motorcyclist fatalities happen? ({selected_year})")

    left, right = st.columns(2)

    with left:
        st.markdown("**Weekday vs. weekend**")
        window_counts = fars_data['Time Window'].value_counts().reset_index()
        window_counts.columns = ['Time Window', 'Fatalities']
        fig_window = px.pie(
            window_counts,
            values='Fatalities',
            names='Time Window',
            hole=0.45,
            color='Time Window',
            color_discrete_map={
                'Weekend (Fri–Sun)': '#ef553b',
                'Weekday (Mon–Thu)': '#636efa'
            }
        )
        fig_window.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_window, use_container_width=True)

    with right:
        st.markdown("**By day of the week**")
        day_order = ["Sunday", "Monday", "Tuesday", "Wednesday",
                     "Thursday", "Friday", "Saturday"]
        day_counts = (
            fars_data.groupby('Day').size()
            .reindex(day_order, fill_value=0)
            .reset_index(name='Fatalities')
        )
        fig_days = px.bar(
            day_counts,
            x='Day', y='Fatalities',
            color='Fatalities', color_continuous_scale='Reds',
            text='Fatalities'
        )
        fig_days.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_showscale=False,
            xaxis_title=None
        )
        st.plotly_chart(fig_days, use_container_width=True)

    st.info(
        f"**Takeaway:** About **{weekend_pct:.0f}%** of motorcyclist fatalities "
        f"in {selected_year} occurred on weekends (Fri–Sun) — the window that "
        f"typically sees more recreational riding."
    )

# ----------------------------------------------------------------------------
# GLOBAL CONTEXT — WHO
# ----------------------------------------------------------------------------
with tab_global:
    st.subheader("How does the US compare globally?")
    st.markdown(
        "Countries where motorcycles make up a large share of the vehicle fleet "
        "also see a large share of road deaths among riders. The US and Puerto "
        "Rico sit at the low-exposure end of this spectrum."
    )

    fig_scatter = px.scatter(
        global_baseline_df,
        x='motorcycle_fleet_pct',
        y='fatality_share_pct',
        text='Country',
        size='fatality_share_pct',
        color='region',
        labels={
            'motorcycle_fleet_pct': 'Motorcycles as % of vehicle fleet',
            'fatality_share_pct': 'Motorcyclists as % of road deaths',
            'region': 'Region'
        }
    )
    fig_scatter.update_traces(textposition='top center')
    fig_scatter.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_scatter, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_fleet = px.bar(
            global_baseline_df.sort_values('motorcycle_fleet_pct'),
            x='motorcycle_fleet_pct', y='Country', orientation='h',
            title='Motorcycles as % of vehicle fleet',
            color='motorcycle_fleet_pct', color_continuous_scale='Blues',
            text='motorcycle_fleet_pct'
        )
        fig_fleet.update_layout(coloraxis_showscale=False, yaxis_title=None,
                                margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_fleet, use_container_width=True)
    with col2:
        fig_share = px.bar(
            global_baseline_df.sort_values('fatality_share_pct'),
            x='fatality_share_pct', y='Country', orientation='h',
            title='Motorcyclists as % of road deaths',
            color='fatality_share_pct', color_continuous_scale='Reds',
            text='fatality_share_pct'
        )
        fig_share.update_layout(coloraxis_showscale=False, yaxis_title=None,
                                margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_share, use_container_width=True)

    st.info(
        "**Takeaway:** In Thailand and Vietnam, riders make up **70–80%** of road "
        "deaths. In the US, motorcyclists are roughly **8%** — but they remain a "
        "high-risk group relative to their small share of the fleet."
    )

# ----------------------------------------------------------------------------
# EXPLORE DATA
# ----------------------------------------------------------------------------
with tab_data:
    st.subheader("Explore the underlying records")

    states_available = ["All"] + sorted(fars_data['State'].unique().tolist())
    pick = st.selectbox("Filter by state", states_available)

    view = fars_data if pick == "All" else fars_data[fars_data['State'] == pick]
    st.caption(f"Showing **{len(view):,}** records"
               + ("" if pick == "All" else f" for **{pick}**"))
    st.dataframe(
        view[['State', 'Region', 'Day', 'Time Window', 'ST_CASE']],
        use_container_width=True, hide_index=True
    )

    st.download_button(
        "⬇️ Download this view (CSV)",
        data=view.to_csv(index=False).encode('utf-8'),
        file_name=f"motorcycle_fatalities_{selected_year}"
                  + ("" if pick == "All" else f"_{pick}") + ".csv",
        mime="text/csv"
    )

st.markdown("---")

# ============================================================================
# CREDITS & DATA SOURCES
# ============================================================================
with st.expander("ℹ️ About this project & data sources"):
    st.markdown(
        """
        **Created by Gian-Carlo Javier.**

        **Data sources**
        - **NHTSA FARS** — Fatality Analysis Reporting System (US motorcyclist
          fatalities). U.S. Department of Transportation, National Highway Traffic
          Safety Administration. <https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars>
        - **WHO Global Status Report on Road Safety** — World Health Organization.
          <https://www.who.int/teams/social-determinants-of-health/safety-and-mobility/global-status-report-on-road-safety-2023>
        - **State names** mapped from official FARS / FIPS state codes.

        *This dashboard is for educational purposes. Figures shown here are
        illustrative until the live FARS files are connected.*
        """
    )

st.caption(
    "Made by Gian-Carlo Javier · Data: NHTSA FARS & WHO Global Status Report on "
    "Road Safety · State names from FARS/FIPS codes."
)
