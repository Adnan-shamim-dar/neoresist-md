import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="NeoResist-MD", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

:root {
    --bg: #0e1117;
    --surface: #1a1d23;
    --surface-soft: #151922;
    --accent: #01696f;
    --text: #e8e8e8;
    --muted: #8b8f99;
    --border: #2a2f3a;
}

html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
    background: var(--bg);
    color: var(--text);
}

[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stSidebar"],
[data-testid="stSidebarContent"],
[data-testid="stToolbar"] {
    background: var(--bg) !important;
}

#MainMenu, footer, [data-testid="stDeployButton"] {
    visibility: hidden;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1.2rem;
    max-width: 1500px;
}

.neo-brand {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin: 0.2rem 0 1rem 0;
    color: var(--text);
    font-weight: 800;
    font-size: 1.02rem;
}
.neo-dot {
    width: 10px;
    height: 10px;
    border-radius: 999px;
    background: var(--accent);
    box-shadow: 0 0 16px rgba(1, 105, 111, 0.75);
}

.neo-title {
    font-size: 1.7rem;
    font-weight: 800;
    margin-bottom: 0.15rem;
}
.neo-subtitle {
    color: var(--muted);
    font-size: 0.9rem;
    margin-bottom: 0.9rem;
}

.neo-kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.85rem;
    margin: 0.2rem 0 1rem 0;
}
.neo-kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid transparent;
    border-radius: 12px;
    padding: 0.9rem 1rem 0.8rem 1rem;
}
.neo-kpi-card.active {
    border-left-color: var(--accent);
}
.neo-kpi-value {
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-size: clamp(1.28rem, 2.8vw, 2rem);
    font-weight: 700;
    color: var(--text);
    line-height: 1.1;
}
.neo-kpi-label {
    color: var(--muted);
    font-size: 0.73rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

.neo-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.85rem;
    margin-bottom: 0.9rem;
}
.neo-patient-id {
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-size: 1.35rem;
    color: var(--text);
    margin-bottom: 0.6rem;
}
.neo-mini-kpis {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.7rem;
    margin-bottom: 0.8rem;
}
.neo-mini-card {
    background: var(--surface-soft);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 10px;
    padding: 0.6rem 0.75rem;
}
.neo-mini-val {
    color: var(--text);
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-weight: 700;
    font-size: 1.06rem;
}
.neo-mini-label {
    color: var(--muted);
    font-size: 0.7rem;
    text-transform: uppercase;
    margin-top: 0.16rem;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden;
}
[data-testid="stDataFrame"] table {
    color: var(--text) !important;
    background: var(--surface) !important;
}
[data-testid="stDataFrame"] tr:nth-child(even) td {
    background: rgba(255, 255, 255, 0.02) !important;
}
[data-testid="stDataFrame"] tr:hover td {
    background: rgba(1, 105, 111, 0.18) !important;
}

.neo-footer {
    color: var(--muted);
    font-size: 0.74rem;
    margin-top: 0.8rem;
    padding-top: 0.65rem;
    border-top: 1px solid var(--border);
}

@media (max-width: 980px) {
    .neo-kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .neo-mini-kpis { grid-template-columns: repeat(1, minmax(0, 1fr)); }
}
</style>
""",
    unsafe_allow_html=True,
)

# Preserve data loading logic
BASE_DIR = Path(__file__).resolve().parent
data_candidates = [
    BASE_DIR / "data" / "final_sarc_cohort.csv",
    BASE_DIR / "neoresist-md" / "data" / "final_sarc_cohort.csv",
]
data_path = next((p for p in data_candidates if p.exists()), None)
if data_path is None:
    st.error("Could not find final_sarc_cohort.csv in expected data folders.")
    st.stop()
df = pd.read_csv(data_path)

# Preserve column references
patient_col = "patient"
mut_col = "mutations"
cand_col = "candidates"
fanout_col = "fanout"

if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = None
if "selected_indices" not in st.session_state:
    st.session_state.selected_indices = []
if "tmb_range" not in st.session_state:
    st.session_state.tmb_range = (0, int(df[mut_col].max()))
if "fanout_range" not in st.session_state:
    st.session_state.fanout_range = (0.0, float(df[fanout_col].max()))
if "cand_range" not in st.session_state:
    st.session_state.cand_range = (0, int(df[cand_col].max()))
if "search" not in st.session_state:
    st.session_state.search = ""
if "sort_by" not in st.session_state:
    st.session_state.sort_by = "Mutations ↓"
if "nav" not in st.session_state:
    st.session_state.nav = "Overview"
if "hla_selector" not in st.session_state:
    st.session_state.hla_selector = "HLA-A*02:01"
if "nav_inline" not in st.session_state:
    st.session_state.nav_inline = st.session_state.nav
if "hla_inline" not in st.session_state:
    st.session_state.hla_inline = st.session_state.hla_selector

with st.sidebar:
    st.markdown('<div class="neo-brand"><span class="neo-dot"></span>NeoResist-MD</div>', unsafe_allow_html=True)
    st.radio("Navigation", ["Overview", "Patient Explorer", "Pipeline Status", "About"], key="nav")
    st.selectbox("HLA Selector", ["HLA-A*02:01"], key="hla_selector")

    max_mutations = int(df[mut_col].max())
    max_fanout = float(df[fanout_col].max())
    max_candidates = int(df[cand_col].max())

    st.slider("TMB Range", 0, max_mutations, st.session_state.tmb_range, key="tmb_range")
    st.slider("Fanout Range", 0.0, max_fanout, st.session_state.fanout_range, key="fanout_range")
    st.slider("Candidate Range", 0, max_candidates, st.session_state.cand_range, key="cand_range")
    st.text_input("Search patient ID", key="search")
    st.selectbox(
        "Sort by",
        ["Mutations ↓", "Candidates ↓", "Fanout ↓", "Mutations ↑", "Candidates ↑", "Fanout ↑"],
        key="sort_by",
    )

    if st.button("Reset all filters", use_container_width=True):
        st.session_state.tmb_range = (0, max_mutations)
        st.session_state.fanout_range = (0.0, max_fanout)
        st.session_state.cand_range = (0, max_candidates)
        st.session_state.search = ""
        st.session_state.sort_by = "Mutations ↓"
        st.session_state.selected_patient = None
        st.session_state.selected_indices = []
        st.rerun()

with st.expander("Dashboard controls", expanded=False):
    st.radio("Navigation", ["Overview", "Patient Explorer", "Pipeline Status", "About"], key="nav_inline")
    if st.session_state.nav_inline != st.session_state.nav:
        st.session_state.nav = st.session_state.nav_inline
    st.selectbox("HLA Selector", ["HLA-A*02:01"], key="hla_inline")
    if st.session_state.hla_inline != st.session_state.hla_selector:
        st.session_state.hla_selector = st.session_state.hla_inline

# Unified filter computation
filtered_df = df.copy()
filtered_df = filtered_df[
    (filtered_df[mut_col] >= st.session_state.tmb_range[0])
    & (filtered_df[mut_col] <= st.session_state.tmb_range[1])
    & (filtered_df[fanout_col] >= st.session_state.fanout_range[0])
    & (filtered_df[fanout_col] <= st.session_state.fanout_range[1])
    & (filtered_df[cand_col] >= st.session_state.cand_range[0])
    & (filtered_df[cand_col] <= st.session_state.cand_range[1])
]
if st.session_state.search:
    filtered_df = filtered_df[
        filtered_df.apply(
            lambda r: st.session_state.search.lower() in str(r.values).lower(),
            axis=1,
        )
    ]

sort_map = {
    "Mutations ↓": (mut_col, False),
    "Mutations ↑": (mut_col, True),
    "Candidates ↓": (cand_col, False),
    "Candidates ↑": (cand_col, True),
    "Fanout ↓": (fanout_col, False),
    "Fanout ↑": (fanout_col, True),
}
sort_col, sort_asc = sort_map[st.session_state.sort_by]
filtered_df = filtered_df.sort_values(sort_col, ascending=sort_asc)

active_df = filtered_df.copy()
# Original plot mode: no scatter selection filtering.
st.session_state.selected_indices = []
st.session_state.selected_patient = None

patients = len(active_df)
total_mutations = int(active_df[mut_col].sum()) if not active_df.empty else 0
total_candidates = int(active_df[cand_col].sum()) if not active_df.empty else 0
mean_fanout = float(active_df[fanout_col].mean()) if not active_df.empty else 0.0

st.markdown('<div class="neo-title">NeoResist-MD</div>', unsafe_allow_html=True)
st.markdown('<div class="neo-subtitle">245-patient TCGA-SARC neoantigen cohort</div>', unsafe_allow_html=True)

if st.session_state.nav in {"Overview", "Patient Explorer"}:
    st.markdown(
        f"""
<div class="neo-kpi-grid">
  <div class="neo-kpi-card active"><div class="neo-kpi-value">{patients:,}</div><div class="neo-kpi-label">Patients</div></div>
  <div class="neo-kpi-card"><div class="neo-kpi-value">{total_mutations:,}</div><div class="neo-kpi-label">Mutations</div></div>
  <div class="neo-kpi-card"><div class="neo-kpi-value">{total_candidates:,}</div><div class="neo-kpi-label">Candidates</div></div>
  <div class="neo-kpi-card"><div class="neo-kpi-value">{mean_fanout:.1f}×</div><div class="neo-kpi-label">Mean Fanout</div></div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="neo-panel">', unsafe_allow_html=True)
    fig_scatter = px.scatter(
        filtered_df,
        x=mut_col,
        y=fanout_col,
        size=cand_col,
        title="TMB vs Fanout",
        hover_name=patient_col,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Patient Cohort")
    table_df = active_df.copy()
    table_df[patient_col] = table_df[patient_col].astype(str).map(lambda v: f"`{v}`")
    st.dataframe(table_df, use_container_width=True, height=360)

    st.subheader("Upload Your Tumor")
    uploaded = st.file_uploader("MAF file", type="maf")
    if uploaded:
        st.success("Ready for NeoVax prediction")
        st.info("Coming soon: Real-time prediction")

if st.session_state.nav == "Pipeline Status":
    st.markdown('<div class="neo-panel">', unsafe_allow_html=True)
    st.metric("Rows in scope", f"{len(active_df):,}")
    st.metric("Data source", str(data_path))
    st.metric("Current HLA", st.session_state.hla_selector)
    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.nav == "About":
    st.markdown('<div class="neo-panel">', unsafe_allow_html=True)
    st.markdown("NeoResist-MD explores cohort-scale neoantigen signals with interactive patient-level analytics.")
    st.markdown(f"**Current HLA preset:** `{st.session_state.hla_selector}`")
    st.markdown(f"**Dataset:** `{data_path}`")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    f'<div class="neo-footer">{len(df):,} TCGA-SARC patients · {int(df[cand_col].sum()):,} neoantigen candidates · '
    f'{float(df[fanout_col].mean()):.1f}× mean fanout · {st.session_state.hla_selector} · Built by one person.</div>',
    unsafe_allow_html=True,
)
