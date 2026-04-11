import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="NeoResist-MD", layout="wide")

st.title("🧬 NeoResist-MD")
st.markdown("**245-patient TCGA-SARC neoantigen cohort**")

# Load your data
df = pd.read_csv("data/final_sarc_cohort.csv")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Patients", len(df))
col2.metric("Mutations", f"{df['mutations'].sum():,}")
col3.metric("Candidates", f"{df['candidates'].sum():,}")
col4.metric("Fanout", f"{df['fanout'].mean():.1f}x")

fig1 = px.histogram(df, x="fanout", nbins=20, title="Fanout Distribution")
st.plotly_chart(fig1)

fig2 = px.scatter(df, x="mutations", y="fanout", size="candidates", 
                  title="TMB vs Fanout", hover_name="patient")
st.plotly_chart(fig2)

st.subheader("Top 10 Patients")
st.dataframe(df.nlargest(10, "mutations"))

st.subheader("Upload Your Tumor")
uploaded = st.file_uploader("MAF file", type="maf")
if uploaded:
    st.success("Ready for NeoVax prediction")
    st.info("Coming soon: Real-time prediction")

st.markdown("---")
st.caption("Built by Dr. Rambe • TCGA-SARC 2026")