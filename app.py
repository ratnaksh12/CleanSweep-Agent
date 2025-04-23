import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from helpers.cleaning_suggestions import get_cleaning_suggestions
from helpers.ai_overview import get_dataset_overview
from helpers.ai_action_plan import generate_action_plan

# ----- SET PAGE CONFIG -----
st.set_page_config(page_title="CleanSweep Agent - AI Data Cleaning Agent", layout="wide")

# ----- LOAD LOCAL CSS -----
def load_local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_local_css("style.css")

# ----- HEADER -----
st.title("🧼 CleanSweep Agent")
st.caption("Your AI-powered data cleaning assistant. ✨")

# ----- FILE UPLOAD -----
uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])
if uploaded_file:
    st.markdown(f"✅ Uploaded: `{uploaded_file.name}`")

# ----- SESSION STATE INIT -----
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

# ----- TABS -----
tabs = ["📊 Data Health", "🧠 Overview", "🧹 AI Suggestions", "🛠️ Manual Cleaning", "🎯 Action Plan", "🆚 Compare"]
selected_tab = st.radio("Navigate:", tabs, horizontal=True)

# ----- DATA HEALTH FUNCTION -----
def calculate_data_health(df):
    missing_pct = df.isnull().mean() * 100
    duplicate_pct = df.duplicated().mean() * 100

    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        z_scores = np.abs(stats.zscore(numeric_df))
        outlier_count = (z_scores > 3).sum().sum()
    else:
        outlier_count = 0

    health = (max(0, 100 - missing_pct.mean()) +
              max(0, 100 - duplicate_pct) +
              max(0, 100 - (outlier_count / len(df)) * 100 if len(df) > 0 else 100)) / 3

    return missing_pct.mean(), duplicate_pct, outlier_count, health

# ----- MAIN LOGIC -----
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if selected_tab == "📊 Data Health":
        st.subheader("📊 Data Health Score")
        with st.spinner("Analyzing your dataset..."):
            miss, dup, outliers, score = calculate_data_health(df)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🧩 Missing Values", f"{miss:.2f}%")
        col2.metric("🌀 Duplicates", f"{dup:.2f}%")
        col3.metric("🚨 Outliers", outliers)
        col4.metric("🧠 Health Score", f"{score:.2f}/100")

    elif selected_tab == "🧠 Overview":
        st.subheader("🧠 Dataset Summary")
        with st.spinner("Generating smart insights..."):
            summary = get_dataset_overview(df)
        st.markdown(summary)

    elif selected_tab == "🧹 AI Suggestions":
        st.subheader("🤖 Smart Suggestions")
        with st.spinner("Detecting issues..."):
            suggestions = get_cleaning_suggestions(df)
        st.markdown(suggestions)

    elif selected_tab == "🛠️ Manual Cleaning":
        st.subheader("🛠️ Manual Cleaning Panel")
        st.markdown("Apply your own fixes with full control.")

        remove_dupes = st.checkbox("🧽 Remove Duplicates")
        drop_cols = st.multiselect("📤 Drop Columns", df.columns)
        null_strategy = st.selectbox("📉 Handle Nulls", ["None", "Fill with 0", "Fill with mean", "Fill with mode"])
        rename_cols = {col: st.text_input(f"Rename `{col}`", value=col) for col in df.columns}

        if st.button("✅ Apply Fixes"):
            cleaned = df.copy()
            if remove_dupes:
                cleaned = cleaned.drop_duplicates()
            if drop_cols:
                cleaned = cleaned.drop(columns=drop_cols)
            cleaned = cleaned.rename(columns=rename_cols)

            if null_strategy == "Fill with 0":
                cleaned = cleaned.fillna(0)
            elif null_strategy == "Fill with mean":
                cleaned = cleaned.fillna(cleaned.mean(numeric_only=True))
            elif null_strategy == "Fill with mode":
                mode_df = cleaned.mode()
                if not mode_df.empty:
                    cleaned = cleaned.fillna(mode_df.iloc[0])

            st.success("✅ Cleaning Applied!")
            st.session_state.cleaned_df = cleaned  # Save for Compare tab
            st.dataframe(cleaned.head(), use_container_width=True)

            csv = cleaned.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Cleaned CSV", data=csv, file_name="cleaned_data.csv", mime="text/csv")

    elif selected_tab == "🎯 Action Plan":
        st.subheader("🎯 AI Action Plan")
        with st.spinner("Crafting data strategy..."):
            action = generate_action_plan(df)
        st.markdown(action)

    elif selected_tab == "🆚 Compare":
        st.subheader("🆚 Before vs After (Preview Only)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("🔴 **Before Cleaning**")
            st.dataframe(df.head(), use_container_width=True)
        with col2:
            if st.session_state.cleaned_df is not None:
                st.markdown("🟢 **After Cleaning**")
                st.dataframe(st.session_state.cleaned_df.head(), use_container_width=True)
            else:
                st.markdown("⚠️ Apply manual cleaning to see results here.")

else:
    st.info("📁 Upload a CSV file to begin cleaning.")
