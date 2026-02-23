"""
app.py - Streamlit application for A/B Testing MLOps project.
Run with: streamlit run app.py
"""

import io
import warnings
import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="A/B Test MLOps Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .sig-badge {
        background: #10b981;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
    }
    .not-sig-badge {
        background: #ef4444;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
    }
    .insight-box {
        background: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
@st.cache_data
def load_data(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


@st.cache_resource
def load_model(model_path: str):
    try:
        return joblib.load(model_path)
    except Exception:
        return None


def run_z_test(ctrl: pd.Series, treat: pd.Series, alpha: float) -> dict:
    n_c, n_t = len(ctrl), len(treat)
    p_c, p_t = ctrl.mean(), treat.mean()
    p_pool = (ctrl.sum() + treat.sum()) / (n_c + n_t)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t)) if p_pool > 0 else 1e-9
    z = (p_t - p_c) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    lift = (p_t - p_c) / p_c * 100 if p_c > 0 else 0
    ci_lower = (p_t - p_c) - 1.96 * se
    ci_upper = (p_t - p_c) + 1.96 * se
    return {
        "control_rate": p_c,
        "treatment_rate": p_t,
        "lift_pct": lift,
        "z_statistic": z,
        "p_value": p_value,
        "significant": p_value < alpha,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/experiment.png", width=80)
st.sidebar.title("⚙️ Controls")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 A/B Testing", "🤖 ML Predictions", "📈 Data Explorer", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

alpha = st.sidebar.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)


# ─────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown('<p class="main-header">📊 A/B Test MLOps Dashboard</p>', unsafe_allow_html=True)
    st.markdown("**Statistical inference for the advert industry — powered by ML & classical testing.**")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Supported Tests", "4", "Z, Chi², T, MWU")
    with col2:
        st.metric("ML Models", "4", "LR, DT, RF, XGB")
    with col3:
        st.metric("Framework", "MLflow", "Experiment Tracking")
    with col4:
        st.metric("Deployment", "Docker", "Containerized")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 🔬 What This Dashboard Does")
        st.markdown("""
        - **A/B Testing**: Run statistical hypothesis tests (Z-test, Chi-square, t-test, Mann-Whitney U)  
        - **Sample Size Calculator**: Determine required sample size for your experiment  
        - **ML Predictions**: Use trained models to predict conversion probability  
        - **Data Explorer**: Visualize distributions, correlations, and group differences  
        """)
    with col_b:
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        1. Upload your CSV file in the sidebar  
        2. Navigate to **A/B Testing** to run statistical tests  
        3. Navigate to **ML Predictions** to score new data  
        4. Use **Data Explorer** to visualize your dataset  
        """)
        st.info("📁 Sample data format: rows = users, columns = features + `converted` target")


# ─────────────────────────────────────────────
# PAGE: A/B TESTING
# ─────────────────────────────────────────────
elif page == "📊 A/B Testing":
    st.title("📊 A/B Statistical Testing")

    if uploaded_file is None:
        st.warning("Please upload a CSV file in the sidebar to begin.")
        st.stop()

    df = load_data(uploaded_file)
    st.success(f"✅ Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Tab layout
    tab1, tab2, tab3 = st.tabs(["🔬 Run Tests", "📐 Sample Size", "📊 Visualizations"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            group_col = st.selectbox("Group Column", df.columns)
        with col2:
            unique_vals = df[group_col].dropna().unique()
            ctrl_val = st.selectbox("Control Group Value", unique_vals)
        with col3:
            treat_options = [v for v in unique_vals if v != ctrl_val]
            treat_val = st.selectbox("Treatment Group Value", treat_options if treat_options else unique_vals)

        metric_col = st.selectbox("Outcome / Metric Column", df.columns)

        if st.button("🚀 Run A/B Tests", type="primary"):
            ctrl_data = df[df[group_col] == ctrl_val][metric_col].dropna()
            treat_data = df[df[group_col] == treat_val][metric_col].dropna()

            if len(ctrl_data) == 0 or len(treat_data) == 0:
                st.error("No data found for one or both groups. Check your column selections.")
                st.stop()

            # Summary Stats
            st.subheader("📋 Group Summary")
            summary_df = pd.DataFrame({
                "Metric": ["N", "Mean", "Std", "Min", "Median", "Max"],
                "Control": [len(ctrl_data), ctrl_data.mean(), ctrl_data.std(),
                            ctrl_data.min(), ctrl_data.median(), ctrl_data.max()],
                "Treatment": [len(treat_data), treat_data.mean(), treat_data.std(),
                              treat_data.min(), treat_data.median(), treat_data.max()],
            })
            st.dataframe(summary_df.style.format({"Control": "{:.4f}", "Treatment": "{:.4f}"}, subset=pd.IndexSlice[1:, :]),
                         use_container_width=True)

            st.markdown("---")
            is_binary = set(ctrl_data.unique()).issubset({0, 1, 0.0, 1.0})

            c1, c2 = st.columns(2)
            with c1:
                if is_binary:
                    st.subheader("Z-Test (Proportions)")
                    z_res = run_z_test(ctrl_data, treat_data, alpha)
                    st.metric("Control Rate", f"{z_res['control_rate']:.2%}")
                    st.metric("Treatment Rate", f"{z_res['treatment_rate']:.2%}")
                    st.metric("Lift", f"{z_res['lift_pct']:+.2f}%")
                    st.metric("p-value", f"{z_res['p_value']:.4f}")
                    badge = "✅ SIGNIFICANT" if z_res["significant"] else "❌ NOT SIGNIFICANT"
                    st.markdown(f"**Result:** {badge}")
                    st.caption(f"95% CI for difference: [{z_res['ci_lower']:.4f}, {z_res['ci_upper']:.4f}]")

            with c2:
                st.subheader("Mann-Whitney U Test")
                u_stat, p_mwu = stats.mannwhitneyu(ctrl_data, treat_data, alternative="two-sided")
                st.metric("U Statistic", f"{u_stat:,.0f}")
                st.metric("p-value", f"{p_mwu:.4f}")
                badge_mwu = "✅ SIGNIFICANT" if p_mwu < alpha else "❌ NOT SIGNIFICANT"
                st.markdown(f"**Result:** {badge_mwu}")

                st.subheader("Welch's T-Test")
                t_stat, p_t = stats.ttest_ind(ctrl_data, treat_data, equal_var=False)
                st.metric("t-statistic", f"{t_stat:.4f}")
                st.metric("p-value", f"{p_t:.4f}")
                badge_t = "✅ SIGNIFICANT" if p_t < alpha else "❌ NOT SIGNIFICANT"
                st.markdown(f"**Result:** {badge_t}")

    with tab2:
        st.subheader("📐 Sample Size Calculator")
        col1, col2, col3 = st.columns(3)
        with col1:
            base_rate = st.number_input("Baseline Conversion Rate", 0.01, 0.99, 0.10, 0.01)
        with col2:
            mde = st.number_input("Minimum Detectable Effect (relative %)", 1.0, 50.0, 10.0, 1.0) / 100
        with col3:
            power = st.slider("Statistical Power", 0.70, 0.99, 0.80, 0.01)

        if st.button("Calculate Sample Size"):
            p1 = base_rate
            p2 = base_rate * (1 + mde)
            z_a = stats.norm.ppf(1 - alpha / 2)
            z_b = stats.norm.ppf(power)
            p_pool = (p1 + p2) / 2
            n = (2 * p_pool * (1 - p_pool) * (z_a + z_b) ** 2) / (p1 - p2) ** 2
            n = int(np.ceil(n))
            st.metric("Required Sample Size (per group)", f"{n:,}")
            st.info(f"Total users needed: **{n*2:,}** | Control: {n:,} | Treatment: {n:,}")

    with tab3:
        if uploaded_file:
            df_vis = load_data(uploaded_file)
            st.subheader("Distribution Comparison")
            metric_col_vis = st.selectbox("Column to visualize", df_vis.select_dtypes(include=np.number).columns)
            group_col_vis = st.selectbox("Group by", df_vis.columns, key="vis_group")

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(data=df_vis, x=metric_col_vis, hue=group_col_vis, ax=axes[0], bins=30, kde=True)
            axes[0].set_title("Distribution")
            sns.boxplot(data=df_vis, x=group_col_vis, y=metric_col_vis, ax=axes[1])
            axes[1].set_title("Box Plot")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ─────────────────────────────────────────────
# PAGE: ML PREDICTIONS
# ─────────────────────────────────────────────
elif page == "🤖 ML Predictions":
    st.title("🤖 ML-Powered Conversion Prediction")

    MODEL_DIR = Path("models")
    model_files = list(MODEL_DIR.glob("*.pkl")) if MODEL_DIR.exists() else []

    if not model_files:
        st.warning("No trained models found. Please train models first by running `python scripts/train.py`")
        st.code("python scripts/train.py --data data/AdSmartABdata.csv")
        st.stop()

    selected_model_file = st.selectbox("Select Model", [f.name for f in model_files])
    model = load_model(str(MODEL_DIR / selected_model_file))

    if model is None:
        st.error("Could not load the selected model.")
        st.stop()

    st.success(f"✅ Model loaded: `{selected_model_file}`")

    st.subheader("Predict on Uploaded Data")
    if uploaded_file:
        df_pred = load_data(uploaded_file)
        st.dataframe(df_pred.head(), use_container_width=True)

        drop_cols = ["converted", "auction_id", "date", "time"]
        X_new = df_pred.drop(columns=[c for c in drop_cols if c in df_pred.columns])

        if st.button("🔮 Generate Predictions", type="primary"):
            try:
                preds = model.predict(X_new)
                probs = model.predict_proba(X_new)[:, 1] if hasattr(model, "predict_proba") else preds

                result_df = df_pred.copy()
                result_df["predicted_conversion"] = preds
                result_df["conversion_probability"] = probs.round(4)

                st.subheader("Prediction Results")
                st.dataframe(result_df[["predicted_conversion", "conversion_probability"]].head(50),
                             use_container_width=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Predictions", f"{len(preds):,}")
                with col2:
                    st.metric("Predicted Conversions", f"{int(preds.sum()):,}")
                with col3:
                    st.metric("Avg. Conv. Probability", f"{probs.mean():.2%}")

                # Probability distribution
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.hist(probs, bins=40, color="#6366f1", edgecolor="white", alpha=0.8)
                ax.set_xlabel("Conversion Probability")
                ax.set_ylabel("Count")
                ax.set_title("Distribution of Predicted Conversion Probabilities")
                st.pyplot(fig)
                plt.close()

                # Download
                csv_bytes = result_df.to_csv(index=False).encode()
                st.download_button(
                    "⬇️ Download Predictions CSV",
                    data=csv_bytes,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.info("Upload a CSV file in the sidebar to generate predictions.")


# ─────────────────────────────────────────────
# PAGE: DATA EXPLORER
# ─────────────────────────────────────────────
elif page == "📈 Data Explorer":
    st.title("📈 Data Explorer")

    if uploaded_file is None:
        st.warning("Please upload a CSV file in the sidebar.")
        st.stop()

    df = load_data(uploaded_file)
    st.success(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

    tab1, tab2, tab3 = st.tabs(["📋 Overview", "📊 Distributions", "🔥 Correlations"])

    with tab1:
        st.subheader("Data Preview")
        st.dataframe(df.head(20), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Types")
            st.dataframe(df.dtypes.rename("dtype").reset_index(), use_container_width=True)
        with col2:
            st.subheader("Missing Values")
            missing = df.isnull().sum().rename("missing").reset_index()
            missing["pct"] = (missing["missing"] / len(df) * 100).round(2)
            st.dataframe(missing, use_container_width=True)

        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe().T, use_container_width=True)

    with tab2:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        col = st.selectbox("Column", num_cols)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        sns.histplot(df[col].dropna(), ax=axes[0], kde=True, color="#6366f1")
        axes[0].set_title("Histogram + KDE")
        stats.probplot(df[col].dropna(), plot=axes[1])
        axes[1].set_title("Q-Q Plot")
        sns.boxplot(y=df[col].dropna(), ax=axes[2], color="#a78bfa")
        axes[2].set_title("Box Plot")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        num_df = df.select_dtypes(include=np.number)
        if num_df.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
            ax.set_title("Correlation Matrix")
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Need at least 2 numeric columns for correlation analysis.")


# ─────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ## A/B Test MLOps Dashboard

    This project demonstrates how to combine **classical A/B testing** with **machine learning** 
    in a production-grade MLOps setup.

    ### Tech Stack
    | Component | Tool |
    |-----------|------|
    | ML Models | scikit-learn, XGBoost |
    | Experiment Tracking | MLflow |
    | Data Versioning | DVC |
    | Dashboard | Streamlit |
    | Containerization | Docker |
    | Statistical Tests | SciPy |

    ### Models Included
    - Logistic Regression (baseline)
    - Decision Tree
    - Random Forest
    - XGBoost

    ### Statistical Tests
    - Two-Proportion Z-Test
    - Chi-Square Test of Independence
    - Welch's T-Test
    - Mann-Whitney U Test (non-parametric)
    - Sample Size Calculator

    ---
    Built with ❤️ | MLOps Portfolio Project
    """)