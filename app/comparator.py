
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ks_2samp, entropy

def render_distribution_comparison(real_df, synth_df):
    st.subheader("📈 Compare Column Distributions")
    common_cols = real_df.columns.intersection(synth_df.columns)
    if len(common_cols) == 0:
        st.warning("No common columns found.")
        return

    selected_col = st.selectbox("Select column", sorted(common_cols))

    if selected_col:
        if pd.api.types.is_numeric_dtype(real_df[selected_col]):
            _compare_numerical(real_df, synth_df, selected_col)
        else:
            _compare_categorical(real_df, synth_df, selected_col)

def _compare_numerical(real_df, synth_df, col):
    fig, ax = plt.subplots()
    sns.kdeplot(real_df[col], label="Original", ax=ax, fill=True, alpha=0.5)
    sns.kdeplot(synth_df[col], label="Synthetic", ax=ax, fill=True, alpha=0.5)
    ax.set_title(f"Distribution of '{col}' (Numeric)")
    ax.legend()
    st.pyplot(fig)

    ks_stat = ks_2samp(real_df[col], synth_df[col])
    st.markdown(f"**KS Statistic:** `{ks_stat.statistic:.4f}` (p-value: `{ks_stat.pvalue:.4g}`)")

def _compare_categorical(real_df, synth_df, col):
    fig, ax = plt.subplots()
    real_counts = real_df[col].value_counts(normalize=True)
    synth_counts = synth_df[col].value_counts(normalize=True)
    compare_df = pd.DataFrame({
        "Original": real_counts,
        "Synthetic": synth_counts
    }).fillna(0)

    compare_df.plot(kind='bar', ax=ax, alpha=0.7)
    ax.set_title(f"Distribution of '{col}' (Categorical)")
    st.pyplot(fig)

    kl_div = entropy(compare_df["Original"], compare_df["Synthetic"])
    st.markdown(f"**KL Divergence:** `{kl_div:.4f}`")
