import streamlit as st
import pandas as pd
import plotly.express as px

# Load the data
data = pd.DataFrame({
    "Dataset": ["Unofficial GuoFeng 2k", "Unofficial GuoFeng 2k", "Unofficial GuoFeng 2k", "Unofficial GuoFeng 2k", "Unofficial GuoFeng 2k", "Unofficial GuoFeng 2k", "WMT23 GuoFeng", "WMT23 GuoFeng", "WMT23 GuoFeng", "WMT23 GuoFeng", "Red Sorghum", "Red Sorghum"],
    "Model": ["GPT-3.5", "Mistral-7B-Base", "Mistral-7B-LoRA", "Mistral-7B-Base", "Mistral-7B-LoRA", "Mistral-7B-LoRA-60k", "GPT-3.5", "Mistral-7B-Base", "Mistral-7B-LoRA", "Mistral-7B-LoRA-60k", "Mistral-7B-Base", "Mistral-7B-LoRA"],
    "Language Pair": ["EN-ZH", "EN-ZH", "EN-ZH", "EN-IT", "EN-IT", "EN-IT", "EN-ZH", "EN-ZH", "EN-ZH", "EN-ZH", "EN-ZH", "EN-ZH"],
    "BLEU": [18.6, 10.3, 48.1, 7.69, 3.6, 13.99, 17.4, 4.1, 35.8, 57.38, 6.3, 7.8],
    "ChrF": [20.0, 13.9, 20.4, 34.75, 18.4, 41.86, 22.3, 12.6, 17.9, 18.18, 8.2, 10.3],
    "COMET": [80.8, 73.7, 80.8, 43.4, 75.58, None, 82.9, 70.7, 77.9, 77.92, 58.0, 67.7]
})

# Set page configuration
st.set_page_config(page_title="Mistral-7B Finetuning Results", layout="wide")

# Add title and description
st.markdown("### Mistral-7B Finetuning Results")

intro_text = """
This interactive dashboard visualizes the performance of the Mistral-7B model after finetuning on three datasets and two language pairs (EN-IT and EN-ZH).

- **Mistral-7B-LoRA**: Finetuned on 20k bilingual sentences per language pair.
- **Mistral-7B-LoRA-60k**: Finetuned on 28k bilingual sentences per language pair, with the EN-IT data having a 60:40 ratio of backtranslated sentences.

The performance is evaluated using three metrics: BLEU, ChrF, and COMET.
"""

st.markdown(intro_text)

# Dataset selection
datasets = data["Dataset"].unique()
selected_dataset = st.sidebar.selectbox("Select Dataset", datasets, key="dataset_select")

# Filter data based on selected dataset
filtered_data = data[data["Dataset"] == selected_dataset]

# Add language pair toggle
language_pairs = filtered_data["Language Pair"].unique()
selected_language_pair = st.sidebar.radio("Select Language Pair", language_pairs, key="lang_pair_select")
filtered_data = filtered_data[filtered_data["Language Pair"] == selected_language_pair]

# Create bar charts for each metric
bleu_chart = px.bar(filtered_data, x="Model", y="BLEU", title="BLEU Scores", barmode="group", text_auto=True)
chrf_chart = px.bar(filtered_data, x="Model", y="ChrF", title="ChrF Scores", barmode="group", text_auto=True)
comet_chart = px.bar(filtered_data, x="Model", y="COMET", title="COMET Scores", barmode="group", text_auto=True)

# Update chart layout
bleu_chart.update_layout(
    xaxis_title="Model",
    yaxis_title="BLEU Score",
    font=dict(family="Arial", size=12),
    title_font_size=16,
    margin=dict(l=50, r=50, t=50, b=50)
)

chrf_chart.update_layout(
    xaxis_title="Model",
    yaxis_title="ChrF Score",
    font=dict(family="Arial", size=12),
    title_font_size=16,
    margin=dict(l=50, r=50, t=50, b=50)
)

comet_chart.update_layout(
    xaxis_title="Model",
    yaxis_title="COMET Score",
    font=dict(family="Arial", size=12),
    title_font_size=16,
    margin=dict(l=50, r=50, t=50, b=50)
)

# Display the charts
col1, col2, col3 = st.columns(3)
with col1:
    st.plotly_chart(bleu_chart, use_container_width=True)
with col2:
    st.plotly_chart(chrf_chart, use_container_width=True)
with col3:
    if comet_chart.data:
        st.plotly_chart(comet_chart, use_container_width=True)
    else:
        st.write("No COMET data available for this selection.")

# Insights and analysis
with st.expander("**Key Insights:**", expanded=True):
    st.markdown("""
    
    - The Mistral-7B-LoRA and Mistral-7B-LoRA-60k models generally outperform the base Mistral-7B-Base model across all metrics and datasets.
    - The performance improvement is particularly significant for the EN-ZH language pair, where the Mistral-7B-LoRA and Mistral-7B-LoRA-60k models achieve much higher BLEU, ChrF, and COMET scores compared to the other models.
    
    - For the EN-IT language pair, the Mistral-7B-LoRA-60k model with backtranslation shows a noticeable improvement over the Mistral-7B-LoRA model without backtranslation.
    - The 'Red Sorghum' dataset (an excerpt by Mo Yan's novel) shows lower scores compared to the other datasets, indicating potential challenges in handling specialized creative content and dialectal language.
    
    - ChrF and BLEU improving shows that the model generates less hallucinated sentences.
    """)