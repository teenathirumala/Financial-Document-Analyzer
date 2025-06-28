import streamlit as st
import pandas as pd
from extractor import load_model, extract_from_pdf
from analyzer import detect_anomalies, cluster_vendors
from visualizer import plot_anomalies, plot_clusters
from utils import extract_entities_from_result

st.title("🧾 Financial Document Analyzer")

uploaded_file = st.file_uploader("Upload Invoice PDF", type=["pdf"])

if uploaded_file:
    st.success("Processing...")
    processor, model = load_model()
    result = extract_from_pdf(uploaded_file.name, processor, model)

    df = extract_entities_from_result(result)
    st.subheader("📈 Summary Stats")
    st.markdown(f"**Total Invoices Processed:** {df.shape[0]}")
    st.markdown(f"**Total Spend (₹):** ₹{df['amount'].sum():,.2f}")
    st.markdown(f"**Top Vendors:**")
    st.dataframe(df['vendor'].value_counts().head(5))

    if not df.empty:
        df = detect_anomalies(df)
        df = cluster_vendors(df)

        st.subheader("📊 Extracted Data")
        st.dataframe(df)

        st.subheader("🔎 Anomaly Detection")
        plot_anomalies(df)

        st.subheader("🌀 Vendor Clusters")
        plot_clusters(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📤 Download CSV", csv, "invoices_summary.csv", "text/csv")

    else:
        st.warning("Could not extract valid vendor or amount fields.")

