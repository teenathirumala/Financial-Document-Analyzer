import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def plot_anomalies(df):
    fig, ax = plt.subplots()
    sns.scatterplot(x="vendor", y="amount", hue="anomaly", data=df, ax=ax)
    st.pyplot(fig)

def plot_clusters(df):
    fig, ax = plt.subplots()
    sns.barplot(x="vendor", y="amount", hue="cluster", data=df, ax=ax)
    st.pyplot(fig)
