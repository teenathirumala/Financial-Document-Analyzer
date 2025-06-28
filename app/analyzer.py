import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

def detect_anomalies(df):
    clf = IsolationForest(contamination=0.05)
    df["anomaly"] = clf.fit_predict(df[["amount"]])
    return df

def cluster_vendors(df):
    km = KMeans(n_clusters=3)
    df["cluster"] = km.fit_predict(df[["amount"]])
    return df

