# app/batch_processor.py

import os
import pandas as pd
from app.extractor import load_model, extract_from_pdf
from app.utils import extract_entities_from_result

def batch_extract(directory):
    processor, model = load_model()
    all_data = []

    for file in os.listdir(directory):
        if file.endswith(".pdf") or file.endswith(".jpg") or file.endswith(".png"):
            path = os.path.join(directory, file)
            result = extract_from_pdf(path, processor, model)
            df = extract_entities_from_result(result)
            if not df.empty:
                df["file"] = file
                all_data.append(df)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv("output/summary.csv", index=False)
        print("✅ Saved summary to output/summary.csv")
        return final_df
    else:
        print("❌ No valid data found.")
        return pd.DataFrame()
