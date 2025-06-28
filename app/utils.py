import pandas as pd
import re

def extract_entities_from_result(result):
    vendors = []
    amounts = []
    current_vendor = []
    current_amount = []

    for page in result:
        for word, label in page:
            if label == "B-VENDOR":
                if current_vendor:
                    vendors.append(" ".join(current_vendor))
                    current_vendor = []
                current_vendor.append(word)
            elif label == "I-VENDOR" and current_vendor:
                current_vendor.append(word)
            elif label == "B-TOTAL":
                if current_amount:
                    amounts.append(" ".join(current_amount))
                    current_amount = []
                current_amount.append(word)
            elif label == "I-TOTAL" and current_amount:
                current_amount.append(word)

    if current_vendor:
        vendors.append(" ".join(current_vendor))
    if current_amount:
        amounts.append(" ".join(current_amount))

    # Clean and convert amounts to numbers
    clean_amounts = []
    for amt in amounts:
        try:
            amt_clean = re.sub(r"[^0-9.]", "", amt)
            clean_amounts.append(float(amt_clean))
        except:
            continue

    return pd.DataFrame({"vendor": vendors, "amount": clean_amounts})

