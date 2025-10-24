#remove unknowns from the analysis
# Effect size + 95% confidence intervals + t-tests for subpopulations
# I have 2 inputs: 1. INPUT_Per_Tweets 2.per user INPUT_Grouped_By_USER each time I am commenting 1
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.weightstats import DescrStatsW, CompareMeans
import statsmodels.api as sm
import os



## This script can use two types of input data:1. INPUT_Per_Tweets – data at the tweet level 2. INPUT_Grouped_By_USER – data aggregated per user.
#  Only one input should be active at a time; comment out the other as needed.
#INPUT_Grouped_By_USER = "data/grouped2_data.pkl"    
INPUT_Per_Tweets      = "data/normalized_data.pkl"


# -------- Load --------
#INPUT_Per_Tweets or INPUT_Grouped_By_USER 
df = pd.read_pickle(INPUT_Per_Tweets)

# -------- Helpers --------
def clean_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

def ordered_counts(series, order):
    vc = series.value_counts(dropna=False)
    extras = [x for x in vc.index if x not in order]
    return vc.reindex(order + extras).fillna(0).astype(int)

def map_gender(x):
    if pd.isna(x) or str(x).strip() == "":
        return "Null"
    s = str(x).strip().lower()
    if s in {"male", "m"}:      return "Male"
    if s in {"female", "f"}:    return "Female"
    if s == "unknown":          return "Unknown"
    return "Other"

def map_verified(x):
    s = clean_str(x)
    if s in {"true","t","yes","verified"}:     return "True"
    if s in {"false","f","no","unverified"}:   return "False"
    if s in {"","unknown","nan","none"}:       return "Unknown"
    return "Other"

def map_acct(x):
    s = clean_str(x)
    if s in {"individual","person","personal"}:                                  return "Individual"
    if s in {"organisational","organizational","organisation","organization","org"}:  return "Organizational"
    if s in {"","unknown","nan","none"}:                                         return "Unknown"
    return "Others"

# Canonical orders
gender_order   = ["Male", "Female", "Unknown", "Null", "Other"]
verified_order = ["True", "False", "Other", "Unknown"]
acct_order     = ["Individual", "Organizational", "Others", "Unknown"]


    # Normalize columns
df["Gender"]  = df.get("Gender", pd.Series(index=df.index, dtype=object)).apply(map_gender)
df["Verified"] = df.get("Verified", pd.Series(index=df.index, dtype=object)).apply(map_verified)
df["Account_Type"] = df.get("Account_Type", pd.Series(index=df.index, dtype=object)).apply(map_acct)



# --- EFFECT SIZE + CONFIDENCE INTERVALS + T-TESTS  for all subpopulations ---
def effect_size_and_ci(group1, group2, label):
    g1 = df[df[label] == group1]['mean_sentiment'].dropna()
    g2 = df[df[label] == group2]['mean_sentiment'].dropna()

    stats1 = DescrStatsW(g1)
    stats2 = DescrStatsW(g2)
    cm = CompareMeans(stats1, stats2)

    mean_diff = np.mean(g1) - np.mean(g2)  #  calculation
    ci_low, ci_high = cm.tconfint_diff()
    t_stat, p_val = ttest_ind(g1, g2, equal_var=False)  # Welch’s t-test (a version of the independent t-test that does not assume equal variances between the groups).

    return {
        'Group 1': group1,
        'Group 2': group2,
        'Label': label,
        'Mean 1': np.mean(g1),
        'Mean 2': np.mean(g2),
        'Mean Difference': mean_diff,
        '95% CI Lower': ci_low,
        '95% CI Upper': ci_high,
        'P-Value': p_val,
        'N1': len(g1),
        'N2': len(g2)
    }

comparisons = [
    ('Male', 'Female', 'Gender'),
    ('True', 'False', 'Verified'),
    ('Organizational', 'Individual', 'Account_Type')
   #  ('Northeast', 'Southwest', 'Region'),
   # ('Midwest', 'West', 'Region'),
    #('Southeast', 'West', 'Region'),
]

results = [effect_size_and_ci(g1, g2, label) for g1, g2, label in comparisons]
comparison_df = pd.DataFrame(results)
comparison_df.to_csv("output/ITS/effect_size_and_ci_results.csv", index=False)
print("\n--- Effect Sizes & Confidence Intervals ---")
print(comparison_df)

# Round all numeric columns to 3 decimals, except counts
numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
comparison_df[numeric_cols] = comparison_df[numeric_cols].round(3)
#comparison_df.to_csv("output/ITS/effect_size_and_ci_results.csv", index=False)

print("\n--- Effect Sizes & 95% Confidence Intervals (Rounded to 3 Decimals) ---")
print(comparison_df)

# ------------------
