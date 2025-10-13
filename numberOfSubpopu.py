#!/usr/bin/env python3
"""
Create Table 1 (per-author and per-tweet counts)

Inputs :
- data/grouped2_data.pkl   (one row per Author; Region already normalized)
- data/processed_data.pkl  (one row per Tweet)

Outputs:
- output/numberOfSubpopu/table1_number_of_authors.csv
- output/numberOfSubpopu/table1_number_of_tweets.csv
- output/numberOfSubpopu/region_top_summary.csv
- output/numberOfSubpopu/region_us_breakdown.csv
- output/numberOfSubpopu/region_top_summary_tweets.csv
- output/numberOfSubpopu/region_us_breakdown_tweets.csv
"""
import pandas as pd
from pathlib import Path

# -------- Paths --------
INPUT_Grouped_By_USER = "data/grouped2_data.pkl"
INPUT_Per_Tweets      = "data/normalized_data.pkl"

OUTPUT_AUTHORS = "output/numberOfSubpopu/table1_number_of_authors.csv"
OUTPUT_TWEETS  = "output/numberOfSubpopu/table1_number_of_tweets.csv"

OUT_TOP_AUTH   = "output/numberOfSubpopu/region_top_summary.csv"
OUT_US_AUTH    = "output/numberOfSubpopu/region_us_breakdown.csv"
OUT_TOP_TWEET  = "output/numberOfSubpopu/region_top_summary_tweets.csv"
OUT_US_TWEET   = "output/numberOfSubpopu/region_us_breakdown_tweets.csv"

Path("output/numberOfSubpopu").mkdir(parents=True, exist_ok=True)

# -------- Load --------
gdf = pd.read_pickle(INPUT_Grouped_By_USER)
print(f"Loaded {len(gdf):,} authors from {INPUT_Grouped_By_USER}")
tdf = pd.read_pickle(INPUT_Per_Tweets)
print(f"Loaded {len(tdf):,} tweets from {INPUT_Per_Tweets}")

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
REGIONS_US     = ["Midwest", "Northeast", "Southeast", "Southwest", "West"]
region_order   = REGIONS_US + ["Other", "out of usa", "Unknown"]

def get_region_series(df, prefer_norm=True):
    # Prefer Region_norm when present (tweets), else Region
    if prefer_norm and "Region_norm" in df.columns:
        ser = df["Region_norm"]
    else:
        ser = df.get("Region", pd.Series(index=df.index, dtype=object))
    ser = ser.fillna("Unknown").astype(str)
    # Keep only expected labels; anything else -> "Other"
    ser = ser.apply(lambda x: x if x in region_order else ("Unknown" if x.strip()=="" else "Other"))
    return ser

def build_table_one(df, is_tweets=False):
    # Normalize columns
    gender   = df.get("Gender", pd.Series(index=df.index, dtype=object)).apply(map_gender)
    verified = df.get("Verified", pd.Series(index=df.index, dtype=object)).apply(map_verified)
    acct     = df.get("Account_Type", pd.Series(index=df.index, dtype=object)).apply(map_acct)
    region   = get_region_series(df, prefer_norm=is_tweets)

    # Top-level region summary
    is_out      = region.eq("out of usa")
    is_unknown  = region.isin(["Unknown", "Null"]) | region.eq("")
    is_us       = region.isin(REGIONS_US + ["Other"])  # "Other" = US but unmapped state

    top_summary = pd.Series({
        "US": int(is_us.sum()),
        "out of usa": int(is_out.sum()),
        "Unknown/Null": int(is_unknown.sum())
    }).to_frame("Count")

    # US breakdown (only US+Other)
    us_region_counts = region[is_us].value_counts().reindex(REGIONS_US + ["Other"]).fillna(0).astype(int).to_frame("Count")

    # Sections for Table 1
    sections = [
        ("Gender",       ordered_counts(gender,   gender_order)),
        ("Verified",     ordered_counts(verified, verified_order)),
        ("Account Type", ordered_counts(acct,     acct_order)),
        ("Region",       ordered_counts(region,   region_order)),
    ]

    rows = []
    for attr, vc in sections:
        rows.append([attr, "", ""])  # section header
        for subpop, n in vc.items():
            rows.append(["", subpop, int(n)])

    label = "Number of Tweets" if is_tweets else "Number of Authors"
    table1 = pd.DataFrame(rows, columns=["Attribute", "Subpopulation", label])
    
    return table1, top_summary, us_region_counts

# -------- Build and Save: AUTHORS --------
table1_auth, top_auth, us_auth = build_table_one(gdf, is_tweets=False)
table1_auth.to_csv(OUTPUT_AUTHORS, index=False)
top_auth.to_csv(OUT_TOP_AUTH)
us_auth.to_csv(OUT_US_AUTH)
print(f"Saved Table 1 (authors) → {OUTPUT_AUTHORS}")
print(f"Saved region summaries (authors) → {OUT_TOP_AUTH}, {OUT_US_AUTH}")

# -------- Build and Save: TWEETS --------
table1_twt, top_twt, us_twt = build_table_one(tdf, is_tweets=True)
tweet_sentiment_counts = tdf["mean_sentiment"].value_counts().sort_index()
print(f"Sentiment counts: {tweet_sentiment_counts.to_dict()}")
table1_twt.to_csv(OUTPUT_TWEETS, index=False)
top_twt.to_csv(OUT_TOP_TWEET)
us_twt.to_csv(OUT_US_TWEET)
print(f"Saved Table 1 (tweets) → {OUTPUT_TWEETS}")
print(f"Saved region summaries (tweets) → {OUT_TOP_TWEET}, {OUT_US_TWEET}")
