#!/usr/bin/env python3


import pandas as pd

# ---------- Config ----------
US_ALIASES = {"United States of America","united states", "united states of america", "usa", "us", "u.s.", "u.s.a."}

STATE_TO_REGION = {
    "Alabama": "Southeast",
    "Alaska": "West",
    "Arizona": "Southwest",
    "Arkansas": "Southeast",
    "California": "West",
    "Colorado": "West",
    "Connecticut": "Northeast",
    "Delaware": "Northeast",
    "Florida": "Southeast",
    "Georgia": "Southeast",
    "Hawaii": "West",
    "Idaho": "West",
    "Illinois": "Midwest",
    "Indiana": "Midwest",
    "Iowa": "Midwest",
    "Kansas": "Midwest",
    "Kentucky": "Southeast",
    "Louisiana": "Southeast",
    "Maine": "Northeast",
    "Maryland": "Northeast",
    "Massachusetts": "Northeast",
    "Michigan": "Midwest",
    "Minnesota": "Midwest",
    "Mississippi": "Southeast",
    "Missouri": "Midwest",
    "Montana": "West",
    "Nebraska": "Midwest",
    "Nevada": "West",
    "New Hampshire": "Northeast",
    "New Jersey": "Northeast",
    "New Mexico": "Southwest",
    "New York": "Northeast",
    "North Carolina": "Southeast",
    "North Dakota": "Midwest",
    "Ohio": "Midwest",
    "Oklahoma": "Southwest",
    "Oregon": "West",
    "Pennsylvania": "Northeast",
    "Rhode Island": "Northeast",
    "South Carolina": "Southeast",
    "South Dakota": "Midwest",
    "Tennessee": "Southeast",
    "Texas": "Southwest",
    "Utah": "West",
    "Vermont": "Northeast",
    "Virginia": "Southeast",
    "Washington": "West",
    "West Virginia": "Southeast",
    "Wisconsin": "Midwest",
    "Wyoming": "West",
    "District of Columbia": "Northeast",
}

# ---------- Helpers ----------
def _clean_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def _is_usa(Country):
    s = _clean_str(Country).lower()
    return s in US_ALIASES

def normalize_region(df, region_col="Region", country_col="Country"):
    """
    Create a normalized region column:
      - If Country ∉ USA → 'out of usa'
      - If Country ∈ USA and Region looks like a state → map via STATE_TO_REGION
      - If missing/empty → 'Unknown'
      - Otherwise → 'Other'
    """
    norm = pd.Series(index=df.index, dtype=object)

    in_usa = df[country_col].apply(_is_usa) if country_col in df.columns else pd.Series(False, index=df.index)

    norm.loc[~in_usa] = "out of usa"

    if region_col in df.columns:
        states = df[region_col].apply(_clean_str)
        mapped = states.map(STATE_TO_REGION)
        norm.loc[in_usa] = mapped.loc[in_usa]

        unknown_like = states.isin(["", "unknown", "nan", "none", "null"])
        idx_unknown = in_usa & mapped.isna() & unknown_like
        norm.loc[idx_unknown] = "Unknown"
        idx_other = in_usa & mapped.isna() & (~unknown_like)
        norm.loc[idx_other] = "Other"
    else:
        norm.loc[in_usa & norm.isna()] = "Unknown"

    norm = norm.fillna("Unknown")
    return norm


def normalize_cols(df):
    df["mean_sentiment"] = df["sentiment_label"] 
    df["Region"] = df["Region_norm"]
    df["Engagement_Type"] = df["Engagement Type"]
    df["Account_Type"] = df["Account Type"]
    df["Verified"] = df["Twitter Verified"]
    return df

# ---------- Main ----------
if __name__ == "__main__":
    print("Loading processed per-tweet data...")
    df = pd.read_pickle("data/processed_data.pkl")
    print(f"Loaded {len(df):,} tweets")

    # Optional: remove exact duplicate rows
    df = df.drop_duplicates()

    df["Region_norm"] = normalize_region(df, region_col="Region", country_col="Country")

    normalized_df = normalize_cols(df)

    print("Saving grouped data...")
    normalized_df.to_pickle("data/normalized_data.pkl")
    print(f"Done. Saved {len(normalized_df):,} tweets to data/normalized_data.pkl")
