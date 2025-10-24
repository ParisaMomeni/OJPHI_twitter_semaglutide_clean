#!/usr/bin/env python3
# for mean sentiment grouped by user I have 2 options: 1. continues value or 2. discrete {−1,0,1} at author level author_label=('sentiment_label', majority_label), I chose contiuous mean sentiment value
"""
Aggregate processed per-tweet data into per-author grouped data.

Input : data/processed_data.pkl  (per-tweet, already has sentiment_label & metadata)
Output: data/grouped2_data.pkl   (per-author, with mean_sentiment & most_common metadata)

Dedup policy: drop exact duplicate rows (optional); we keep all tweets per author so
most_common can resolve categorical fields correctly.
"""

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
    #empty series to avoid chained assignment warnings
    norm = pd.Series(index=df.index, dtype=object)

    # Determine USA / non-USA
    in_usa = df[country_col].apply(_is_usa) if country_col in df.columns else pd.Series(False, index=df.index)

    # For non-USA rows
    norm.loc[~in_usa] = "out of usa"

    # For USA rows: map state -> region
    if region_col in df.columns:
        states = df[region_col].apply(_clean_str)
        mapped = states.map(STATE_TO_REGION)
        # Fill mapped where in USA
        norm.loc[in_usa] = mapped.loc[in_usa]

        # Unmapped USA states/labels: set Unknown if empty/unknown-like; else Other
        unknown_like = states.isin(["", "unknown", "nan", "none", "null"])
        # For USA + unmapped + unknown_like -> 'Unknown'
        idx_unknown = in_usa & mapped.isna() & unknown_like
        norm.loc[idx_unknown] = "Unknown"
        # For USA + unmapped + not unknown_like -> 'Other'
        idx_other = in_usa & mapped.isna() & (~unknown_like)
        norm.loc[idx_other] = "Other"
    else:
        # No region column: mark Unknown for USA rows not yet set
        norm.loc[in_usa & norm.isna()] = "Unknown"

    # For any leftover NaNs (shouldn't happen): set 'Unknown'
    norm = norm.fillna("Unknown")
    return norm

def most_common(x):
    vc = x.value_counts(dropna=True)
    return vc.index[0] if not vc.empty else None

def collect_all_interests(series):
    """
    Combine all interests mentioned by a user into one comma-separated string.
    Removes duplicates and empty entries.
    """
    all_items = []
    for cell in series.dropna():
        items = [x.strip() for x in str(cell).split(",") if x.strip()]
        all_items.extend(items)

    # Deduplicate and sort
    unique_items = sorted(set(all_items))
    return ", ".join(unique_items) if unique_items else ""

def group_by_author(df):
    grouped_df = df.groupby("Author").agg(
        mean_sentiment=("sentiment_label", "mean"),   # continuous [-1, 1]
        Gender=("Gender", most_common),
        Country=("Country", most_common),
        Region=("Region_norm", most_common),          # <-- use normalized region
        Engagement_Type=("Engagement Type", most_common),
        Interest=("Interest", collect_all_interests),
        Account_Type=("Account Type", most_common),
        Verified=("Twitter Verified", most_common),
    ).reset_index()
    return grouped_df

# ---------- Main ----------
if __name__ == "__main__":
    print("Loading processed per-tweet data...")
    df = pd.read_pickle("data/processed_data.pkl")
    print(f"Loaded {len(df):,} tweets")

    # Optional: remove exact duplicate rows
    df = df.drop_duplicates()

    df["Region_norm"] = normalize_region(df, region_col="Region", country_col="Country")

    print("Aggregating to author level...")
    grouped_df = group_by_author(df)

    print("Saving grouped data...")
    grouped_df.to_pickle("data/grouped2_data.pkl")
    print(f"Done. Saved {len(grouped_df):,} authors to data/grouped2_data.pkl")
