# table 4: Country and Regional sentiment means

import pandas as pd
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    RegionalRoot = Path("output/regional")
    CountryRoot  = Path("output/country")

 #------------4 df inputs : -----------------------------------
 # 1. grouped2_data.pkl 2. normalized_data.pkl 3. grouped2_data_removed_retweets.pkl 4. processed_data.pkl
 # Outcoments one df each time
 #-------------------------------------------------------------
    df = pd.read_pickle("../data/normalized_data.pkl") # all tweets and retweets
    #df = df[df['Engagement Type'] != 'RETWEET'] # remove retweets
    #df = pd.read_pickle("../data/grouped2_data.pkl") # all tweets and retweets grouped by user
    #df= pd.read_pickle("../data/grouped2_data_removed_retweets.pkl") # all tweets without retweets grouped by user
    #df = pd.read_pickle("../data/grouped2_data.pkl").copy()
#-----------------------------------------------------------
    
    # Normalize Region text and exclude Unknown/Other/blank + NaN
    df['Region'] = df['Region'].astype(str).str.strip()
    #---------------------------
# Country-level analysis
#---------------------------
    US_REGIONS = {"Midwest", "Northeast", "Southeast", "Southwest", "West"}

    def map_country(region: str) -> str:
        r = (region or "").strip()
        if r in US_REGIONS:
            return "US"
        if r.lower() == "out of usa":
            return "Non-US"
        # anything else (if it sneaks through) -> Unknown
        return "Unknown"

    df["Country"] = df["Region"].apply(map_country)

    df_country = df[df["Country"].isin(["US", "Non-US", "Unknown"])].copy()

    country_group = (
        df_country.groupby("Country", as_index=False)["mean_sentiment"]
                  .agg(['mean', 'std', 'count'])
                  .reset_index()
                  .rename(columns={'index': 'Country'})
        )
    print(country_group)
    country_group.to_csv(f"{CountryRoot}_mean.csv", index=False)


    #---------------------------



    mask_valid_region = df['Region'].notna() & ~df['Region'].str.lower().isin(
        ["unknown", "other", ""]
    )
    df = df[mask_valid_region]

    # (Optional) keep only the five U.S. macro-regions used in the paper
    # allowed_regions = {"Midwest", "Northeast", "Southeast", "Southwest", "West"}
    # df = df[df['Region'].isin(allowed_regions)]

    # Ensure sentiment exists
    df = df.dropna(subset=['mean_sentiment'])

    # Aggregate
    region_group = (
        df.groupby('Region', as_index=False)['mean_sentiment']
          .agg(['mean', 'std', 'count'])
          .reset_index()
          .rename(columns={'index': 'Region'})
    )

    print(region_group.head())
    region_group.to_csv(f"{RegionalRoot}_mean.csv", index=False)


