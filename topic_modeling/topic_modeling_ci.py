import pandas as pd
import numpy as np
from scipy.stats import norm, chi2_contingency

state_to_region = {
        'Alabama': 'Southeast',
        'Alaska': 'West',
        'Arizona': 'Southwest',
        'Arkansas': 'Southeast',
        'California': 'West',
        'Colorado': 'West',
        'Connecticut': 'Northeast',
        'Delaware': 'Northeast',
        'Florida': 'Southeast',
        'Georgia': 'Southeast',
        'Hawaii': 'West',
        'Idaho': 'West',
        'Illinois': 'Midwest',
        'Indiana': 'Midwest',
        'Iowa': 'Midwest',
        'Kansas': 'Midwest',
        'Kentucky': 'Southeast',
        'Louisiana': 'Southeast',
        'Maine': 'Northeast',
        'Maryland': 'Northeast',
        'Massachusetts': 'Northeast',
        'Michigan': 'Midwest',
        'Minnesota': 'Midwest',
        'Mississippi': 'Southeast',
        'Missouri': 'Midwest',
        'Montana': 'West',
        'Nebraska': 'Midwest',
        'Nevada': 'West',
        'New Hampshire': 'Northeast',
        'New Jersey': 'Northeast',
        'New Mexico': 'Southwest',
        'New York': 'Northeast',
        'North Carolina': 'Southeast',
        'North Dakota': 'Midwest',
        'Ohio': 'Midwest',
        'Oklahoma': 'Southwest',
        'Oregon': 'West',
        'Pennsylvania': 'Northeast',
        'Rhode Island': 'Northeast',
        'South Carolina': 'Southeast',
        'South Dakota': 'Midwest',
        'Tennessee': 'Southeast',
        'Texas': 'Southwest',
        'Utah': 'West',
        'Vermont': 'Northeast',
        'Virginia': 'Southeast',
        'Washington': 'West',
        'West Virginia': 'Southeast',
        'Wisconsin': 'Midwest',
        'Wyoming': 'West',
        'District of Columbia': 'Northeast',
}

df = pd.read_csv("../../data/Semaglutide_Twitter_Topic_mapping.csv") # Full dataset not included in repository
df['state_to_region'] = df['Region'].map(state_to_region)
df['state_to_region'] = df['state_to_region'].fillna('Unknown')
df_topics_only = df.dropna(subset=["Topic_mapping", "Date"])
df_topics_only['Topic_mapping'] = df_topics_only['Topic_mapping'].astype(int)

def calculate_ci_for_subpop(attribute = None, value = None):
    df_filtered = df_topics_only
    if attribute is not None and value is not None:
        if attribute == "Interest":
            df_filtered = df[df["Interest"].str.contains(value, na=False)]
        else:
            df_filtered = df_topics_only[df_topics_only[attribute] == value]
    counts = df_filtered["Topic_mapping"].value_counts(sort=False).sort_index()
    total = counts.sum()
    proportions = counts / total

    # Normal approximation confidence intervals
    z = norm.ppf(0.975)  # 95% CI -> 1.96
    stderr = np.sqrt(proportions * (1 - proportions) / total)

    ci_low = proportions - z * stderr
    ci_high = proportions + z * stderr

    ci_df = pd.DataFrame({
        "count": counts,
        "proportion": proportions,
        "ci_low": ci_low,
        "ci_high": ci_high
    }).round(3)

    # Compare observed distribution to uniform (expected equally likely categories)
    expected = np.ones(len(counts)) * total / len(counts)
    chi2, _, _, _ = chi2_contingency([counts, expected])

    # Cramér’s V formula (https://press.princeton.edu/books/paperback/9780691005478/mathematical-methods-of-statistics)
    k = len(counts)  # number of categories
    cramers_v = np.sqrt(chi2 / (total * (min(k-1, 1))))  # For 1D vs uniform, min(k-1,1)

    print(ci_df)
    print("\nCramér's V (vs uniform distribution):", round(cramers_v, 3))

#calculate_ci_for_subpop()
calculate_ci_for_subpop("Gender", "male")