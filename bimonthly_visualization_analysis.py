import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15  # Base font size
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['legend.title_fontsize'] = 15
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'



# processed_df = pd.read_pickle('data/grouped2_data.pkl') what I want to do with user Date in aggregation. first date? last date? most common date? So it make more sense to use per tweet data
processed_df = pd.read_pickle('data/normalized_data.pkl')

df = processed_df.copy()
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])

df['Gender'] = df['Gender'].replace({
    'male': 'Male',
    'female': 'Female', #here if want to add 95% CI label
    'unknown': 'Unknown'
})
# Standardize Account_Type
df['Account Type'] = df['Account Type'].replace({
    'individual': 'Individual',
    'organisational': 'Organizational',
    'unknown': 'Unknown'
})


bin_edges = pd.date_range(start="2021-01-01", end="2024-12-30", freq="2MS")
labels = [f"{bin_edges[i].month}/{bin_edges[i].year}--{bin_edges[i + 1].month}/{bin_edges[i + 1].year}" for i in range(len(bin_edges) - 1)]
df['Bimonthly'] = pd.cut(df['Date'], bins=bin_edges, labels=labels, right=True)
grouped_gender = df.groupby(['Gender', 'Bimonthly'])['mean_sentiment'].mean().reset_index()

sentimentByInterestTemporal = {}
for _, row in df.iterrows():
    if not pd.isna(row["Interest"]):
        sentiment_value = int(row["mean_sentiment"])  # -1, 0, or 1
        bimonthly_period = row["Bimonthly"]
        for interest in row["Interest"].split(","):
            interest = interest.strip()
            if interest not in sentimentByInterestTemporal:
                sentimentByInterestTemporal[interest] = {}
            if bimonthly_period not in sentimentByInterestTemporal[interest]:
                sentimentByInterestTemporal[interest][bimonthly_period] = {"negative": 0, "neutral": 0, "positive": 0}
            if sentiment_value == -1:
                sentimentByInterestTemporal[interest][bimonthly_period]["negative"] += 1
            elif sentiment_value == 0:
                sentimentByInterestTemporal[interest][bimonthly_period]["neutral"] += 1
            elif sentiment_value == 1:
                sentimentByInterestTemporal[interest][bimonthly_period]["positive"] += 1

temporal_data = []
for interest, periods in sentimentByInterestTemporal.items():
    for period, counts in periods.items():
        total = sum(counts.values())
        if total > 0:
            real_mean_sentiment = (counts["positive"] - counts["negative"]) / total
            temporal_data.append({
                "Interest": interest,
                "Bimonthly": period,
                "Mean Sentiment": real_mean_sentiment,
                "Count": total
                })
temporal_df = pd.DataFrame(temporal_data)
interestsOutputRoot = "output/interests"
os.makedirs(interestsOutputRoot, exist_ok=True)
temporal_df.to_csv(f"{interestsOutputRoot}/interests_temporal.csv", index=False)
unique_interests = temporal_df['Interest'].unique()
unique_interests_df = pd.DataFrame(unique_interests, columns=['Interest'])


#  verified vs non-verified sentiment calculation
df['Twitter Verified'] = df['Twitter Verified'].astype(str).str.strip().str.lower()
df['Twitter Verified'] = df['Twitter Verified'].replace({
    "true": "Verified", "t": "Verified", "yes": "Verified", "verified": "Verified",
    "false": "Non-Verified", "f": "Non-Verified", "no": "Non-Verified", "unverified": "Non-Verified",
    "": "Unknown", "unknown": "Unknown", "nan": "Unknown", "none": "Unknown"
})
df['Twitter Verified'] = df['Twitter Verified'].replace({
    'Verified': 'Verified', #if wnat to add 95% CI label: 'Verified': 'Verified 95%',
    'Non-Verified': 'Non-Verified',
    'Unknown': 'Unknown'
})
unknown_verified_df = df[df['Twitter Verified'] == "Unknown"]
print(unknown_verified_df['Date'])

first_date = unknown_verified_df['Date'].min()
last_date  = unknown_verified_df['Date'].max()
common_date = unknown_verified_df['Date'].mode()[0] if not unknown_verified_df['Date'].mode().empty else None
print("Unknown Verified - Date Summary:")
print("Earliest:", first_date)
print("Latest:", last_date)
print("Most Common:", common_date)


    # Count Unknown only
unknown_count = (df['Twitter Verified'] == "Unknown").sum()
print("Number of Unknown Verified entries:", unknown_count)

counts = df['Twitter Verified'].value_counts(dropna=False)
print(counts)

#grouped_verified = df.groupby(['Twitter Verified', 'Bimonthly'])['mean_sentiment'].mean().reset_index()
grouped_verified = (
    df.groupby(['Twitter Verified', 'Bimonthly'])['mean_sentiment']
      .mean()
      .reset_index()
)
#--------------------- Account Type Sentiment Calculation ----------------

df['Account Type'] = df['Account Type'].fillna('Unknown') 
grouped_account = df.groupby(['Account Type', 'Bimonthly'])['mean_sentiment'].mean().reset_index()


# ---- Country mapping (US / Non-US / Unknown) ----
US_REGIONS = ["Midwest", "Northeast", "Southeast", "Southwest", "West"]
df["Region"] = df["Region"].astype(str).str.strip()

def map_country(region):
    if region in US_REGIONS:
        return "US"
    if region.strip().lower() == "out of usa":
        return "Non-US"
    return "Unknown"
df["Country"] = df["Region"].apply(map_country)

# Aggregate by Country + Bimonthly
grouped_country = (
    df.groupby(["Country", "Bimonthly"])["mean_sentiment"]
      .mean()
      .reset_index()
)
#---------------------------------------------------

df_us_only = df[df["Region"].isin(US_REGIONS)].copy()
df_us_only["Region"] = pd.Categorical(df_us_only["Region"],
                                      categories=US_REGIONS, ordered=True)
grouped_region = (
    df_us_only.groupby(["Region", "Bimonthly"])["mean_sentiment"]
              .mean()
              .reset_index()
)
print(grouped_region.head())
grouped_region = grouped_region.dropna(subset=['mean_sentiment'])

#-----------------------------------------------------
# === HELPER FUNCTION: COMPUTE 95% CI ===
def compute_ci(data, group_cols, value_col='mean_sentiment'):
    stats = data.groupby(group_cols)[value_col].agg(['mean', 'std', 'count']).reset_index()
    stats['sem'] = stats['std'] / np.sqrt(stats['count'])
    stats['ci95'] = 1.96 * stats['sem']
    return stats.rename(columns={'mean': 'mean_sentiment'})

# === GROUPING ===
gender_stats = compute_ci(df, ['Gender', 'Bimonthly'])
verified_stats = compute_ci(df, ['Twitter Verified', 'Bimonthly'])
account_stats = compute_ci(df, ['Account Type', 'Bimonthly'])
country_stats = compute_ci(df, ['Country', 'Bimonthly'])

region_stats = compute_ci(df_us_only, ['Region', 'Bimonthly'])
#-----------------------------------------------------

# Plotting
fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
gender_styles = ['-', '--', ':']
interest_styles = ['-', '--', ':', '-.', '-']
verified_styles = ['-', '--']
account_styles = ['-', '--', ':', '-.']
region_styles = ['-', '--', ':', '-.', '-', '--']

def style_subplot(ax, title):
    ax.set_title(title, pad=15, fontsize=15, fontweight='bold')
    ax.set_ylabel("Avg Sentiment", fontsize=15, fontweight='bold')
    ax.tick_params(axis='both', labelsize=15, fontweight='bold')
    ax.grid(visible=True, linestyle='--', alpha=0.6)
    ax.legend(title_fontsize=15, fontsize=15, bbox_to_anchor=(1.02, 1), loc='upper left', fontweight='bold')

# Plot 1: Gender-based analysis
sns.lineplot(ax=axes[0], data=gender_stats, x='Bimonthly', y='mean_sentiment', 
             hue='Gender', style='Gender',
             markers=['o', 's', '^'], dashes=[(1, 0), (2, 2), (3, 2)],
             markersize=8, linewidth=2.5, palette='Set2')

#  CI bands
for gender, grp in gender_stats.groupby('Gender'):
    axes[0].fill_between(grp['Bimonthly'], 
                         grp['mean_sentiment'] - grp['ci95'], 
                         grp['mean_sentiment'] + grp['ci95'],
                         alpha=0.2, 
                         color=sns.color_palette('Set2')[list(gender_stats['Gender'].unique()).index(gender)],
                         label=None)  

axes[0].set_title("Bimonthly Sentiment Analysis by Gender", fontsize=18, fontweight='bold')
axes[0].set_ylabel("Avg Sentiment", fontsize=15, fontweight='bold')
axes[0].grid(visible=True, linestyle='--', alpha=0.6)
axes[0].legend(title='Gender', loc='upper left', bbox_to_anchor=(1.02, 1))

    

# Plot 2: Interest-based analysis (choose top 5 interests for clarity)
# Calculate the total popularity for each interest
interest_popularity = {}
for interest, periods in sentimentByInterestTemporal.items():
    total_count = sum(
        sum(period_data.values()) for period_data in periods.values()
    )
    interest_popularity[interest] = total_count
popularity_df = pd.DataFrame(
    list(interest_popularity.items()), columns=["Interest", "Count"]
)
popularity_df = popularity_df.sort_values(by="Count", ascending=False)
top_popular_interests = popularity_df.head(5)
print(f"top interests: {top_popular_interests}")
top_interest_df = df[df['Interest'].isin(top_popular_interests['Interest'])]
print(top_interest_df)
top_interests = top_interest_df.groupby(['Interest'])['mean_sentiment'].mean().reset_index()
interest_stats = compute_ci(df[df['Interest'].isin(top_popular_interests['Interest'])],
                            ['Interest', 'Bimonthly'])

top_interests_plot = temporal_df[temporal_df['Interest'].isin(top_popular_interests['Interest'])]



sns.lineplot(ax=axes[1], data=interest_stats,
             x='Bimonthly', y='mean_sentiment',
             hue='Interest', style='Interest',
             markers=['o', 's', '^', 'D', 'v'],
             dashes=[(1, 0), (2, 2), (3, 2), (1, 2), (3, 1)],
             markersize=8, linewidth=2.5, palette='tab10')

#  95% CI bands
palette = sns.color_palette('tab10', n_colors=len(interest_stats['Interest'].unique()))
for i, (interest, grp) in enumerate(interest_stats.groupby('Interest')):
    axes[1].fill_between(grp['Bimonthly'],
                         grp['mean_sentiment'] - grp['ci95'],
                         grp['mean_sentiment'] + grp['ci95'],
                         alpha=0.2, color=palette[i])

axes[1].set_title("Bimonthly Sentiment Analysis by Interest", fontsize=18, fontweight='bold')
axes[1].set_ylabel("Avg Sentiment", fontsize=15, fontweight='bold')
axes[1].legend(title='Interest')
axes[1].grid(visible=True, linestyle='--', alpha=0.6)

# Plot 3: Verified vs Non-Verified Users Analysis
verified_order = ['Verified', 'Non-Verified', 'Unknown']
verified_palette = {
    'Verified': '#EA5F94',
    'Non-Verified': '#3B4CC0',
    'Unknown': '#999999'
}

sns.lineplot(ax=axes[2], data=verified_stats, x='Bimonthly', y='mean_sentiment',
             hue='Twitter Verified', style='Twitter Verified',
             hue_order=verified_order, style_order=verified_order,
             markers=['o', 's', '^'], dashes=[(1, 0), (2, 2), (3, 2)],
             markersize=8, linewidth=2.5, palette=verified_palette)

#  CI bands manually
for status in verified_order:
    grp = verified_stats[verified_stats['Twitter Verified'] == status].dropna(subset=['mean_sentiment', 'ci95'])
    
    if not grp.empty:
        axes[2].fill_between(grp['Bimonthly'],
                             grp['mean_sentiment'] - grp['ci95'],
                             grp['mean_sentiment'] + grp['ci95'],
                             alpha=0.2,
                             color=verified_palette[status],
                             label=None)


#  styling
axes[2].set_title("Bimonthly Sentiment Analysis by Verified Status", fontsize=18, fontweight='bold')
axes[2].set_ylabel("Avg Sentiment", fontsize=15, fontweight='bold')
axes[2].grid(visible=True, linestyle='--', alpha=0.6)
axes[2].legend(title='Twitter Verified', loc='upper left', bbox_to_anchor=(1.02, 1))

# Plot 4: Account Type Temporal Sentiment Analysis
sns.lineplot(ax=axes[3], data=account_stats, x='Bimonthly', y='mean_sentiment',
             hue='Account Type', style='Account Type',
             markers=['o', 's'],  # Only keep markers for known categories
             dashes=[(1, 0), (2, 2)],
             markersize=8, linewidth=2.5, palette='Set2')
for i, (acct_type, grp) in enumerate(account_stats.groupby('Account Type')):
    axes[3].fill_between(grp['Bimonthly'],
                         grp['mean_sentiment'] - grp['ci95'],
                         grp['mean_sentiment'] + grp['ci95'],
                         alpha=0.2,
                         color=sns.color_palette('Set2')[i],
                         label=None)  # Avoid extra legend lines

axes[3].set_title("Bimonthly Sentiment Analysis by Account Type", fontsize=18, fontweight='bold')
axes[3].set_ylabel("Avg Sentiment", fontsize=15, fontweight='bold')
axes[3].grid(visible=True, linestyle='--', alpha=0.6)
axes[3].legend(title='Account Type', loc='upper left', bbox_to_anchor=(1.02, 1))


# Plot 5: Country (US vs Non-US)
'''sns.lineplot(ax=axes[4], data=grouped_country, x='Bimonthly', y='mean_sentiment',
             hue='Country', style='Country',
             markers=['o', 's', '^'], dashes=[(1, 0), (2, 2), (3, 2)],
             markersize=8, linewidth=2.5, palette='Set2')

axes[4].set_title("Bimonthly Sentiment Analysis by Country", fontsize=18, fontweight='bold')
axes[4].set_ylabel("Avg Sentiment", fontsize=15, fontweight='bold')
axes[4].legend(title='Country')
axes[4].grid(visible=True, linestyle='--', alpha=0.6)'''

# Plot 5: Regional Temporal Sentiment Analysis
sns.lineplot(ax=axes[4], data=region_stats,
             x='Bimonthly', y='mean_sentiment',
             hue='Region', style='Region',
             markers=['o','s','^','D','v'],
             dashes=[(1,0),(2,2),(3,2),(1,2),(3,1)],
             markersize=8, linewidth=2.5, palette='Set2')

#  95% CI bands
palette = sns.color_palette('Set2', n_colors=len(region_stats['Region'].unique()))
for i, (region, grp) in enumerate(region_stats.groupby('Region')):
    axes[4].fill_between(grp['Bimonthly'],
                         grp['mean_sentiment'] - grp['ci95'],
                         grp['mean_sentiment'] + grp['ci95'],
                         alpha=0.2, color=palette[i])

axes[4].set_title("Bimonthly Sentiment Analysis by US Region", fontsize=18, fontweight='bold')
axes[4].set_ylabel("Avg Sentiment", fontsize=15, fontweight='bold')
axes[4].legend(title='US Region')
axes[4].grid(visible=True, linestyle='--', alpha=0.6)

for ax in axes:
    ax.legend(             # explanation boxes (legends) outside
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=15,
        title_fontsize=15
    )
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=20)
plt.subplots_adjust(right=0.85, bottom=0.1, hspace=0.4)


plt.margins(y=0.2) 
plt.tight_layout(pad=2, h_pad=1)  # h_pad adjusts vertical spacing between subplots


output_dir = "output/bimonthly/"
plt.savefig(f"{output_dir}bimonthly_gender_interest_sentiment_analysis.V1.png",bbox_inches='tight')
plt.show()