import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt

# Set global font sizes and style
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

exclude_retweets = processed_df[processed_df['Engagement Type'] != 'RETWEET']
print(f"Number of rows after removing retweets: {len(exclude_retweets)}")
df = exclude_retweets.copy()
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])

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
#print(f"temporaldata: {temporal_data}")
temporal_df = pd.DataFrame(temporal_data)
interestsOutputRoot = "output/interests"
os.makedirs(interestsOutputRoot, exist_ok=True)
temporal_df.to_csv(f"{interestsOutputRoot}/interests_temporal.csv", index=False)
unique_interests = temporal_df['Interest'].unique()
unique_interests_df = pd.DataFrame(unique_interests, columns=['Interest'])
#print(f"intersts number: {len(unique_interests)}")


#  verified vs non-verified sentiment calculation
df['Twitter Verified'] = df['Twitter Verified'].fillna(False)  # Ensure no missing data for verified status
grouped_verified = df.groupby(['Twitter Verified', 'Bimonthly'])['mean_sentiment'].mean().reset_index()

#Group by Account Type and Bimonthly period to calculate mean sentiment
df['Account Type'] = df['Account Type'].fillna('Unknown')  # Fill with a default value
grouped_account = df.groupby(['Account Type', 'Bimonthly'])['mean_sentiment'].mean().reset_index()



grouped_region = df.groupby(['Region', 'Bimonthly'])['mean_sentiment'].mean().reset_index()
print(grouped_region.head())
grouped_region = grouped_region.dropna(subset=['mean_sentiment'])

# Plotting
#Create figure and axes
fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

#Define line styles for each subplot
gender_styles = ['-', '--', ':']
interest_styles = ['-', '--', ':', '-.', '-']
verified_styles = ['-', '--']
account_styles = ['-', '--', ':', '-.']
region_styles = ['-', '--', ':', '-.', '-', '--']

def style_subplot(ax, title):
    ax.set_title(title, pad=15, fontsize=15, fontweight='bold')
    ax.set_ylabel("Average Sentiment Score", fontsize=15, fontweight='bold')
    ax.tick_params(axis='both', labelsize=15, fontweight='bold')
    ax.grid(visible=True, linestyle='--', alpha=0.6)
    ax.legend(title_fontsize=15, fontsize=15, bbox_to_anchor=(1.02, 1), loc='upper left', fontweight='bold')

# Plot 1: Gender-based analysis
sns.lineplot(ax=axes[0], data=grouped_gender, x='Bimonthly', y='mean_sentiment', 
             hue='Gender', style='Gender',
             markers=['o', 's', '^'], dashes=[(1, 0), (2, 2), (3, 2)],
             markersize=8, linewidth=2.5, palette='Set2')

axes[0].set_title("Bimonthly Sentiment Analysis by Gender", fontsize=18, fontweight='bold')
axes[0].set_ylabel("Average Sentiment Score", fontsize=15, fontweight='bold')
axes[0].legend(title='Gender')
axes[0].grid(visible=True, linestyle='--', alpha=0.6)

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

top_interests_plot = temporal_df[temporal_df['Interest'].isin(top_popular_interests['Interest'])]
sns.lineplot(ax=axes[1], data=top_interests_plot,
             x='Bimonthly', y='Mean Sentiment', 
             hue='Interest', style='Interest',
             markers=['o', 's', '^', 'D', 'v'], 
             dashes=[(1, 0), (2, 2), (3, 2), (1, 2), (3, 1)],
             markersize=8, linewidth=2.5, palette='tab10')
axes[1].set_title("Bimonthly Sentiment Analysis by Interest", fontsize=18, fontweight='bold')
axes[1].set_ylabel("Average Sentiment Score", fontsize=15, fontweight='bold')
axes[1].legend(title='Interest')
axes[1].grid(visible=True, linestyle='--', alpha=0.6)

# Plot 3: Verified vs Non-Verified Users Analysis
sns.lineplot(ax=axes[2], data=grouped_verified, x='Bimonthly', y='mean_sentiment',
             hue='Twitter Verified', style='Twitter Verified',
             markers=['o', 's'], dashes=[(1, 0), (2, 2)],
             markersize=8, linewidth=2.5, palette='coolwarm')
axes[2].set_title("Bimonthly Sentiment Analysis by Verified Status", fontsize=18, fontweight='bold')
axes[2].set_ylabel("Average Sentiment Score", fontsize=15, fontweight='bold')
axes[2].legend(title='Twitter Verified')
axes[2].grid(visible=True, linestyle='--', alpha=0.6)

# Plot 4: Account Type Temporal Sentiment Analysis
sns.lineplot(ax=axes[3], data=grouped_account, x='Bimonthly', y='mean_sentiment',
             hue='Account Type', style='Account Type',
             markers=['o', 's', '^', 'D'], 
             dashes=[(1, 0), (2, 2), (3, 2), (1, 2)],
             markersize=8, linewidth=2.5, palette='Set2')
axes[3].set_title("Bimonthly Sentiment Analysis by Account Type", fontsize=18, fontweight='bold')
axes[3].set_ylabel("Average Sentiment Score", fontsize=15, fontweight='bold')
axes[3].legend(title='Account Type')
axes[3].grid(visible=True, linestyle='--', alpha=0.6)

# Plot 5: Account Type Temporal Sentiment Analysis
sns.lineplot(ax=axes[4], data=grouped_region, x='Bimonthly', y='mean_sentiment',
             hue='Region', style='Region',
             markers=['o', 's', '^', 'D', 'v', 'p'], 
             dashes=[(1, 0), (2, 2), (3, 2), (1, 2), (3, 1), (2, 1)],
             markersize=8, linewidth=2.5, palette='Set2')

axes[4].set_title("Bimonthly Sentiment Analysis by Regional", fontsize=18, fontweight='bold')
axes[4].set_ylabel("Average Sentiment Score", fontsize=15, fontweight='bold')
axes[4].legend(title='Region')
axes[4].grid(visible=True, linestyle='--', alpha=0.6)

for ax in axes:
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=20)
plt.subplots_adjust(right=0.85, bottom=0.1, hspace=0.4)


plt.margins(y=0.2) 
plt.tight_layout(pad=2, h_pad=1)  # h_pad adjusts vertical spacing between subplots


# Save the combined plot
output_dir = "output/bimonthly/"
plt.savefig(f"{output_dir}bimonthly_gender_interest_sentiment_analysis.V2.png",bbox_inches='tight')
plt.show()

