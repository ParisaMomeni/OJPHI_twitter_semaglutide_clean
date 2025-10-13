# --- SETUP ---
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.weightstats import DescrStatsW, CompareMeans
import statsmodels.api as sm
import os

df = pd.read_pickle('data/normalized_data.pkl')
# --- CLEANING ---
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])

# --- PART 2: INTERRUPTED TIME SERIES (ITS) ANALYSIS ---

# Create bimonthly bins
bin_edges = pd.date_range(start="2021-01-01", end="2024-12-30", freq="2MS")
labels = [f"{bin_edges[i].month}/{bin_edges[i].year}--{bin_edges[i + 1].month}/{bin_edges[i + 1].year}" 
          for i in range(len(bin_edges) - 1)]
df['Bimonthly'] = pd.cut(df['Date'], bins=bin_edges, labels=labels, right=True)

# Group by Bimonthly
ts_df = df.groupby('Bimonthly')['mean_sentiment'].mean().reset_index()
ts_df = ts_df.dropna(subset=['mean_sentiment']) 
ts_df['Time'] = np.arange(len(ts_df))

# ITS Event: define where the change happens (e.g., index of "11/2022--1/2023")
event_index = ts_df[ts_df['Bimonthly'] == "11/2022--1/2023"].index[0]
ts_df['Event'] = (ts_df.index >= event_index).astype(int)
ts_df['PostTime'] = ts_df['Time'] * ts_df['Event']

# Run ITS regression
X = sm.add_constant(ts_df[['Time', 'Event', 'PostTime']])
its_model = sm.OLS(ts_df['mean_sentiment'], X).fit()

# Print ITS model summary
print("\n--- Interrupted Time Series Model ---")
print(its_model.summary())

# Save summary to text file
with open("output/ITS/its_model_summary.txt", "w") as f:
    f.write(its_model.summary().as_text())

# --- PART 3: ITS PLOT ---

plt.figure(figsize=(12, 6))
sns.lineplot(data=ts_df, x='Time', y='mean_sentiment', marker='o', label='Observed Mean Sentiment')
plt.axvline(event_index, color='red', linestyle='--', label='Event Window Start')
plt.title("Interrupted Time Series Analysis of Sentiment", fontdict={'fontsize': 14}, fontweight='bold')
plt.xlabel("Bimonthly Time Index", fontsize=14)
plt.ylabel("Mean Sentiment", fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("output/ITS/its_sentiment_plot.png", dpi=300)
plt.show()
