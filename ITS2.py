# it is same as ITS.py but colored annotations have been added
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

# --- LOAD AND CLEAN ---
df = pd.read_pickle('data/normalized_data.pkl')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])

# --- INTERRUPTED TIME SERIES (ITS) ANALYSIS ---

# Create bimonthly bins
bin_edges = pd.date_range(start="2021-01-01", end="2024-12-30", freq="2MS")
labels = [f"{bin_edges[i].month}/{bin_edges[i].year}--{bin_edges[i + 1].month}/{bin_edges[i + 1].year}"
          for i in range(len(bin_edges) - 1)]
df['Bimonthly'] = pd.cut(df['Date'], bins=bin_edges, labels=labels, right=True)

# Group by Bimonthly
ts_df = df.groupby('Bimonthly')['mean_sentiment'].mean().reset_index()
ts_df = ts_df.dropna(subset=['mean_sentiment'])
ts_df['Time'] = np.arange(len(ts_df))

# Define ITS intervention
event_index = ts_df[ts_df['Bimonthly'] == "11/2022--1/2023"].index[0]
ts_df['Event'] = (ts_df.index >= event_index).astype(int)
ts_df['PostTime'] = ts_df['Time'] * ts_df['Event']

# Run ITS regression
X = sm.add_constant(ts_df[['Time', 'Event', 'PostTime']])
its_model = sm.OLS(ts_df['mean_sentiment'], X).fit()

print("\n--- Interrupted Time Series Model ---")
print(its_model.summary())

# Save model summary
os.makedirs("output/ITS", exist_ok=True)
with open("output/ITS/its_model_summary.txt", "w") as f:
    f.write(its_model.summary().as_text())

# --- ITS PLOT ---
ts_df['Bimonthly'] = pd.Categorical(ts_df['Bimonthly'], categories=labels, ordered=True)

plt.figure(figsize=(14, 6))
sns.lineplot(data=ts_df, x='Bimonthly', y='mean_sentiment',
             marker='o', label='Observed Mean Sentiment', linewidth=2)
plt.axvline(x="11/2022--1/2023", color='red', linestyle='--', label='Event Window Start')

# === Annotate External Events ===
annotations = [
    ("5/2022--7/2022", "Mid 2022 – Ozempic shortage begins"),
    ("11/2022--1/2023", "Adverse side effect reports"),
    ("1/2023--3/2023", "Celebrity endorsements & 'Hollywood drug' media"),
    ("9/2023--11/2023", "SELECT trial – positive CV outcomes"),
    ("1/2024--3/2024", "CROI & FDA label expansion announcements")
]

colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
y_offsets = [0.07, -0.03, 0.13, 0.06, -0.06]
x_offsets = [-0.10, -1.90, -0.90, -0.50, -3.65]

for i, (x, text) in enumerate(annotations):
    if x in ts_df['Bimonthly'].values:
        x_val = ts_df.loc[ts_df['Bimonthly'] == x].index[0]
        y_val = ts_df.loc[ts_df['Bimonthly'] == x, 'mean_sentiment'].values[0]

        # Add small colored circle at annotation date
        plt.scatter(x_val, y_val, s=80, color=colors[i], zorder=5, edgecolors='white', linewidths=1.2)

        # Add arrow + colored text box
        plt.annotate(
            text,
            xy=(x_val, y_val),
            xytext=(x_val + x_offsets[i], y_val + y_offsets[i]),
            textcoords='data',
            fontsize=11,
            fontweight='bold',
            ha='left',
            va='bottom',
            color=colors[i],
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='white',
                edgecolor=colors[i],
                linewidth=1.4,
                alpha=0.9
            ),
            arrowprops=dict(
                arrowstyle='->',
                color=colors[i],
                lw=1.6,
                shrinkA=4,
                shrinkB=5,
                connectionstyle='arc3,rad=0.1'
            )
        )

# === Styling ===
#plt.title("Interrupted Time Series Analysis of Sentiment with External Events",  fontsize=17, fontweight='bold')
plt.xlabel("Bimonthly Period", fontsize=15, fontweight='bold')
plt.ylabel("Mean Sentiment", fontsize=15, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=13, fontweight='bold')
plt.yticks(fontsize=13, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)

plt.legend(
    loc='upper right',
    frameon=True,
    fontsize=12,
    title_fontsize=13,
    fancybox=True,
    framealpha=0.85
)

plt.tight_layout()
plt.savefig("output/ITS/its_sentiment_plot_with_events.png", dpi=400, bbox_inches='tight')
plt.show()
