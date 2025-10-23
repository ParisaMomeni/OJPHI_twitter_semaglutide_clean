# Breakdown of topics by sentiment
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from matplotlib import rcParams

plt.rcParams['font.family'] = 'Serif'
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

df = pd.read_csv("../../data/Semaglutide_Twitter_Topic_mapping.csv") # Full dataset not included in repository

# Create a figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(15, 16))

# Plot for Gender
gender_counts = df.groupby(['Topic_mapping', 'Gender', 'roberta_sentiment']).size().reset_index(name='count')
pivot_gender = gender_counts.pivot_table(index=['Topic_mapping', 'Gender'], columns='roberta_sentiment', values='count', fill_value=0)
ax1 = pivot_gender.plot(
    kind='bar',
    stacked=True,
    color=['blue', 'red'],
    edgecolor='black',
    ax=axes[0] # Plot on the first subplot
)

bars1 = ax1.patches
n1 = len(pivot_gender)
hatches = ['///', '\\\\\\']

for i, bar in enumerate(bars1):
    hatch = hatches[i // n1]
    bar.set_hatch(hatch)

ax1.legend(title='Sentiment', labels=['Negative', 'Positive'], loc='best')
ax1.set_title('Breakdown of Sentiment by Topic Number and Gender')
ax1.set_xlabel('Topic Number / Gender')
ax1.set_ylabel('Number of Posts')

# Plot for Verification Status
verified_counts = df.groupby(['Topic_mapping', 'Twitter Verified', 'roberta_sentiment']).size().reset_index(name='count')
pivot_verified = verified_counts.pivot_table(index=['Topic_mapping', 'Twitter Verified'], columns='roberta_sentiment', values='count', fill_value=0)
ax2 = pivot_verified.plot(
    kind='bar',
    stacked=True,
    color=['blue', 'red'],
    edgecolor='black',
    ax=axes[1] # Plot on the second subplot
)

bars2 = ax2.patches
n2 = len(pivot_verified)
for i, bar in enumerate(bars2):
    hatch = hatches[i // n2]
    bar.set_hatch(hatch)

ax2.legend(title='Sentiment', labels=['Negative', 'Positive'], loc='best')
ax2.set_title('Breakdown of Sentiment by Topic Number and Verification Status')
ax2.set_xlabel('Topic Number / Verification Status')
ax2.set_ylabel('Number of Posts')

# Plot for Account Type
acct_type_counts = df.groupby(['Topic_mapping', 'Account Type', 'roberta_sentiment']).size().reset_index(name='count')
pivot_acct_type = acct_type_counts.pivot_table(index=['Topic_mapping', 'Account Type'], columns='roberta_sentiment', values='count', fill_value=0)
ax3 = pivot_acct_type.plot(
    kind='bar',
    stacked=True,
    color=['blue', 'red'],
    edgecolor='black',
    ax=axes[2] # Plot on the third subplot
)

bars3 = ax3.patches
n3 = len(pivot_acct_type)
for i, bar in enumerate(bars3):
    hatch = hatches[i // n3]
    bar.set_hatch(hatch)

ax3.legend(title='Sentiment', labels=['Negative', 'Positive'], loc='best')
ax3.set_title('Breakdown of Sentiment by Topic Number and Account Type')
ax3.set_xlabel('Topic Number / Account Type')
ax3.set_ylabel('Number of Posts')

plt.tight_layout()
plt.show()