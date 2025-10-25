import pandas as pd
import numpy as np
from scipy.stats import norm, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
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
plt.rcParams['figure.titleweight'] = 'bold'

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

df_per_post = pd.read_pickle('/content/drive/MyDrive/cscw_paper_twitter_semaglutide/normalized_data.pkl')
df_per_user = pd.read_pickle('/content/drive/MyDrive/cscw_paper_twitter_semaglutide/grouped2_data.pkl')

df_per_post['Account_Type'] = df_per_post['Account_Type'].replace('organisational', 'organizational')
df_per_user['Account_Type'] = df_per_user['Account_Type'].replace('organisational', 'organizational')


df_per_post['state_to_region'] = df_per_post['Region'].map(state_to_region)
df_per_post['state_to_region'] = df_per_post['state_to_region'].fillna('Unknown')
df_per_post = df_per_post[df_per_post['Region'] != 'out of usa']
df_per_user['state_to_region'] = df_per_user['Region'].map(state_to_region)
df_per_user['state_to_region'] = df_per_user['state_to_region'].fillna('Unknown')
df_per_user = df_per_user[df_per_user['Region'] != 'out of usa']


fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 15), sharey=True)

colors = ['plum', 'g', 'orange', 'b', 'r', 'skyblue']

gender_plots_post = df_per_post.boxplot(column='mean_sentiment', by='Gender', showmeans=True, meanline=True, ax=axes[0, 0], widths=0.6, patch_artist=True)
axes[0, 0].set_title("Gender (Per Post)")
axes[0, 0].set_xlabel("")
axes[0, 0].set_ylabel("Mean Sentiment")
for i, box_artist in enumerate(gender_plots_post.patches):
    box_artist.set_facecolor(colors[i % len(colors)])
    box_artist.set_edgecolor('black')
    box_artist.set_alpha(0.8)

gender_plots_user = df_per_user.boxplot(column='mean_sentiment', by='Gender', showmeans=True, meanline=True, ax=axes[0, 1], widths=0.6, patch_artist=True)
axes[0, 1].set_title("Gender (Per User)")
axes[0, 1].set_xlabel("")
axes[0, 1].set_ylabel("")
for i, box_artist in enumerate(gender_plots_user.patches):
    box_artist.set_facecolor(colors[i % len(colors)])
    box_artist.set_edgecolor('black')
    box_artist.set_alpha(0.8)

verified_plots_post = df_per_post.boxplot(column='mean_sentiment', by='Verified', showmeans=True, meanline=True, ax=axes[1, 0], widths=0.6, patch_artist=True)
axes[1, 0].set_title("Verified (Per Post)")
axes[1, 0].set_xlabel("Twitter Verified")
axes[1, 0].set_ylabel("Mean Sentiment")
for i, box_artist in enumerate(verified_plots_post.patches):
    box_artist.set_facecolor(colors[i % len(colors)])
    box_artist.set_edgecolor('black')
    box_artist.set_alpha(0.8)

verified_plots_user = df_per_user.boxplot(column='mean_sentiment', by='Verified', showmeans=True, meanline=True, ax=axes[1, 1], widths=0.6, patch_artist=True)
axes[1, 1].set_title("Verified (Per User)")
axes[1, 1].set_xlabel("Twitter Verified")
axes[1, 1].set_ylabel("")
for i, box_artist in enumerate(verified_plots_user.patches):
    box_artist.set_facecolor(colors[i % len(colors)])
    box_artist.set_edgecolor('black')
    box_artist.set_alpha(0.8)


acct_type_plots_post = df_per_post.boxplot(column='mean_sentiment', by='Account_Type', showmeans=True, meanline=True, ax=axes[2, 0], widths=0.6, patch_artist=True)
axes[2, 0].set_title("Account Type (Per Post)")
axes[2, 0].set_xlabel("Account Type")
axes[2, 0].set_ylabel("Mean Sentiment")
for i, box_artist in enumerate(acct_type_plots_post.patches):
    box_artist.set_facecolor(colors[i % len(colors)])
    box_artist.set_edgecolor('black')
    box_artist.set_alpha(0.8)

acct_type_plots_user = df_per_user.boxplot(column='mean_sentiment', by='Account_Type', showmeans=True, meanline=True, ax=axes[2, 1], widths=0.6, patch_artist=True)
axes[2, 1].set_title("Account Type (Per User)")
axes[2, 1].set_xlabel("Account Type")
axes[2, 1].set_ylabel("")
for i, box_artist in enumerate(acct_type_plots_user.patches):
    box_artist.set_facecolor(colors[i % len(colors)])
    box_artist.set_edgecolor('black')
    box_artist.set_alpha(0.8)


region_plots_post = df_per_post.boxplot(column='mean_sentiment', by='Region', showmeans=True, meanline=True, ax=axes[3, 0], widths=0.6, patch_artist=True)
axes[3, 0].set_title("Region (Per Post)")
axes[3, 0].set_xlabel("Region")
axes[3, 0].set_ylabel("Mean Sentiment")
for i, box_artist in enumerate(region_plots_post.patches):
    box_artist.set_facecolor(colors[i % len(colors)])
    box_artist.set_edgecolor('black')
    box_artist.set_alpha(0.8)

region_plots_user = df_per_user.boxplot(column='mean_sentiment', by='Region', showmeans=True, meanline=True, ax=axes[3, 1], widths=0.6, patch_artist=True)
axes[3, 1].set_title("Region (Per User)")
axes[3, 1].set_xlabel("Region")
axes[3, 1].set_ylabel("")
for i, box_artist in enumerate(region_plots_user.patches):
    box_artist.set_facecolor(colors[i % len(colors)])
    box_artist.set_edgecolor('black')
    box_artist.set_alpha(0.8)


for row in axes:
    for ax in row:
        for line in ax.lines:
            line.set_linewidth(2)
            line.set_color('black')

fig.suptitle("")
plt.tight_layout()
plt.show()