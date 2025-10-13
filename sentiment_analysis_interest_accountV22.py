import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from matplotlib import rcParams

processed_df = pd.read_pickle('data/normalized_data.pkl')
grouped_df = pd.read_pickle('data/grouped2_data.pkl')
df = processed_df.copy()

#_______________________________________________________________________________________________________________________
#  global font properties
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['legend.title_fontsize'] = 14
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

#_______________________________________________________________________________________________________________________
# Load pre-computed interest popularity
pop_df_user = pd.read_csv("output/interests/interests_User_stats2.csv")
pop_df_user = pop_df_user.rename(columns={"interest": "Interest", "count": "Count"})
pop_df_user = pop_df_user.sort_values(by="Count", ascending=False)

pop_df_tweet = pd.read_csv("output/interests/interests_tweets_stats2.csv")
pop_df_tweet = pop_df_tweet.rename(columns={"interest": "Interest", "count": "Count_tweet"})

#_______________________________________________________________________________________________________________________
df['Account_Type'] = df['Account_Type'].fillna('Unknown') 
grouped_account = df.groupby(['Account_Type'])['mean_sentiment'].mean().reset_index()

# Top N interests
TOP_N = 21
top_user = pop_df_user.head(TOP_N)["Interest"]
top_tweet = pop_df_tweet.head(TOP_N)["Interest"]

# Intersection of user & tweet interests
top_interests = pd.Index(sorted(set(top_user).intersection(set(top_tweet))))

# Calculate mean sentiments and differences
sentiment_means = df[df['Interest'].isin(top_interests)].groupby(
    ['Interest', 'Account_Type']
)['mean_sentiment'].mean().unstack()

if 'individual' in sentiment_means.columns and 'organizational' in sentiment_means.columns:
    sentiment_diff = abs(sentiment_means['individual'] - sentiment_means['organisational'])
    max_diff_interest = sentiment_diff.idxmax()
else:
    max_diff_interest = None

#_______________________________________________________________________________________________________________________
# Encode categorical features for regression
df_ml = df[['mean_sentiment', 'Interest', 'Account_Type']].dropna()
df_ml['Interest'] = LabelEncoder().fit_transform(df_ml['Interest'])
df_ml['Account_Type'] = LabelEncoder().fit_transform(df_ml['Account_Type'])
X = df_ml[['Interest', 'Account_Type']]
y = df_ml['mean_sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(f"R^2: {r2_score(y_test, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

#_______________________________________________________________________________________________________________________
# Heatmaps

#_______________________________________________________________________________________________________________________
# Heatmaps

# Tweet heatmap
tweet_heatmap = df[df['Interest'].isin(top_interests)].pivot_table(
    index='Interest',
    columns='Account_Type',
    values='mean_sentiment',
    aggfunc='mean'
)


tweet_heatmap = tweet_heatmap.rename(columns={"organisational": "Organizational", 
                                              "individual": "Individual"})

tweet_heatmap = tweet_heatmap.rename(columns=lambda x: f"{x} (Tweet)")


# User heatmap
user_heatmap = grouped_df[grouped_df['Interest'].isin(top_interests)].pivot_table(
    index='Interest',
    columns='Account_Type',
    values='mean_sentiment',
    aggfunc='mean'
)


user_heatmap = user_heatmap.rename(columns={"organisational": "Organizational", 
                                            "individual": "Individual"})

user_heatmap = user_heatmap.rename(columns=lambda x: f"{x} (User)")


# Combine
combined_heatmap = pd.concat([tweet_heatmap, user_heatmap], axis=1)
combined_heatmap = combined_heatmap.reindex(index=top_interests)

desired = ["Individual (Tweet)", "Individual (User)", 
           "Organizational (Tweet)", "Organizational (User)"]
present = [c for c in desired if c in combined_heatmap.columns]
others  = [c for c in combined_heatmap.columns if c not in present]  # e.g., Unknown(...)
combined_heatmap = combined_heatmap[present + others]


# Plot
plt.figure(figsize=(12, 10))

ax = sns.heatmap(
    combined_heatmap, annot=True, cmap="coolwarm", fmt=".2f",
    linewidths=0.5, linecolor='gray'
)

# Add thicker line between Individual and Organizational
# This is after the two "Individual" columns
split_pos = 2   # Individual (Tweet), Individual (User)

ax.axvline(x=split_pos, color='black', linewidth=3)

#sns.heatmap(combined_heatmap, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, linecolor='gray')
plt.xlabel("Account Type & Perspective", fontsize=14, fontweight='bold')
plt.ylabel("Interest", fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('output/interest_account/Combined_Sentiment_Heatmap_Tweet_User.png', dpi=300, bbox_inches='tight')
plt.show()
