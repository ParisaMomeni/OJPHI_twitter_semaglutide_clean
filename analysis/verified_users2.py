# 1.  VERIFIED USER DESCRIPTIVE STATISTICS
# df is input data frame. we might have 4: 1.per tweet , 2.per user 3.per tweet without retweets 4.per user without retweets. Each time we uncomment the relevant df loading line:
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
import os
if __name__ == "__main__":
    #df = pd.read_pickle("../data/normalized_data.pkl") # all tweets and retweets
    #df = df[df['Engagement Type'] != 'RETWEET'] # remove retweets
    df = pd.read_pickle("../data/grouped2_data.pkl") # all tweets and retweets grouped by author
    #df= pd.read_pickle("../data/grouped2_data_removed_retweets.pkl") # all tweets without retweets grouped by author
    output_root = "output/verified"  
    
    # Separate verified and non-verified users
    verified = df[df['Verified'] == True]['mean_sentiment']
    not_verified = df[df['Verified'] == False]['mean_sentiment']
    unknown = df[df['Verified'].isna()]['mean_sentiment']

#mean of verified and not verified
    verified_mean = verified.mean()
    not_verified_mean = not_verified.mean()
    unknown_mean = unknown.mean()
    verified_median = verified.median()
    not_verified_median = not_verified.median()
    verified_std = verified.std()
    not_verified_std = not_verified.std()
    verified_count = verified.count()
    not_verified_count = not_verified.count()
    
    all_stats = pd.DataFrame({
        'User Type': ['Verified', 'Not Verified', 'Unknown'],
        'Mean': [verified_mean, not_verified_mean, unknown_mean],
        #'Median': [verified_median, not_verified_median],
        #'Std Dev': [verified_std, not_verified_std],
        #'Count': [verified_count, not_verified_count]
    })

    all_stats.to_csv(f"{output_root}/verified_user_stats.csv", index=False, header=True)

    print(f"Verified users analysis complete. Results saved in {output_root}")

