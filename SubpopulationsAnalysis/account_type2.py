
# Calculate sentiment statistics by account type and save to CSV. 
# 4 inputs df: 1. tweets with retweets, 2. tweets without retweets, 3. grouped by user with retweets, 4. grouped by user without retweets. 
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import csv
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
from pathlib import Path
if __name__ == "__main__":
    output_root = "output/account_type"
    
    #--------------- Uncomment one of the following lines to choose the input data file----------------------
    df = pd.read_pickle("../data/normalized_data.pkl") # all tweets and retweets
    df = df[df['Engagement Type'] != 'RETWEET'] # remove retweets
    #df = pd.read_pickle("../data/grouped2_data.pkl") # all tweets and retweets grouped by user
    user_lenght = len(df)
    print(f"Total users: {user_lenght:,}")
    #df= pd.read_pickle("../data/grouped2_data_removed_retweets.pkl") # all tweets without retweets grouped by user
    #------------------------------------------------------------------------------------------------------
    df = df.dropna(subset=['Account_Type'])
    account_type_stats = df.groupby('Account_Type')['mean_sentiment'].agg(['mean', 'std']).reset_index()
    account_type_stats.columns = ['Account_Type', 'Mean_Sentiment', 'Std_Sentiment']

    account_type_stats = account_type_stats.sort_values('Mean_Sentiment', ascending=False)

    output_file = os.path.join(output_root, 'account_type_sentiment_stats.csv')
    account_type_stats.to_csv(f"{output_root}/mean_results.csv", index=False, header=True)


    print(f"Account type sentiment statistics have been written to {output_file}")
    print("\nAccount Type Sentiment Statistics:")
    print(account_type_stats)

#______________________________________________________________________________________
    

