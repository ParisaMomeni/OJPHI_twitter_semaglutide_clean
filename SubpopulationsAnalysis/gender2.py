#1.  CALCULATE OVERALL SENTIMENT 2. gender DESCRIPTIVE STATISTICS
#df is input data frame. we might have 4 df: 1.per tweet , 2.per user 3.per tweet without retweets 4.per user without retweets. 
import pandas as pd
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import csv
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
from pathlib import Path

if __name__ == "__main__":
    output_dir = "output/gender/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #--------------------------
    #Each time we uncomment the relevant df loading line:
    #--------------------------
    df = pd.read_pickle("../data/normalized_data.pkl")
    total_posts = len(df)
    df = df[df['Engagement Type'] != 'RETWEET']
    retweet_exxluded_posts = len(df)
    print(f"Total posts before exclusion: {total_posts:,}")
    print(f"Total posts after retweet exclusion: {retweet_exxluded_posts:,}")
    #df = pd.read_pickle("../data/grouped2_data.pkl")
    #df= pd.read_pickle("../data/grouped2_data_removed_retweets.pkl")
    #--------------------------  
    df = df.dropna(subset=['mean_sentiment'])

# --- CALCULATE OVERALL SENTIMENT ---
    overall_stats = df['mean_sentiment'].agg(['mean', 'std', 'count']).rename({
        'mean': 'Mean Sentiment',
        'std': 'Standard Deviation',
        'count': 'Number of Tweets'
    })
    overall_stats_df = overall_stats.to_frame().reset_index()
    overall_stats_df.columns = ['Metric', 'Value']
# --- SAVE RESULTS ---
    overall_stats_df.to_csv(f"output/overalscore/overall_sentiment_summary.csv", index=False)
    print("\n--- Overall Sentiment Summary ---")
    print(overall_stats_df)
    
#---------------------------

    df = df.dropna(subset=['Gender', 'mean_sentiment'])
    overall_stats = df['mean_sentiment'].agg([ 'mean', 'std', 'count'])
    gender_stats = df.groupby('Gender')['mean_sentiment'].agg([ 'mean', 'std', 'count'])
    print(f"Overall stats: {overall_stats}")
    print(gender_stats.head()) 
    gender_stats = gender_stats.reset_index()  # This brings 'Gender' back as a column
    print(gender_stats.head()) 
    gender_stats.fillna(0, inplace=True)
    gender_stats.to_csv(f"{output_dir}gender_descriptive_stats.csv", index=False, header=True)

    # Pairwise t-tests
    male_sentiments = df[df['Gender'] == 'male']['mean_sentiment']
    female_sentiments = df[df['Gender'] == 'female']['mean_sentiment']
    unknown_sentiments = df[df['Gender'] == 'unknown']['mean_sentiment']

    t_stat_mf, p_value_mf = ttest_ind(male_sentiments, female_sentiments, equal_var=False)
    t_stat_mu, p_value_mu = ttest_ind(male_sentiments, unknown_sentiments, equal_var=False)
    t_stat_fu, p_value_fu = ttest_ind(female_sentiments, unknown_sentiments, equal_var=False)

    # Write t-test results to CSV
    with open(f"{output_dir}gender_ttest_results.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['genderGroup1', 'genderGroup2', 't-statistic', 'p-value'])
        writer.writerow(['Men', 'Women', t_stat_mf, p_value_mf])
        writer.writerow(['Men', 'Unknown', t_stat_mu, p_value_mu])
        writer.writerow(['Women', 'Unknown', t_stat_fu, p_value_fu])

    # One-Way ANOVA
    model = ols('mean_sentiment ~ C(Gender)', data=df).fit()
    anova_table = anova_lm(model, typ=2)

    # Save ANOVA results to CSV
    anova_table.fillna(0, inplace=True)
    anova_table.to_csv(f"{output_dir}gender_anova_results.csv", index=False, header=True)

    # Post-hoc Test: Tukey's HSD
    tukey = pairwise_tukeyhsd(df['mean_sentiment'], df['Gender'])

    # Convert Tukey results to a DataFrame and save to CSV
    tukey_results = pd.DataFrame(
        data=tukey.summary().data[1:],  # Exclude header
        columns=tukey.summary().data[0]  # Use header from the Tukey summary
    )
    tukey_results.fillna(0, inplace=True)
    tukey_results.to_csv(f"{output_dir}gender_tukey_results.csv", index=False, header=True)

#_____________________________________________________________________________________________
