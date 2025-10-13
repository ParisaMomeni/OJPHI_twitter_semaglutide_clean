import pandas as pd
import numpy as np
import os

def analyze_author_activity(df):
    print("Available columns:")
    print(df.columns)

    output_root = "output/author_activity"
    os.makedirs(output_root, exist_ok=True)

    # Group by author and aggregate relevant metrics
    author_stats = df.groupby('Author').agg({
        'Twitter Followers': 'first',
        'Twitter Following': 'first',
        'Twitter Tweets': 'first',
        'Author': 'count',  # This counts the tweets per author
        'sentiment_label': 'mean'  # Calculate average sentiment for each author
    })

    author_stats.columns = ['Followers', 'Following', 'Total_Tweets', 'Dataset_Tweets', 'Avg_Sentiment']
    author_stats = author_stats.reset_index()

    # Calculate activity score
    author_stats['Activity_Score'] = (
        author_stats['Dataset_Tweets'] * 0.5 +
        np.log1p(author_stats['Followers']) * 0.2 +
        np.log1p(author_stats['Following']) * 0.1 +
        np.log1p(author_stats['Total_Tweets']) * 0.2
    )

    # Categorize authors based on activity score
    author_stats['Activity_Category'] = pd.qcut(
        author_stats['Activity_Score'], 
        q=3, 
        labels=['Least Active', 'Medium Active', 'Most Active']
    )

    # Save detailed author stats
    author_stats.to_csv(f"{output_root}/author_activity_stats.csv", index=False)

    # Create summary statistics
    summary_stats = author_stats.groupby('Activity_Category').agg({
        'Author': 'count',
        'Followers': 'mean',
        'Following': 'mean',
        'Total_Tweets': 'mean',
        'Dataset_Tweets': 'mean',
        'Activity_Score': 'mean',
        'Avg_Sentiment': 'mean'
    }).reset_index()

    summary_stats.columns = ['Activity_Category', 'Author_Count', 'Avg_Followers', 
                             'Avg_Following', 'Avg_Total_Tweets', 'Avg_Dataset_Tweets', 
                             'Avg_Activity_Score', 'Avg_Sentiment']

    summary_stats.to_csv(f"{output_root}/activity_summary_stats.csv", index=False)

    print("Analysis complete. Results saved in", output_root)

    return author_stats, summary_stats