#grouped_df is separate from the original df. It doesn't modify df, but instead creates a new, summarized DataFrame grouped by author.
# extract the # of rows
#current data in grouped_data.pkl: sentiment_label but it should be mean_sentiment in new run of preprocess_data.py.
# remove group_by_author from here since we have aggregate_by_author.py
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import sys
import argparse
from tqdm import tqdm

##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------

def calculate_sentiment_scores(df, model, tokenizer):
    positive_scores = []
    neutral_scores = []
    negative_scores = []  
    for row in tqdm(range(len(df)), desc="Calculating sentiment scores"):
        text = df["Snippet"].iloc[row]
        encoded_input = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)  # [negative, neutral, positive]
        negative_scores.append(scores[0])
        neutral_scores.append(scores[1])
        positive_scores.append(scores[2])
    df['positive_score'] = positive_scores
    df['neutral_score'] = neutral_scores
    df['negative_score'] = negative_scores
    return df

##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------

def assign_sentiment_labels(df):
        # Assign labels based on the highest score
    conditions = [
        (df['positive_score'] > df['neutral_score']) & (df['positive_score'] > df['negative_score']),
        (df['neutral_score'] > df['positive_score']) & (df['neutral_score'] > df['negative_score']),
        (df['negative_score'] > df['positive_score']) & (df['negative_score'] > df['neutral_score'])
        ]
    choices = [1, 0, -1]
    df['sentiment_label'] = np.select(conditions, choices, default=0)
    return df

##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------


def group_by_author(df):
    def most_common(x):
        value_counts = x.value_counts()
        return value_counts.index[0] if not value_counts.empty else None
#creates a new DataFrame grouped_df; by grouping the original DataFrame df by the 'Author' column.
#df.groupby('Author'):  groups all rows in the df by the 'Author' col. This means all rows with the same author will be considered together.
#.agg({...}):   aggregation functions to specific columns within each group.
    '''grouped_df = df.groupby('Author').agg({
        'sentiment_label': 'mean',
        'Gender': 'first',
        'Country': 'first',
        'Region': 'first',
        'Engagement Type': most_common,
        'Interest': most_common, 
         
    }).reset_index()'''
    
    
    grouped_df = df.groupby('Author').agg(
        mean_sentiment=('sentiment_label', 'mean'),     # continuous [-1,1]
        Gender=('Gender', most_common),
        Country=('Country', most_common),
        Region=('Region', most_common),
        Engagement_Type=('Engagement Type', most_common),
        Interest=('Interest', most_common),
        Account_Type=('Account type', most_common),     # NEW
        Verified=('Verified', most_common),             # NEW
    ).reset_index()

    return grouped_df

##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------
##------------------------------------------------------------------------------------------------
def preprocess_data(df, rows_to_process=None):
    if rows_to_process:
        df = df.head(rows_to_process)
    print(f"Processing {len(df)} rows")
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    print(f"Using model: {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    df = calculate_sentiment_scores(df, model, tokenizer)
    df = assign_sentiment_labels(df)
    grouped_by_author_df = group_by_author(df)
    return df, grouped_by_author_df

if __name__ == "__main__":
    # Load data here
    df = pd.read_pickle("data/Semaglutide_Twitter_20210601_20240331.pkl")
    print(f"Total number of rows in the original dataset: {len(df)}")

    #whole rows in orginal DF = 859751
    #tweetsLimit = 100
    #tweetsLimit = min(tweetsLimit, len(df.index))
    #df = df.head(tweetsLimit)
    # Preprocess the data
    #processed_df contains the processed version of the original data.
    #grouped_df is the grouped-by-author version we discussed earlier.
    processed_df, grouped_df = preprocess_data(df, len(df))
    
    # Save the processed data
    processed_df.to_pickle("data/processed_data.pkl")
    grouped_df.to_pickle("data/grouped_data.pkl")