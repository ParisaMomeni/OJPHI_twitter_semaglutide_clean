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
    return df

if __name__ == "__main__":
    df = pd.read_pickle("data/Semaglutide_Twitter_20210601_20240331.pkl")
    print(f"Total number of rows in the original dataset: {len(df)}")

    
    # Preprocess the data
    #processed_df contains the processed version of the original data.
    processed_df = preprocess_data(df, len(df))

    processed_df.to_pickle("data/processed_data.pkl")
