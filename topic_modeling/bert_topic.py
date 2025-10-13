# !pip install umap
# !pip install hdbscan
# !pip install bertopic
# !pip install openai

import numpy as np
import pandas as pd
#from load_files import *
from clean_text import *
from tqdm.notebook import tqdm
tqdm.pandas()
import openai
from umap import umap_ as UMAP
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['figure.dpi'] = 500
rcParams['savefig.dpi'] = 500
rcParams['font.family'] = 'Serif'
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['legend.fontsize'] = 14
rcParams['figure.titlesize'] = 16

# non_df = df[~df['Snippet'].str.startswith('RT @')] # remove retweets
# original_df = df[df['Engagement Type'].isna() | (df['Engagement Type'] == '')] #only keep the original tweets

def process_tweet(tweet):
    """Process the tweet text to remove the RT @username: prefix."""
    if isinstance(tweet, str) and tweet.startswith('RT @'):
        parts = tweet.split(' ', 2)
        if len(parts) > 2:
            return parts[2].lstrip()
    return tweet

def filter_tweets(df, sentiment):
    """Filter the tweets by sentiment and keep original tweet content."""
    filter_df = df[df['roberta_sentiment'] == sentiment]
    filter_df['Snippet'] = filter_df['Snippet'].astype(str)
    filter_df['original_text'] = filter_df['Snippet'].apply(process_tweet)
    retweet_df = filter_df[filter_df['Engagement Type'] == 'RETWEET']
    retweet_df = retweet_df.drop_duplicates(subset='original_text', keep='first')
    nonretweet_df = filter_df[filter_df['Engagement Type'] != 'RETWEET']
    origin_df = pd.concat([retweet_df, nonretweet_df])
    origin_df = origin_df[~((origin_df.duplicated(subset='original_text', keep=False)) & (origin_df['Engagement Type'] == 'RETWEET'))] # drop duplicated retweets based on original tweets
    origin_df = origin_df.sort_index()
    origin_df = origin_df.reset_index(drop=True)
    origin_df = origin_df.dropna(subset=['original_text'])
    origin_df['clean_text'] = origin_df['original_text'].apply(clean_text)
    print(f"The number of all tweets: {len(df)}")
    print(f"The number of filtered tweets: {len(origin_df)}")
    return origin_df

df = pd.read_pickle('/content/drive/MyDrive/cscw_paper_twitter_semaglutide/Semaglutide_Twitter_roberta_sentiment.pkl')
pos_df = filter_tweets(df, 'positive')
neg_df = filter_tweets(df, 'negative')

pos_df.to_pickle('Semaglutide_Twitter_sentiment_postive.pkl')
neg_df.to_pickle('Semaglutide_Twitter_sentiment_negative.pkl')

# if no need to remove duplicated text after clean_text, then directly run: docs = pos_df['clean_text'].to_list()
pos_docs = pos_df['clean_text'].to_list() # change this to pos_df['original_text'] if no need to clean the text
#print(len(pos_docs))
pos_docs = list(set(pos_docs))
pos_docs = [doc for doc in pos_docs if str(doc).strip()]
#print(len(pos_docs))

neg_docs = neg_df['clean_text'].to_list() # change this to pos_df['original_text'] if no need to clean the text
#print(len(neg_docs))
neg_docs = list(set(neg_docs))
neg_docs = [doc for doc in neg_docs if str(doc).strip()]
#print(len(neg_docs))

### Topic Modeling - KMeans with BERT embedding, using the Elbow Method to determine optimal number of clusters ###
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
umap_model = UMAP.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine') #default n_neighbors=15
vectorizer_model = CountVectorizer()
cluster_model = KMeans(n_clusters=25)
ctfidf_model = ClassTfidfTransformer()
representation_model = KeyBERTInspired()

topic_model = BERTopic(
  embedding_model=embedding_model,          # Step 1 - Extract embeddings
  umap_model=umap_model,                    # Step 2 - Reduce dimensionality
  hdbscan_model=cluster_model,              # Step 3 - Cluster reduced embeddings
  vectorizer_model=vectorizer_model,        # Step 4 - Tokenize topics
  ctfidf_model=ctfidf_model,                # Step 5 - Extract topic words
  representation_model=representation_model # Step 6 - (Optional) Fine-tune topic represenations
)

# Extract the embeddings from documents - used for Elbow method
topics, _ = topic_model.fit_transform(pos_docs)
embeddings = topic_model._extract_embeddings(pos_docs, method="document")

# Reduce the dimensionality of the embeddings using UMAP method
reduced_embeddings = umap_model.fit_transform(embeddings)

# Plot the changes of square sum of distances based on number of clusters
ssd = []
K = range(2, 200)

for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(reduced_embeddings)
    ssd.append(kmeans.inertia_)

plt.figure(figsize=(9, 5))
plt.plot(K, ssd, 'bx-', linewidth=2, markersize=10, alpha=0.7)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Distances')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


### Positive Topics ###
topics, probs = topic_model.fit_transform(pos_docs)
representation = topic_model.get_topic_info()
representation.to_excel("pos_kmeans_bert_25_topic_representation.xlsx", index=False)
document_info = topic_model.get_document_info(pos_docs)
document_info.to_excel("pos_kmeans_bert_25_document_representation.xlsx", index=False)
#representation

# POSITIVE BARCHART
topic_model.visualize_barchart(top_n_topics=25)

### Negative Topics ###
topics, probs = topic_model.fit_transform(neg_docs)
representation = topic_model.get_topic_info()
representation.to_excel("neg_kmeans_bert_25_topic_representation.xlsx", index=False)
document_info = topic_model.get_document_info(neg_docs)
document_info.to_excel("neg_kmeans_bert_25_document_representation.xlsx", index=False)
#representation

# NEGATIVE BARCHART
topic_model.visualize_barchart(top_n_topics=25)