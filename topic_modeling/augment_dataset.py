from clean_text import clean_text
import pandas as pd
 
def generate_docs_files():
    df = pd.read_pickle("data/Semaglutide_Twitter_roberta_sentiment.pkl")
    posDocs = pd.read_csv("topic_modeling/pos_kmeans_bert_100_document_representation.csv")
    posDocs = posDocs[['Document', 'Topic']]
    negDocs = pd.read_csv("topic_modeling/neg_kmeans_bert_100_document_representation.csv")
    negDocs = negDocs[['Document', 'Topic']]

    umbrellaTopics = pd.read_csv("topic_modeling/topic_annotation_gabriel.csv")
    unbrellaTopicsList = umbrellaTopics['Umbrella topic mapping'].tolist()

    def mapTopic(topic):
        return unbrellaTopicsList[topic]
    
    posDocs["Topic_mapping"] = posDocs['Topic'].apply(mapTopic)
    posDocs = posDocs[['Document', 'Topic_mapping']]
    print(posDocs.head(50))
    posDocs.to_pickle("data/pos_docs.pkl")

    negDocs['Topic'] = negDocs['Topic'] + 100
    negDocs["Topic_mapping"] = negDocs['Topic'].apply(mapTopic)
    negDocs = negDocs[['Document', 'Topic_mapping']]
    print(negDocs.head(50))
    negDocs.to_pickle("data/neg_docs.pkl")

def add_snippet_cleaned_to_data():
    df = pd.read_pickle("data/Semaglutide_Twitter_roberta_sentiment.pkl")
    df["snippet_cleaned"] = df['Snippet'].apply(clean_text.clean_text)
    print(df.head(20))
    df.to_pickle("data/Semaglutide_Twitter_cleaned_snippet.pkl")

def merge_neg_pos_docs():
    df_neg = pd.read_pickle("data/neg_docs.pkl")
    df_pos = pd.read_pickle("data/pos_docs.pkl")
    df_merged = pd.concat([df_neg,df_pos])
    print(df_merged.head(10))
    df_merged.to_pickle("data/merged_docs")

def merge_topic_numbers_into_data():
    df_data = pd.read_pickle("data/Semaglutide_Twitter_cleaned_snippet.pkl")
    df_docs = pd.read_pickle("data/merged_docs.pkl")
    df_docs = df_docs.rename(columns={"Document": "snippet_cleaned"}) # rename "Document" to "snippet_cleaned"
    df_merged = pd.merge(df_data, df_docs, on="snippet_cleaned", how="left")
    print(df_merged.head(50))
    df_merged.to_csv("data/Semaglutide_Twitter_Topic_mapping.csv")
    df_merged.to_pickle("data/Semaglutide_Twitter_Topic_mapping.pkl")