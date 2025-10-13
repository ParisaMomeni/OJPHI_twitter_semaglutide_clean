import pandas as pd
import numpy as np

if __name__ == "__main__":
    """
    Analyze user interests at the author level.
    - Each author may list zero, one, or multiple interests (comma-separated).
    - Uses continuous mean_sentiment in [-1, 1].
    - Produces per-interest statistics: count, mean, std.
    """
    #df
    df = pd.read_pickle("../data/normalized_data.pkl")
    interestsOutputRoot = "../output/interests/interests_tweets"

    sentimentByInterest = {}

    for _, row in df.iterrows():
        if pd.isna(row["Interest"]) or row["Interest"].strip() == "":
            continue  # skip authors with no interests

        sentiment_value = float(row["mean_sentiment"])  # continuous value
        for interest in row["Interest"].split(","):
            interest = interest.strip()
            if not interest:
                continue

            if interest not in sentimentByInterest:
                sentimentByInterest[interest] = []

            sentimentByInterest[interest].append(sentiment_value)

    # Calculate statistics for each interest
    interest_stats = {}
    for interest, values in sentimentByInterest.items():
        values = np.array(values)
        interest_stats[interest] = {
            "count": len(values),
            "mean": values.mean(),
            "std": values.std(ddof=1) if len(values) > 1 else 0.0,
        }

    interest_stats_df = pd.DataFrame.from_dict(interest_stats, orient="index")
    interest_stats_df.index.name = "interest"
    interest_stats_df = interest_stats_df.sort_values("count", ascending=False)

    interest_stats_df.to_csv(f"{interestsOutputRoot}_stats2.csv")

    print(f"Saved per-interest stats to {interestsOutputRoot}_stats2.csv")
