
    ```
data folders:
    normalized_data --> normalized data per tweets ---> I remove retweets in .py files per tweets analysises
    grouped2_data --> aggregated data per user
    gruoped2_data_removeRetweets --> aggregated data per user after removing retweets

code files:
    1. preprocess_data.py --> for preprocessing the data
    2. normalize_per_tweets.py --> normalize per tweets to make sure columns are same for different analysis
    3. aggregate_by_author.py --> for main analysis group by user
    4. aggregate_by_author_removeRetweets.py --> for sentivity analysis
    5. numberOfSubpopu.py ---> table 1
    6. BertVersion.py --> BERT version per reviewer comment
    7. sentiment_analysus_interest_accountV22.py --> sentiment and account analysis ---> heatmap
    8. bimonthly_visualization_analysis.py --> for main analysis
    9. bimonthly_visualization_analysisـretweetExclusion.py --> for sentivity analysis
    10. folder analysis: gender2.py, verified_users2.py, account_type2.py, regional2.py, 
    11. verified_users2.py --> for main analysis
    12. 95% confidence interval.py --> for confidence interval calculation
    13. ITS.py --> for ITS analysis


output folders:
    output --> for main analysis
    output--> inside analysis folder --> for attribute analysis
