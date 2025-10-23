
📁 Data Folders:

    normalized_data/ – Tweet-level normalized data
         ↳ Retweets removed via .py scripts for per-tweet analysis 
         
    grouped2_data/ – Aggregated data grouped by user
    
    grouped2_data_removeRetweets/ – Same as above, but with retweets excluded
__________________________________________________________________________________________________________


📌 Preprocess Code Files:

    1. preprocess_data.py --> for preprocessing the data
    
    2. normalize_per_tweets.py --> normalize per tweets to make sure columns are same for different analysis
    
    3. aggregate_by_author.py --> for main analysis group by user
    
    4. aggregate_by_author_removeRetweets.py --> for sentivity analysis
__________________________________________________________________________________________________________


📊 Analysis & Figures

    5. numberOfSubpopu.py ---> Appendix --> table s1
    
    6. BertVersion.py --> BERT version per reviewer comment
    
    7. interest_accountV22.py --> sentiment and account analysis ---> heatmap ---> Figure 5
    
    8. bimonthly_visualization_analysis.py --> for main analysis---> Figure 2
    
    9. bimonthly_visualization_analysisـretweetExclusion.py --> for sentivity analysis ---> Appendix Figure S3
    
    10. 95% confidence interval.py --> for confidence interval calculation ---> Table 2
    
    11. ITS2.py --> for ITS analysis  ---> Table 1,  Figure  
    ____________________________________________________________________________________________________________
    

👥 Subpopulation Analysis --->folder SubpopulationAnalysis: 

    gender2.py, verified_users2.py, account_type2.py, regional2.py,  verified_users2.py --> for subpopulation analysis---> Figure 4 and Appendix Table S1
    
__________________________________________________________________________________________________________

📂 Output Folders:

    output --> for main analysis
    
    output--> inside SubpopulationAnalysis folder --> for attribute analysis



