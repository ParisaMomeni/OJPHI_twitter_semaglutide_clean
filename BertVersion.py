from huggingface_hub import HfApi

api = HfApi()
info = api.model_info("cardiffnlp/twitter-roberta-base-sentiment-latest")
print("Model commit SHA:", info.sha) #REVISION = "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"



MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
REVISION = "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"  #  print("Model commit SHA:", info.sha)

api = HfApi()
info = api.model_info(MODEL_ID, revision=REVISION)

print("=== Model Version Information ===")
print(f"Model ID: {MODEL_ID}")
print(f"Commit SHA: {info.sha}")
print(f"Last modified: {info.lastModified}")
print(f"Private?: {info.private}")
print(f"Downloads: {info.downloads}")

print("\n=== Config Details ===")
if info.cardData:
    card_dict = info.cardData.__dict__  # convert ModelCardData object to dict
    for k, v in card_dict.items():
        print(f"{k}: {v}")
else:
    print("No card data available.")

print("\n=== Files in Snapshot ===")
for f in info.siblings[:10]:  # show only first 10 files
    print(f"- {f.rfilename}")
