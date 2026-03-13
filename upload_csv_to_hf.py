# from huggingface_hub import HfApi
# api = HfApi()
# api.create_repo(repo_id="ns00/water-billing-data", repo_type="dataset", private=True)
# api.upload_file(
#     path_or_fileobj="output/verified_water.csv",
#     path_in_repo="verified_water.csv",
#     repo_id="ns00/water-billing-data",
#     repo_type="dataset",
# )

from huggingface_hub import HfApi
import os
from dotenv import load_dotenv
load_dotenv()

api = HfApi(token=os.environ["HF_TOKEN"])

# Create private dataset repo
api.create_repo(
    repo_id="ns00/water-billing-data",
    repo_type="dataset",
    private=True
)

# Upload CSV
api.upload_file(
    path_or_fileobj="output/verified_water.csv",
    path_in_repo="verified_water.csv",
    repo_id="ns00/water-billing-data",
    repo_type="dataset",
)
print("Done!")