#
#
# run this once locally before deploying. It's a one-time setup script 
# that creates indexes on your Qdrant Cloud collection.
# Once the indexes exist in the cloud they persist forever, 
# so Docker/HuggingFace never needs to run it
#

from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType
from dotenv import load_dotenv
import os

load_dotenv()

client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    timeout=300
)

for field in ["Meter", "Premise", "POD"]:
    client.create_payload_index(
        collection_name="water_billing",
        field_name=field,
        field_schema=PayloadSchemaType.KEYWORD,
        wait=False
    )
    print(f"Indexed {field}")

print("Done — indexing running in background, wait ~1-2 min then restart the app.")