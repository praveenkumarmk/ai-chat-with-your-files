import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv()

PINE_CONE_API_KEY = os.getenv("PINE_CONE_API_KEY")
pc = Pinecone(PINE_CONE_API_KEY)  # reads PINECONE_API_KEY

if not pc.has_index("docs-index"):
    pc.create_index(
        name="docs-index",
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )