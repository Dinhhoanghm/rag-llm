import os

from llama_index.core import Settings

from rag_chatbot.core import LocalEmbedding
from rag_chatbot.eval import QAGenerator
from rag_chatbot.setting import RAGSettings

Settings.embed_model = LocalEmbedding.set()

settings = RAGSettings()
settings.ingestion.embed_llm = "BAAI/bge-large-en-v1.5"

# Create QA Generator with the local embedding model
generator = QAGenerator(
    embed_model="BAAI/bge-large-en-v1.5",
    llm="qwen/qwen3-32b",
)

# Generate QA pairs and docstore
generator.generate(
    input_files=[
        os.path.join(os.getcwd(), "data", "data", x) for x in os.listdir("data/data")
    ],
    output_dir="val_dataset",  # Specify output directory for docstore.json
)
