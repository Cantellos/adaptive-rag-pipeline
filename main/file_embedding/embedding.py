from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=".env")

# Calcola gli embeddings per ogni documento a partire dal testo estratto
def get_embedding(text):

    # Configura client per chiamata API di Azure OpenAI
    client = AzureOpenAI(
        azure_endpoint = os.getenv("EMBEDDING_URL_1"),
        api_key = os.getenv("OPENAI_API_KEY"),
        api_version = os.getenv("EMBEDDING_VERSION_1")
    )

    # Esegui chiamata embedding
    response = client.embeddings.create(
        input = text,
        model = os.getenv("EMBEDDING_MODEL_1")
    )

    # Estrai il vettore embedding
    embedding_vector = response.data[0].embedding

    return embedding_vector