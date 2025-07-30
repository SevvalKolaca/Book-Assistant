from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

config = Config()