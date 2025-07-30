from config import config
from langchain_community.llms import Ollama
import torch

class LLM:
    def __init__(self, model_name=config.MODEL_NAME):
        self.model = Ollama(model=model_name)

#llm = LLM()
