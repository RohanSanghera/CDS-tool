from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.deepinfra import DeepInfraEmbeddingModel
from llama_index.llms.deepinfra import DeepInfraLLM
from llama_index.core import Settings  
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

class ModelConfig:
    """model configuration with DeepInfra embeddings"""
    _global_settings_initialised = False

    @classmethod
    def initialise_global_settings(cls):
        """Initialise global LlamaIndex settings"""
        if not cls._global_settings_initialised:  
            Settings.embed_model = cls.get_embedding_model("qwen")
            print("LlamaIndex set with Qwen embeddings")
            cls._global_settings_initialised = True
            return True
        else:
            print("Global settings set")
            return False

    @classmethod
    def get_embedding_model(cls, model_type="qwen"):
        if model_type == "qwen":
            return DeepInfraEmbeddingModel(
                model_id="Qwen/Qwen3-Embedding-8B",
                api_token=os.getenv("DEEPINFRA_API_KEY"),
                normalize=True  # Optional: normalize embeddings
            )
        elif model_type == "openai":
            return OpenAIEmbedding(
                model="text-embedding-3-small",
                api_key=os.getenv("OPENAI_API_KEY")
            )
    
    @classmethod
    def get_llm(cls, model_name):
            if model_name == "qwen3-30b":
                return DeepInfraLLM(
                    model="Qwen/Qwen3-30B-A3B",
                    api_key=os.getenv("DEEPINFRA_API_KEY"), 
                    temperature=0.1,  # 
                    max_tokens=3000, 
                    additional_kwargs={"top_p": 0.9}
                )
            elif model_name == "llama-3.3-70b":
                return DeepInfraLLM(
                    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    api_key=os.getenv("DEEPINFRA_API_KEY"),
                    temperature=0.1,  
                    max_tokens=2000,  
                )
            elif model_name == "llama-4-17b":
                return DeepInfraLLM(
                    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                    api_key=os.getenv("DEEPINFRA_API_KEY"),
                    temperature=0.1,  
                    max_tokens=2000,  
                )
            elif model_name == "gpt-4o-mini":
                return OpenAI(
                    model="gpt-4o-mini",
                    api_key=os.getenv("OPENAI_API_KEY")  
                )
            else:
                raise ValueError(f"Unknown LLM model: {model_name}")