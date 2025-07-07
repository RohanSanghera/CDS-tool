"""
Knowledge_layer.py Knowledge layer for loading, indexing, and querying the NICE CKS and local guidelines.
"""

from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.readers.json import JSONReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor, SummaryExtractor
from llama_index.core.schema import MetadataMode
from llama_index.core.node_parser import HierarchicalNodeParser, SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from config import ModelConfig
import qdrant_client
import json
import os


class CKSProcessor:
    """
    Processes a dictionary loaded from the CKS JSON file. Flatens into a list of docs
    """
    def process_json_data(self, json_data):

        processed_docs = []
        print(f"Processing {len(json_data)} conditions from CKS JSON")
        
        for condition_name, condition_data in json_data.items():
            if not isinstance(condition_data, dict):
                continue 

            for section_name, content_text in condition_data.items():
                

                if content_text and isinstance(content_text, str) and content_text.strip():
                    
                    metadata = {
                        "condition": condition_name,
                        "section": section_name,
                        "source": "NICE CKS",
                        "guideline_type": "UK-national",
                    }
                    

                    doc = Document(text=content_text, metadata=metadata)
                    processed_docs.append(doc)

        print(f"Flattened CKS json into {len(processed_docs)} docs")
        return processed_docs
    

class KnowledgeLayer:
    """
    Ingestion pipeline for NICE CKS and local guidelines
    """
    def __init__(self, use_cache=True, cache_dir="./guideline_cache"):
        self.embed_model = ModelConfig.get_embedding_model("qwen")
        self.llm = ModelConfig.get_llm("qwen3-30b")
        self.cks_processor = CKSProcessor()
        self.use_cache = use_cache
        self.cache = IngestionCache(cache_dir=cache_dir, collection = "guidelines") if use_cache else None

        self.cks_index = None
        self.local_index = None

        qdrant_url = os.getenv('QDRANT_URL')
        qdrant_api_key = os.getenv('QDRANT_API_KEY')

        if qdrant_url:
            print(f"Using Qdrant cloud")
            client = qdrant_client.QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key
            )
        else:
            print("Using file-based Qdrant ")
            client = qdrant_client.QdrantClient(path="./vector_db")

        self.cks_vector_store = QdrantVectorStore(client=client, collection_name="cks_guideline")
        self.local_vector_store = QdrantVectorStore(client=client, collection_name="local_guideline")

    def vector_store_exists(self):
        """Check if vector store existsalready"""
        return os.path.exists("./vector_db") and len(os.listdir("./vector_db")) > 0

    def setup_pipelines(self):
        """setup ingestion pipelines"""

        #extractors = [
        #    HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 256], chunk_overlap=100),
        #    TitleExtractor(llm=self.llm, nodes=3),
        #    QuestionsAnsweredExtractor(
        #        questions=3, 
        #        llm=self.llm, 
        #         metadata_mode=MetadataMode.EMBED
        #    ),
        #    self.embed_model
        #] 
        #Simplified extractor
        extractors = [
            SentenceSplitter(chunk_size=512, chunk_overlap=50),
            #SummaryExtractor(llm=self.llm,summaries=['self']),
            self.embed_model
        ] 

        self.cks_pipeline = IngestionPipeline(
            transformations=extractors,
            vector_store=self.cks_vector_store,
            cache=self.cache if self.use_cache else None
            #show_progress=True
        )
        #local pipeline
        self.local_pipeline = IngestionPipeline(
            transformations=extractors,
            vector_store=self.local_vector_store,
            cache=self.cache if self.use_cache else None
            #show_progress=True
        )


    def data_loader(self, cks_path="cks_conditions_data.json", local_dir="paed-guidelines"):
        """Load cks and local guidelines"""
        if 1==0:
        #if self.vector_store_exists():
            print("Found existing vector store, loading...")
            self.cks_index = VectorStoreIndex.from_vector_store(
                self.cks_vector_store,
                embed_model=self.embed_model 
            )
            self.local_index = VectorStoreIndex.from_vector_store(
                self.local_vector_store,
                embed_model=self.embed_model 
            )
            print("✓ Vector store loaded successfully!")
            return self.cks_index, self.local_index
        else:

            try:
                print('Loading and processing CKS guidelines...')
                # Load the JSON file into a Python dictionary
                with open(cks_path, 'r', encoding='utf-8') as f:
                    cks_json_data = json.load(f)
                
                # Pass the dictionary directly to your new processor method
                processed_cks_docs = self.cks_processor.process_json_data(cks_json_data)
            except Exception as e:
                print(f"Error processing CKS JSON: {e}")
                processed_cks_docs = []
            
            try:
                print('Loading and processing local guidelines...')
                local_documents = SimpleDirectoryReader(input_dir=local_dir).load_data()
                for doc in local_documents:
                    doc.metadata.update({
                        'source': 'institutional',
                        'guideline_type': 'local-ouh',
                        'filename': doc.metadata.get('file_name', 'unknown.pdf')
                    })
                processed_local_docs = local_documents
                print(f'Loaded and processed {len(processed_local_docs)} local documents')
            except Exception as e:
                print(f"Error processing local guidelines: {e}")
                processed_local_docs = []


            self.setup_pipelines()
            
            print('Running CKS through ingestion pipeline...')
            cks_nodes = self.cks_pipeline.run(documents=processed_cks_docs, num_workers=1, show_progress=True)
            print(f'Loaded {len(cks_nodes)} CKS nodes')

            print('Running local guidelines through ingestion pipeline...')
            local_nodes = self.local_pipeline.run(documents=processed_local_docs, num_workers=1, show_progress=True)
            print(f'Loaded {len(local_nodes)} local nodes')

            self.cks_index = VectorStoreIndex.from_vector_store(self.cks_vector_store)
            self.local_index = VectorStoreIndex.from_vector_store(self.local_vector_store)

            print('ingestion complete')
            return self.cks_index, self.local_index
