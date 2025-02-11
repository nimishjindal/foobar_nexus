from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings



SIMILARITY_THRESHOLD = 0.85

class PromptDatabase:  

    def __init__(self, pinecone_pass, pinecone_index="prompts", vector_dim=384):
        """Initialize vector database for similarity search"""        
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")        
        self.vector_store = PineconeVectorStore(pinecone_api_key=pinecone_pass, index_name=pinecone_index, embedding=embedding_model)
        
        self.prompts = {}
        self.id_counter = 0
    
    def add_prompt(self, prompt: str, user="Nimish"):
        """Embeds and stores a prompt in the vector database"""
        
        idx = str(self.id_counter)
        
        self.prompts[idx] = prompt
        
        document = {
                    "id": idx, 
                    'text': prompt, 
                }
        
        self.vector_store.add_documents([document])
        
        self.id_counter += 1

    def search_prompt(self, query: str, top_k=1):
        """Finds the most similar prompt from the database"""        
        results = self.vector_store.similarity_search(query, k=top_k)
        
        print(f"vector responses: {results}")
        
        if results:
                    
            best_match_idx = results[0]["id"]
            best_match_distance = results[0]["score"]
            
            if best_match_idx in self.prompts and best_match_distance < (1 - SIMILARITY_THRESHOLD):
                print(f"Similarity score: {best_match_distance}")
                return self.prompts[best_match_idx]
        
        return None
