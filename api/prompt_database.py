import numpy as np
from pinecone import Pinecone


SIMILARITY_THRESHOLD = 0.85

class PromptDatabase:  

    def __init__(self, embedding_model, pinecone_pass, pinecone_index="prompts", vector_dim=384):
        """Initialize vector database for similarity search"""
        self.index = Pinecone(api_key=pinecone_pass).Index(pinecone_index)
        self.prompts = {}
        self.id_counter = 0
        self.embedding_model=embedding_model

    def __encode(self, prompt: str):
        """Encodes a prompt using the SentenceTransformer model"""
        return self.embedding_model.encode(prompt).astype(np.float32)
    
    def add_prompt(self, prompt: str, user="Nimish"):
        """Embeds and stores a prompt in the vector database"""
        
        idx = str(self.id_counter)
        
        self.prompts[idx] = prompt
        
        embedding = self.__encode(prompt)

        self.index.upsert(
            vectors=[
                {
                    "id": idx, 
                    "values": embedding, 
                }
            ],
            namespace= user
        )
        self.id_counter += 1

    def search_prompt(self, query: str, top_k=1):
        """Finds the most similar prompt from the database"""
        query_embedding = self.__encode(query).tolist()
        
        results = self.index.query(vector=query_embedding,top_k=top_k)
        
        print(f"vector responses: {results}")
        
        if results["matches"]:
        
            matched_record = results["matches"][0]
            
            best_match_idx = matched_record["id"]
            best_match_distance = matched_record["score"]
            
            if best_match_idx in self.prompts and best_match_distance < (1 - SIMILARITY_THRESHOLD):
                print(f"Similarity score: {best_match_distance}")
                return self.prompts[best_match_idx]
        
        return None
