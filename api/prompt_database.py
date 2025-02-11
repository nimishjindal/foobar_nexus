import faiss
import numpy as np

SIMILARITY_THRESHOLD = 0.85

class PromptDatabase:    

    def __init__(self, embedding_model, vector_dim=384,):
        """Initialize FAISS vector database for similarity search"""
        self.index = faiss.IndexFlatL2(vector_dim)
        self.prompts = {}
        self.id_counter = 0
        self.embedding_model=embedding_model

    def add_prompt(self, prompt: str):
        """Embeds and stores a prompt in the vector database"""
        embedding = self.embedding_model.encode(prompt).astype(np.float32)
        self.index.add(np.array([embedding]))
        self.prompts[self.id_counter] = prompt
        self.id_counter += 1

    def search_prompt(self, query: str, top_k=1):
        """Finds the most similar prompt from the database"""
        query_embedding = self.embedding_model.encode(query).astype(np.float32)
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        best_match_idx = indices[0][0]
        best_match_distance = distances[0][0]
        if best_match_idx in self.prompts and best_match_distance < (1 - SIMILARITY_THRESHOLD):
            return self.prompts[best_match_idx], best_match_distance
        return None, None
