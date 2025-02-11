from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from uuid import uuid4


SIMILARITY_THRESHOLD = 0.85

class PromptDatabase:  

    def __init__(self):
        """Initialize vector database for similarity search"""        
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")        
        self.vector_store = PineconeVectorStore(embedding=embedding_model)
        
    def add_prompt(self, prompt: str, user="Nimish"):
        """Embeds and stores a prompt in the vector database"""
        
        document = Document(page_content=prompt)
        self.vector_store.add_documents(documents=[document], ids=[str(uuid4())])

    def search_prompt(self, query: str, top_k=1):
        """Finds the most similar prompt from the database"""        
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        
        print(f"vector responses: {results}")
        
        if results:
            
            print(results[0])
            
            document, score = results[0]
            
            if score > SIMILARITY_THRESHOLD:
                return document.page_content
        
        return None
