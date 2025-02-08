import spacy
import faiss
import numpy as np
import requests
from llmlingua import PromptCompressor
from sentence_transformers import SentenceTransformer
import openai

# Load NLP and embedding models
nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# FAISS Vector Similarity Search
SIMILARITY_THRESHOLD = 0.85
# Load OpenAI API
client = openai.OpenAI(api_key="")

class PromptDatabase:
    def __init__(self, vector_dim=384):
        """Initialize FAISS vector database for similarity search"""
        self.index = faiss.IndexFlatL2(vector_dim)
        self.prompts = {}
        self.id_counter = 0

    def add_prompt(self, prompt: str):
        """Embeds and stores a prompt in the vector database"""
        embedding = embedding_model.encode(prompt).astype(np.float32)
        self.index.add(np.array([embedding]))
        self.prompts[self.id_counter] = prompt
        self.id_counter += 1

    def search_prompt(self, query: str, top_k=1):
        """Finds the most similar prompt from the database"""
        query_embedding = embedding_model.encode(query).astype(np.float32)
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        best_match_idx = indices[0][0]
        best_match_distance = distances[0][0]
        if best_match_idx in self.prompts and best_match_distance < (1 - SIMILARITY_THRESHOLD):
            return self.prompts[best_match_idx], best_match_distance
        return None, None

prompt_db = PromptDatabase()

llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,
    device_map="cpu"
)

def compress_prompt_api(prompt: str):
    return llm_lingua.compress_prompt(
            prompt, rate=0.8, force_tokens=['\n', '?']
        )['compressed_prompt']

def apply_ner(prompt: str):
    """Perform Named Entity Recognition (NER) and replace key entities"""
    doc = nlp(prompt)
    anonymized_prompt = prompt
    entity_mapping = {}
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "DATE"]:
            placeholder = f"[{ent.label_}]"  # Example: "Jeff" → "[PERSON]"
            entity_mapping[placeholder] = ent.text  # Store mapping
            anonymized_prompt = anonymized_prompt.replace(ent.text, placeholder)
            # anonymized_prompt = anonymized_prompt.replace(ent.text, f"[{ent.label_}]")

    return anonymized_prompt, entity_mapping

def reverse_ner(anonymized_prompt: str, entity_mapping: dict):
    """Replaces placeholders with original user-provided values"""
    restored_prompt = anonymized_prompt
    for placeholder, original_text in entity_mapping.items():
        restored_prompt = restored_prompt.replace(placeholder, original_text)
    
    return restored_prompt

class PromptOptimizer:
    def __init__(self):
        self.optimization_prompt = """### PROMPT OPTIMIZATION TASK
        Improve this prompt for LLM processing while maintaining EXACT semantic meaning:

        "{prompt}"

        ### CONSTRAINTS
        1. Ensure there are no grammatical errors.
        2. Preserve all technical terms, numbers, and key requirements.
        3. Use markdown sections (## for main headers) for better readability.
        4. Maintain the original intent and scope.
        5. Output only the optimized prompt.
        """

    def optimize(self, original_prompt: str) -> str:
        """Optimizes a prompt using GPT-4"""
        messages = [
            {"role": "system", "content": "You are a professional prompt engineer specializing in LLM input optimization."},
            {"role": "user", "content": self.optimization_prompt.format(prompt=original_prompt)}
        ]

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.2,
            max_tokens=1000,
            top_p=0.95
        )
        return response.choices[0].message.content
    
prompt_optimizer = PromptOptimizer()

def optimize_prompt(original_prompt: str):
    return prompt_optimizer.optimize(original_prompt)

# def compress_prompt_api(prompt: str):
#     VERTEX_AI_URL = "https://your-vertex-endpoint.com"
#     response = requests.post(VERTEX_AI_URL, json={"prompt": prompt})
#     if response.status_code == 200:
#         return response.json().get("compressed_prompt", prompt)
#     return prompt

def evaluate_compression(prompt: str) -> str:
    """Evaluates the compression quality of a prompt using GPT-4."""
    
    evaluation_prompt = """  
    You will be given a prompt. Your task is to evaluate how well it retains key details while being as short as possible.  
    You are assessing **prompt compression**—meaning grammar mistakes or missing words are acceptable as long as the meaning remains clear and logical flow is intact. 

    Rate the prompt on a scale of 1 to 4, where:  
    1: Too long and redundant; could be much shorter without losing meaning.  
    2: Slightly compressed but still contains unnecessary details.  
    3: Well-compressed, mostly retaining key details with minimal redundancy.  
    4: Excellent—highly compressed while keeping all essential information clear.  

    Provide your feedback in the following format:  

    **Total rating:** (your rating as a number between 1 and 4)  

    Now, evaluate the following prompt:  

    **Prompt:**  
    {prompt}  

    **Final Output:** (numerical value only)  
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": evaluation_prompt.format(prompt=prompt)}]
    )

    return response.choices[0].message.content

def extract_score(eval_text):
    """Extracts the first integer found in the evaluation output."""
    import re
    match = re.search(r'\b[1-4]\b', eval_text)
    return int(match.group()) if match else 1




