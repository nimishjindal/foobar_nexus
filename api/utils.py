from llmlingua import PromptCompressor
from sentence_transformers import SentenceTransformer

from api.ner import NER
from api.open_ai import OpenAI
from api.prompt_database import PromptDatabase

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

prompt_db = PromptDatabase(embedding_model=embedding_model)

llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,
    device_map="cpu"
)

def compress_prompt_api(prompt: str):
    return llm_lingua.compress_prompt(
            prompt, rate=0.8, force_tokens=['\n', '?']
        )['compressed_prompt']

ner_wrapper = NER()

def apply_ner(prompt: str):
    return ner_wrapper.apply_ner(prompt)

def reverse_ner(anonymized_prompt: str):
    return ner_wrapper.reverse_ner(anonymized_prompt)

open_ai_wrapper = OpenAI()

def optimize_prompt(original_prompt: str):
    return open_ai_wrapper.optimize(original_prompt)

def evaluate_compression(prompt: str) -> str:
    return open_ai_wrapper.evaluate_compression(prompt)
