from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

from sentence_transformers import SentenceTransformer

from api.compressor import Compressor
from api.ner import NER
from api.open_ai import OpenAI
from api.prompt_database import PromptDatabase

open_ai_wrapper = OpenAI()
ner_wrapper = NER()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
prompt_db = PromptDatabase(embedding_model=embedding_model)

compressor = Compressor()

@csrf_exempt
def process_prompt_api(request):
    if request.method == "POST":
        data = json.loads(request.body)
        user_prompt = data.get("prompt")
        
        # Step 1: Apply NER
        anonymized_prompt = ner_wrapper.apply_ner(user_prompt)
        entity_mapping = {}
        
        # Step 2: Search FAISS
        matched_prompt = prompt_db.search_prompt(anonymized_prompt)
        if matched_prompt:
            final_prompt = ner_wrapper.reverse_ner(matched_prompt)
        else:
            print("No sufficiently similar prompt found in database.")
            optimized_prompt = open_ai_wrapper.optimize(user_prompt)
            compressed_prompt = compressor.compress_prompt_api(optimized_prompt)
            final_prompt = compressed_prompt
            anonymized_compressed_prompt = ner_wrapper.apply_ner(compressed_prompt)
            prompt_db.add_prompt(anonymized_compressed_prompt)
        
        user_prompt_eval_score = open_ai_wrapper.evaluate_compression(user_prompt)
        final_prompt_eval_score = open_ai_wrapper.evaluate_compression(final_prompt)
        
        return JsonResponse({
                "original_prompt": user_prompt, 
                "original_evaluation_score": user_prompt_eval_score, 
                "final_prompt": final_prompt, 
                "final_evaluation_score": final_prompt_eval_score
            })
    return JsonResponse({"error": "Invalid request"}, status=400)
