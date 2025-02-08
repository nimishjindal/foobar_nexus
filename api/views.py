from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .utils import apply_ner, optimize_prompt, compress_prompt_api, evaluate_compression, reverse_ner, prompt_db

@csrf_exempt
def process_prompt_api(request):
    if request.method == "POST":
        data = json.loads(request.body)
        user_prompt = data.get("prompt", "")
        
        # Step 1: Apply NER
        anonymized_prompt, entity_mapping = apply_ner(user_prompt)
        
        # Step 2: Search FAISS
        matched_prompt, similarity_score = prompt_db.search_prompt(anonymized_prompt)
        if matched_prompt:
            print(f"Similarity score: {similarity_score}")
            final_prompt = reverse_ner(matched_prompt, entity_mapping)
        else:
            print("No sufficiently similar prompt found in database.")
            optimized_prompt = optimize_prompt(user_prompt)
            compressed_prompt = compress_prompt_api(optimized_prompt)
            final_prompt = compressed_prompt
            anonymized_compressed_prompt, _ = apply_ner(compressed_prompt)
            prompt_db.add_prompt(anonymized_compressed_prompt)
        
        user_prompt_eval_score = evaluate_compression(user_prompt)
        final_prompt_eval_score = evaluate_compression(final_prompt)
        
        return JsonResponse({"original_prompt": user_prompt, "original_evaluation_score": user_prompt_eval_score, "final_prompt": final_prompt, "final_evaluation_score": final_prompt_eval_score})
    return JsonResponse({"error": "Invalid request"}, status=400)
