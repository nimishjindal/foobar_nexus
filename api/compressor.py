from llmlingua import PromptCompressor


class Compressor:
    
    def __init__(self, model = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"):
        self.__llm_lingua = PromptCompressor(model_name=model, use_llmlingua2=True, device_map="cpu")

    def compress_prompt_api(self, prompt: str):
        return self.__llm_lingua.compress_prompt(prompt, rate=0.8, force_tokens=['\n', '?'])['compressed_prompt']
