import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
import re


class OpenAI:
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature = 0.2, max_tokens = 1000, top_p = 0.95)
    
    def optimize(self, original_prompt: str) -> str:
                
        optimization_prompt = """### PROMPT OPTIMIZATION TASK
            Improve this prompt for LLM processing while maintaining EXACT semantic meaning:

            "{prompt}"

            ### CONSTRAINTS
            1. Ensure there are no grammatical errors.
            2. Preserve all technical terms, numbers, and key requirements.
            3. Use markdown sections (## for main headers) for better readability.
            4. Maintain the original intent and scope.
            5. Output only the optimized prompt.
            """
                    
        messages = [
                {"role": "system", "content": "You are a professional prompt engineer specializing in LLM input optimization."},
                {"role": "user", "content": optimization_prompt.format(prompt=original_prompt)}
            ]
        
        return self.llm.invoke(messages).content

    def evaluate_compression(self, prompt: str) -> str:

        llm = ChatOpenAI(model="gpt-4", temperature = 0.2, max_tokens = 1000, top_p = 0.95)

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
        
        messages=[{"role": "user", "content": evaluation_prompt.format(prompt=prompt)}]

        score = self.llm.invoke(messages).content
        
        match = re.search(r'\b[1-4]\b', score)
        return int(match.group()) if match else 1


if "__main__" == __name__:

    print("Testing the utility functions...")
    load_dotenv(find_dotenv())

    openai = OpenAI()
    improved = openai.optimize("ummm. hi how ar eyou. help me write an email to nimish, ")
    scores = openai.evaluate_compression(improved)
    
    print(improved)
    print(scores)