import spacy

class NER:
    # class to support Named Entity Recognition (NER)
    
    def __init__(self, entity_mapping):
        self.nlp = spacy.load("en_core_web_sm")
        self.entity_mapping = entity_mapping
    
    def apply_ner(self, prompt: str):
        # NER and replace key entities
        doc = self.nlp(prompt)
        anonymized_prompt = prompt
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "DATE"]:
                placeholder = f"[{ent.label_}]"  # Example: "Jeff" → "[PERSON]"
                self.entity_mapping[placeholder] = ent.text  # Store mapping
                anonymized_prompt = anonymized_prompt.replace(ent.text, placeholder)

        return anonymized_prompt

    def reverse_ner(self, anonymized_prompt: str):
        # Replaces placeholders with original user-provided values
        restored_prompt = anonymized_prompt
        for placeholder, original_text in self.entity_mapping.items():
            restored_prompt = restored_prompt.replace(placeholder, original_text)
        
        return restored_prompt
