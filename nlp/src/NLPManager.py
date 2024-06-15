from typing import Dict, List
import torch
from transformers import AutoTokenizer, EncoderDecoderModel

device = "cuda" if torch.cuda.is_available() else "cpu"

class NLPManager:
    def __init__(self):
        # initialize the model here
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("/workspace/models/tokenizer")

        self.bert2bert = EncoderDecoderModel.from_pretrained("/workspace/models/nlp-ft").to(device)
        
        self.bert2bert.eval()

    def qa(self, context: List[str]) -> Dict[str, str]:
        # perform NLP question-answering
        with torch.no_grad():
            input_tokenized = self.tokenizer(context, padding="max_length", truncation=True, max_length=64, return_tensors="pt").to(device)
            outputs = self.bert2bert.generate(input_tokenized.input_ids.to(device), attention_mask=input_tokenized.attention_mask.to(device))
            output_str = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        outputs_formated = []

        for output in output_str:

            try:
                tool, heading, target = output.split("|")
            except:
                heading = ""
                tool = ""
                target = ""
            
            outputs_formated.append({"heading": heading.strip(), "tool": tool.strip(), "target": target.strip()})

        return outputs_formated

if __name__ == "__main__":

    nlp_manger = NLPManager()