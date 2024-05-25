from typing import Dict
import logging


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig
from jsonformer.main import Jsonformer


class HuggingFaceLLM:
    def __init__(self, temperature=0, top_k=50, model_name="microsoft/Phi-3-mini-4k-instruct"):
        nf4_config = BitsAndBytesConfig(
          load_in_4bit=True,
          bnb_4bit_quant_type="nf4",
          bnb_4bit_use_double_quant=True,
          bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_path = "/workspace/models/model"
        tokenizer_path = "/workspace/models/tokenizer"

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=nf4_config, device_map="auto")

        # self.model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=True, quantization_config=nf4_config, device_map="auto")
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_cache=True)

        # self.model.save_pretrained("./src/models/model")
        # self.tokenizer.save_pretrained("./src/models/tokenizer")

        self.top_k = top_k

    def generate(self, prompt, max_length=1024):
        json = {
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "tool": {"type": "string"},
                "heading": {"type": "string"},
            }
        }

        builder = Jsonformer(
            model=self.model,
            tokenizer=self.tokenizer,
            json_schema=json,
            prompt= prompt,
            max_string_token_length=20
        )

        print("Generating...")
        output = builder()
        return output

class NLPManager:
    def __init__(self):
        # initialize the model here
        self.llm = HuggingFaceLLM(temperature=0)
        self.template = """
        You are an expert admin people who will extract core information from documents

        {content}

        Above is the content; please try to extract all data points from the content above:
        {data_points}
        """

        self.default_data_points = """{
            "target": "Identified Target object to attack",
            "tool": "Tool to be deployed to neutralize the target",
            "heading": "Heading of object in integers",
        }"""

    def qa(self, context: str) -> Dict[str, str]:
        # perform NLP question-answering
        default_result = {"heading": "", "tool": "", "target": ""}
        try:
            formatted_template = self.template.format(content=context, data_points=self.default_data_points)
            data = self.llm.generate(formatted_template)

            if data is None:
                print('Failed to extract data from response: %s', context)
                print('LLM generated response: %s', data)
                return default_result

            # Validate that data is a dictionary and contains the required keys
            if not isinstance(data, dict) or not all(key in data for key in ['target', 'heading', 'tool']):
                print('Data does not adhere to the required format: %s', data)
                print('LLM generated response: %s', data)
                return default_result

            logging.info('Successfully extracted data: %s', data)
            print('Successfully extracted data: %s', data)

        except Exception as e:
            logging.error('Failed to extract data due to error: %s', e)
            print('Failed to extract data due to error: %s', e)
            return default_result

        return data
