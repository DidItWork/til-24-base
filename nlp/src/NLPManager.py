from typing import Dict

import os
import json

from rich.pretty import pprint

# Prompt templates for dynamic values
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate, # I included this one so you know you'll have it but we won't be using it
    HumanMessagePromptTemplate
)

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.output_parsers import PydanticOutputParser

# To create our chat messages
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompt_values import StringPromptValue
from typing import List, Optional

from langchain.llms.huggingface_pipeline import HuggingFacePipeline
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig

# Define your desired data structure.
class Objective(BaseModel):
    """The data to be extracted from the transcript"""
    target: str = Field(description="Identified Target object to attack")
    heading: str = Field(description="Heading of object in integers")
    tool: str = Field(description="Tool to be deployed to neutralize the target")

class NLPManager:
    def __init__(self):
        # initialize the model here
        nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        )

        model_id = "microsoft/Phi-3-mini-4k-instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        quantization_config=nf4_config
                        )

        pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    )

        self.llm_phi3 = HuggingFacePipeline(pipeline=pipe)

        self.chain = self._create_chain()

    def _create_chain(self):
        # create the chain here
        template = """
        Given the transcript delimitted by triple backticks,
        extract all key information in a structured format.
        {format_instructions}
        Return the JSON object only, nothing else.

        Here are some examples of how to perform this task:
        TRANSCRIPT: Target is red helicopter, heading is zero one zero, tool to deploy is surface-to-air missiles.
        OUTPUT: 'target': 'red helicopter', 'heading': '010', 'tool': 'surface-to-air missiles'

        TRANSCRIPT: Target is grey and white fighter jet, heading is zero six five, tool to deploy is electromagnetic pulse.
        OUTPUT: 'target': 'grey and white fighter jet', 'heading': '065', 'tool': 'electromagnetic pulse'

        TRANSCRIPT: Alfa, Echo, Mike Papa, deploy EMP tool heading zero eight five, engage purple, red, and silver fighter jet.
        OUTPUT: 'target': 'purple, red, and silver fighter jet', 'heading': '085', 'tool': 'EMP'

        Now take the following transcript and extract the key information in a structured format
        TRANSCRIPT:
        ```{transcript}```
        """
        output_parser = JsonOutputParser(pydantic_object=Objective)

        prompt = PromptTemplate(
            template=template,
            input_variables=["transcript"],
            partial_variables={"format_instructions": output_parser.get_format_instructions()},
        )

        chain = prompt | self.llm_phi3 | self._extract_json_after_response
        return chain

    def _clean_json_string(self, json_str: str) -> str:
        # Find the start of the first JSON object
        start = json_str.find('{')

        # Find the end of the first JSON object
        end = json_str.find('}') + 1

        # Extract the substring from 'start' to 'end'
        cleaned_json_str = json_str[start:end]

        return cleaned_json_str

    def _extract_json_after_response(self, message: str) -> dict:
        # Find the start of the JSON object
        start = message.lower().find('response') + len('response')

        # Extract the substring from 'start' to the end of the message
        json_str = message[start:].strip()
        json_str = self._clean_json_string(json_str)

        try:
            # Parse the JSON string into a Python dictionary
            data = json.loads(json_str)
        except json.JSONDecodeError:
            print(f"Error parsing JSON string: {json_str}")
            return None

        return data

    def qa(self, context: str) -> Dict[str, str]:
        # perform NLP question-answering
        return self.chain.invoke({"transcript": context})
