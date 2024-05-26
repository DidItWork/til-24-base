# To get environment variables
import os
import json

from rich.pretty import pprint


# To split our transcript into pieces
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Our chat model. We'll use the default which is gpt-3.5-turbo
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline
from langchain.chains.summarize import load_summarize_chain

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
from typing import List, Optional

llm_phi3 = HuggingFacePipeline.from_model_id(
    model_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 512, "temperature": 0.1,},
)


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


# Define your desired data structure.
class Objective(BaseModel):
    """The data to be extracted from the transcript"""
    target: str = Field(description="Identified Target object to attack")
    heading: str = Field(description="Heading of object in integers")
    tool: str = Field(description="Tool to be deployed to neutralize the target")

def clean_json_string(json_str: str) -> str:
    # Find the start of the first JSON object
    start = json_str.find('{')

    # Find the end of the first JSON object
    end = json_str.find('}') + 1

    # Extract the substring from 'start' to 'end'
    cleaned_json_str = json_str[start:end]

    return cleaned_json_str

def extract_json_after_response(message: str) -> dict:
    # Find the start of the JSON object
    start = message.lower().find('response') + len('response')

    # Extract the substring from 'start' to the end of the message
    json_str = message[start:].strip()
    json_str = clean_json_string(json_str)

    try:
        # Parse the JSON string into a Python dictionary
        data = json.loads(json_str)
    except json.JSONDecodeError:
        print(f"Error parsing JSON string: {json_str}")
        raise

    return data

output_parser = JsonOutputParser(pydantic_object=Objective)

prompt = PromptTemplate(
    template=template,
    input_variables=["transcript"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)

chain = prompt | llm_phi3 | extract_json_after_response


from tqdm import tqdm
import json

def load_test_cases(filename: str) -> list:
  test_cases = []
  with open(filename, 'r') as file:
    for line in file:
      # Each line is a separate JSON object
      test_case = json.loads(line)
      test_cases.append(test_case)
  return test_cases

# Load the test cases from the JSONL file
test_cases = load_test_cases('nlp.jsonl')

# Counter for passed test cases
passed_tests = 0

# Now you can use 'test_cases' in your tests
for test_case in tqdm(test_cases):
  transcript = test_case['transcript']
  expected_output = {
    'target': test_case['target'],
    'heading': test_case['heading'],
    'tool': test_case['tool']
  }

  # Invoke the chain with the transcript from the test case
  extracted_output = chain.invoke({"transcript": transcript})

  # Check if the extracted output matches the expected output
  if extracted_output == expected_output:
    passed_tests += 1
  else:
    print(f"For transcript: {transcript}, expected: {expected_output}, but got: {extracted_output}")

# Print the number of passed tests
print(f"Passed {passed_tests} out of {len(test_cases)} tests.")