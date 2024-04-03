from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain.chains import create_extraction_chain
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


prompt =ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a top-tier algorithm for extracting information from text. "
            "Only extract information that is relevant to the provided text. "
            "If no information is relevant, use the schema and output "
            "an empty list where appropriate."
        ),
        ("user",
            "I need to extract information from "
            "the following text: ```\n{text}\n```\n",
        ),
    ]
)
# Schema
schema = {
  "type": "object",
  "title": "Recipe Information Extractor",
  "$schema": "http://json-schema.org/draft-07/schema#",
  "required": [
    "recipes"
  ],
  "properties": {
    "recipes": {
      "type": "array",
      "items": {
        "type": "object",
        "required": [
          "name",
          "ingredients"
        ],
        "properties": {
          "name": {
            "type": "string",
            "description": "The name of the recipe."
          },
          "ingredients": {
            "type": "array",
            "items": {
              "type": "object",
              "required": [
                "name",
                "amount",
                "unit"
              ],
              "properties": {
                "name": {
                  "type": "string",
                  "description": "The name of the ingredient."
                },
                "unit": {
                  "type": "string",
                  "description": "The unit of the amount of the ingredient."
                },
                "amount": {
                  "type": "number",
                  "description": "The numeric amount of the ingredient."
                }
              }
            }
          }
        }
      }
    }
  },
  "description": "Schema for extracting recipe information from text."
}

loader = PyMuPDFLoader("./recipe.pdf")
docs = loader.load()

def split_docs(documents, chunk_size=int(128_000 * 0.8), chunk_overlap=20):
    
    # Initializing the RecursiveCharacterTextSplitter with
    # chunk_size and chunk_overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Splitting the documents into chunks
    chunks = text_splitter.split_documents(documents=documents)
    
    # returning the document chunks
    return chunks
documents = split_docs(documents=docs)
# Run chain
llm = OllamaFunctions(model="mistral:7b-instruct", temperature=0)
chain = prompt | create_extraction_chain(schema, llm)
responses = []
for document in documents:
  input_data = {
          "text": document,
          "json_schema": schema,  
          "instruction": (
              "recipe.each recipe has a name and list of ingredients.ingredients should have a name,numeric amount,and unit of amount"
          )
      }
  response = chain.invoke(input_data)
  responses.append(response)
for response in responses:
    result = response['text']
    print(json.dumps(result, indent=4))