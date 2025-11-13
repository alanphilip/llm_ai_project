
# This python code uses gemini-2.5-flash model API to generate response based on user chat

import google.generativeai as genai
import os
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# Configure Gemini Client
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("Error: 'GEMINI_API_KEY' not found in environment variables.")
    exit()

# 1. Define your system instruction
system_instruction = "You are a helpful assistant."

# Select Generative AI Model
model = genai.GenerativeModel(
    model_name='gemini-2.5-flash',
    system_instruction=system_instruction
)

# Define Generation Configuration
generation_config = {
    "max_output_tokens": 2000,
    "temperature": 0.7, # Added a temperature for good measure
}

# Define the Prompt
prompt = "how to cook ravioli?."

contents = [
    {'role': 'user', 'parts': [prompt]}
]

# Get the response and print in consol
response = model.generate_content(
    contents,
    generation_config=generation_config
)

try:
    print(response.text)
except ValueError:
    # If .text fails, print the whole response object for debugging
    print("Error: Response has no text. Printing full response for debugging:")
    print(response)

# (Optional) You can also print the finish reason to see why it stopped
print(f"\nFinish Reason: {response.candidates[0].finish_reason}")
