import google.generativeai as genai
import os
from dotenv import load_dotenv

# This python code uses gemini-2.5-flash model API to generate text based on user prompt (text generation)

# load environment variables from .env file
load_dotenv()

# Configure Gemini Client
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("Error: 'GEMINI_API_KEY' not found in environment variables.")
    exit()

# Select Generative AI Model
model = genai.GenerativeModel(
    model_name='gemini-2.5-flash'
)

# Define the Prompt
prompt = "Complete the following: hello excuse me, where is the.."

# Define Generation Configuration
# A value closer to 0 makes the output more deterministic and a value closer to 1 increases randomness
generation_config = {
    "temperature": 0.8,           # Controls randomness
    "max_output_tokens": 800    # Sets max token limit
}

# Get the response and print in consol
response = model.generate_content(
    prompt,
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