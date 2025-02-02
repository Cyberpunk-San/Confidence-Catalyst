import cohere
import os
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load API key from .env file
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Validate API key
if not COHERE_API_KEY:
    logging.error("Cohere API key not found. Please check your .env file.")
    raise ValueError("Cohere API key is required.")

# Initialize Cohere client
co = cohere.Client(COHERE_API_KEY)

def generate_response(user_input):
    try:
        # Construct a detailed prompt for better context
        prompt = (
            f"you need to teach user what to speak during conversation, now teach them other user is saying :'{user_input}'"
        )

        # Generate the response using Cohere's API
        response = co.generate(
            model="command",
            prompt=prompt,
            temperature=0.7, 
            max_tokens=70 
        )
        response_text = response.generations[0].text.strip()
        return response_text

    except cohere.CohereError as e:
        logging.error(f"Cohere API error: {e}")
        return "ERROR 1"

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return "ERROR"
