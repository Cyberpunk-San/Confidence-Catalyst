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

def generate_motivation(user_input):
    """
    Generate an encouraging and calming response to help reduce stress.
    
    Args:
        user_input (str): The user's input or concern.
    
    Returns:
        str: A motivational and calming response.
    """
    try:
        # Construct a detailed prompt for better context
        prompt = (
            f"You are a motivational and calming assistant. Your goal is to help the user feel better "
            f"by providing uplifting and thoughtful responses. The user has shared the following: "
            f"'{user_input}'. Respond in a way that is empathetic, encouraging, and calming. "
            f"Provide a complete and detailed response to address their concern fully."
        )

        # Generate the response using Cohere's API
        response = co.generate(
            model="command",  # Use Cohere's "command" model
            prompt=prompt,
            temperature=0.7,  # Adjust creativity (0.7 is a good balance)
            # Remove stop_sequences to allow full responses
        )

        # Extract and clean the generated text
        motivation_text = response.generations[0].text.strip()
        return motivation_text

    except cohere.CohereError as e:
        logging.error(f"Cohere API error: {e}")
        return "I'm here for you. Take a deep breath, and everything will be okay. 🌿💙"

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return "I'm here for you. Let's take a moment to breathe and refocus. 🌿💙"

# Example usage
if __name__ == "__main__":
    user_input = "I'm feeling really stressed about work."
    response = generate_motivation(user_input)
    print(f"Motivation Bot: {response}")