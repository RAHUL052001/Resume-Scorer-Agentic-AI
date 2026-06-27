import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load variables from .env file
load_dotenv()

def generate_text_langchain(prompt: str) -> str:
    try:
        # Initialize the Google Gemini model via LangChain
        # We use the standard fast model: 'gemini-2.5-flash'
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            # max_tokens=200
        )
        
        # Invoke the model using the LangChain standard interface
        response = llm.invoke(prompt)
        # print(response)
        
        # LangChain returns an AIMessage object; .content extracts the string response
        return response.content
        
    except Exception as e:
        return f"Error: {e}"

# Test your function
output = generate_text_langchain("Give me a one-sentence motivational quote about coding.")
print(output)