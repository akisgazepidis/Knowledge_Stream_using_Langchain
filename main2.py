import warnings
from utils.utils import setup_agent
from dotenv import load_dotenv
import os

# Filter out specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", ".*Hugging Face Hub.*")

# Load environment variables
load_dotenv()

def main():
    # Get directory configurations from environment variables
    files_directory = os.getenv('FILES_DIRECTORY', 'files')
    persist_directory = os.getenv('VECTOR_STORE_DIRECTORY', 'vector_store')
    
    # Initialize the agent with directory configurations
    agent = setup_agent(files_directory, persist_directory)
    
    # Define the task for the agent with specific instructions
    user_goal = """
    Please follow these steps in order:
    1. Process the PDF documents from the files directory
    2. Store the processed documents as vector embeddings in the database
    3. Confirm when the documents have been successfully stored
    Do not perform any vector comparisons or additional operations."""
    
    try:
        # Using invoke instead of deprecated run method
        result = agent.invoke({"input": user_goal})
        print(f"\nTask completed: {result}")
    except Exception as e:
        print(f"Error running agent: {str(e)}")

if __name__ == "__main__":
    main()