"""Setup script for OpenAI API key"""

import os
import sys

def setup_openai_key():
    """Set up OpenAI API key"""
    print("Setting up OpenAI API key...")
    
    # Check if key is already set
    if os.getenv("OPENAI_API_KEY"):
        print("OpenAI API key is already set.")
        return
    
    # Get key from user
    key = input("Enter your OpenAI API key: ").strip()
    if not key:
        print("Error: API key cannot be empty")
        sys.exit(1)
    
    # Write key to .env file
    with open(".env", "a") as f:
        f.write(f"\nOPENAI_API_KEY={key}\n")
    
    print("OpenAI API key has been saved to .env file.")
    print("You can now run the evaluation script.")

if __name__ == "__main__":
    setup_openai_key()
