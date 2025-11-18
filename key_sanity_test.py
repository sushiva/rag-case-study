import os
from anthropic import Anthropic
from dotenv import load_dotenv # Only needed if using the .env method

# Load the environment file if you chose method 1
load_dotenv()

# Retrieve the key and print a confirmation (don't print the key itself!)
api_key = os.environ.get("ANTHROPIC_API_KEY")

if not api_key:
    print("❌ ERROR: ANTHROPIC_API_KEY environment variable not found.")
else:
    print("✅ Key found in environment.")
    
    try:
        # Initialize client (will use the key from the environment)
        client = Anthropic(api_key=api_key)
        
        # Use a stable, widely available model for the test
        TEST_MODEL = "claude-3-haiku-20240307" 
        
        # Make a simple, cheap API call
        print(f"⌛ Attempting a simple request with model: {TEST_MODEL}...")
        
        response = client.messages.create(
            model=TEST_MODEL,
            max_tokens=10,
            messages=[
                {"role": "user", "content": "hello"}
            ]
        )
        
        # Check if a response was generated
        if response.content and response.content[0].text:
            print(f"🎉 SUCCESS! API call worked.")
        else:
            print("⚠️ WARNING: API call succeeded but returned empty content.")

    except Exception as e:
        # This will catch 401 Unauthorized (bad key) or 404 Not Found (model issue)
        print(f"❌ API SANITY CHECK FAILED:")
        print(f"   Error: {e}")