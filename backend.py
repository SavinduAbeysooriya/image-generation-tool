from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import sys
from dotenv import load_dotenv  # Optional: For environment variables

# Load environment variables from a .env file (optional but recommended)
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Check Python version
print(f"Running on Python {sys.version}")

# Use environment variable for token (secure practice); fallback to hardcoded value for testing
HF_TOKEN = os.getenv("HF_TOKEN", "hf_##############################")
API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"  # Fallback to a widely accessible model
headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

def query(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)  # Increased timeout
        print("API Response Status:", response.status_code)
        print("API Response Text:", response.text)  # Log the full response for debugging
        
        if response.status_code == 404:
            raise Exception(f"404 Error: Model or endpoint not found - {response.text}")
        elif response.status_code == 429:
            raise Exception(f"429 Error: Rate limit exceeded - {response.text}")
        elif response.status_code == 401:
            raise Exception(f"401 Error: Unauthorized - Please check your token or model access - {response.text}")
        elif response.status_code == 403:
            raise Exception(f"403 Error: Forbidden - You may need a paid tier or model access - {response.text}")
        elif response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
        
        # Check if the response contains an image
        if "image" not in response.headers.get("Content-Type", "").lower():
            raise Exception(f"Unexpected response format: {response.text}")
        
        return response.content
    except requests.RequestException as e:
        raise Exception(f"Request failed: {str(e)}")

@app.route("/generate", methods=["POST"])
def generate_image():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")

        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        # Enhanced prompt for better quality
        prompt = f"{prompt}, highly detailed, 4k resolution, vibrant colors, realistic lighting, ultra-realistic, in Sri Lankan style"

        # Prepare payload with optimized parameters
        payload = {
            "inputs": prompt,
            "parameters": {
                "num_inference_steps": 30,  # Reduced for faster testing; adjust to 50 for SDXL if accessible
                "guidance_scale": 7.5,     # Adjustable up to 12 for better detail
                "height": 512,             # Default resolution for stable-diffusion-v1-5
                "width": 512               # Match height for square output
            }
        }

        # Query the API and get the image bytes
        image_bytes = query(payload)
        return app.response_class(image_bytes, mimetype="image/png")

    except Exception as e:
        print(f"Exception in generate_image: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
