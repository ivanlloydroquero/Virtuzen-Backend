# app.py
import os
import logging
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('VirtuzenAI')

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')
chat_model = model.start_chat(history=[])

def handle_gemini_error(error):
    """Handle Gemini API errors and return appropriate response"""
    logger.error(f"Gemini API Error: {str(error)}")
    return jsonify({
        "error": "AI service temporarily unavailable",
        "details": str(error)
    }), 503

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint for service health check"""
    return jsonify({
        "status": "online",
        "version": "2.3.1",
        "models": ["gemini-pro", "gemini-1.5-flash"]
    })

@app.route('/api/chat', methods=['POST'])
def chat_handler():
    """Main chat endpoint with context-aware processing"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({"error": "Empty message received"}), 400
        
        # Enhanced context handling
        context = {
            "system_prompt": "You are Virtuzen AI, an advanced AI assistant. Respond in markdown when appropriate.",
            "user_history": data.get('context', []),
            "current_model": "gemini-2.0-flash"
        }
        
        # Generate response with safety settings
        response = chat_model.send_message(
            f"{context['system_prompt']}\n\nUser: {message}",
            safety_settings={
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
            },
            generation_config={
                'max_output_tokens': 2048,
                'temperature': 0.7
            }
        )
        
        return jsonify({
            "candidates": [{
                "content": {
                    "parts": [{"text": response.text}],
                    "role": "model"
                }
            }],
            "context": context
        })
        
    except Exception as e:
        return handle_gemini_error(e)

@app.route('/api/chat2', methods=['POST'])
def tutor_handler():
    """Specialized tutoring endpoint with educational focus"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({"error": "Empty message received"}), 400
        
        # Tutor-specific configuration
        tutor_model = genai.GenerativeModel('gemini-1.5-flash')
        response = tutor_model.generate_content(
            f"You are Virtuzen Tutor, an expert educational AI. Provide detailed, step-by-step explanations for: {message}",
            safety_settings={
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_ONLY_HIGH',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_ONLY_HIGH'
            }
        )
        
        return jsonify({
            "candidates": [{
                "content": {
                    "parts": [{"text": response.text}],
                    "role": "model"
                }
            }]
        })
        
    except Exception as e:
        return handle_gemini_error(e)

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    )