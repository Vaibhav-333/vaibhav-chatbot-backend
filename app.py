from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import difflib
import requests
import os
import re
import logging
import sys
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import time
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ],
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["*"])

# Configuration
class Config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GEMINI_MODEL = 'gemini-1.5-flash'
    # FIX: Base URL without parameters
    GEMINI_BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    
    BASE_DIR = os.path.dirname(os.path.abspath(__name__))
    FAQ_FILE = os.path.join(BASE_DIR, 'personal_faq.json')
    
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    RATE_LIMIT_DELAY = 1

# Global FAQ data storage
faq_data = []
last_request_time = 0

class FAQMatcher:
    @staticmethod
    def normalize_text(text: str) -> str:
        if not text: return ""
        text = re.sub(r'[^\w\s]', '', text.lower())
        return ' '.join(text.split())
    
    @staticmethod
    def extract_keywords(text: str) -> List[str]:
        stop_words = {'is', 'are', 'was', 'were', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = FAQMatcher.normalize_text(text).split()
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    @staticmethod
    def calculate_similarity(query: str, question: str) -> float:
        query_keywords = set(FAQMatcher.extract_keywords(query))
        question_keywords = set(FAQMatcher.extract_keywords(question))
        if not query_keywords or not question_keywords: return 0.0
        intersection = len(query_keywords.intersection(question_keywords))
        union = len(query_keywords.union(question_keywords))
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def is_vaibhav_related(query: str) -> bool:
        vaibhav_indicators = {
            'personal': ['vaibhav', 'awasthi', 'you', 'your', 'yourself', 'tell me about'],
            'skills': ['skill', 'technical', 'programming', 'language', 'technology'],
            'projects': ['project', 'work', 'built', 'created', 'developed'],
            'experience': ['intern', 'job', 'experience', 'company', 'employment'],
            'education': ['study', 'student', 'school', 'college', 'iit'],
            'contact': ['contact', 'email', 'phone', 'reach', 'github', 'linkedin']
        }
        query_lower = query.lower()
        for keywords in vaibhav_indicators.values():
            if any(keyword in query_lower for keyword in keywords): return True
        return False

    @staticmethod
    def find_best_match(query: str, faq_data: List[Dict]) -> Optional[Dict]:
        if not faq_data or not query: return None
        query_lower = query.lower()
        
        # Priority 1: Keyword/Pattern Match
        if "project" in query_lower and ("all" in query_lower or "list" in query_lower):
            for item in faq_data:
                if "all" in item['question'].lower() and "project" in item['question'].lower():
                    return item

        # Priority 2: Fuzzy/Similarity
        best_match = None
        best_score = 0
        for item in faq_data:
            score = FAQMatcher.calculate_similarity(query, item['question'])
            if score > best_score and score >= 0.3:
                best_score = score
                best_match = item
        return best_match

class GeminiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def generate_response(self, query: str, is_vaibhav_related: bool = False) -> str:
        global last_request_time
        current_time = time.time()
        if current_time - last_request_time < Config.RATE_LIMIT_DELAY:
            time.sleep(Config.RATE_LIMIT_DELAY)
        last_request_time = time.time()

        context = self._build_context(query, is_vaibhav_related)
        payload = {"contents": [{"parts": [{"text": context}]}]}
        
        # FIX: Constructing URL exactly as Google requires
        target_url = f"{Config.GEMINI_BASE_URL}?key={self.api_key}"
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = requests.post(target_url, json=payload, timeout=Config.REQUEST_TIMEOUT)
                response.raise_for_status()
                return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
            except Exception as e:
                logger.error(f"Gemini Attempt {attempt+1} failed: {e}")
                if attempt == Config.MAX_RETRIES - 1: return "I'm having trouble connecting to my brain. Please try again later."
                time.sleep(2)
        return "Sorry, I can't answer right now."

    def _build_context(self, query: str, is_vaibhav_related: bool) -> str:
        if is_vaibhav_related:
            return f"You are Vaibhav Awasthi's assistant. Vaibhav is an IIT Patna student (CGPA 8.43) skilled in Python, AI, and Data Science. Projects include Fraud Detection and Sign Language Detection. Answer this professionally: {query}"
        return f"Be a helpful AI assistant. Answer: {query}"

def load_faq_data() -> List[Dict]:
    try:
        if os.path.exists(Config.FAQ_FILE):
            with open(Config.FAQ_FILE, "r", encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"FAQ Load Error: {e}")
    return []

# Initialize
gemini_client = GeminiClient(Config.GEMINI_API_KEY)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message", "").strip()
        if not user_input: return jsonify({"error": "Empty message"}), 400
        
        # FAQ First
        faq_match = FAQMatcher.find_best_match(user_input, faq_data)
        if faq_match:
            return jsonify({"reply": faq_match["answer"], "source": "faq"})
        
        # Gemini Second
        is_vaibhav = FAQMatcher.is_vaibhav_related(user_input)
        reply = gemini_client.generate_response(user_input, is_vaibhav)
        return jsonify({"reply": reply, "source": "gemini"})
    except Exception as e:
        logger.error(f"Route Error: {e}")
        return jsonify({"reply": "Error occurred!"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "faq_items": len(faq_data), "gemini_ready": bool(Config.GEMINI_API_KEY)})

def initialize_app():
    global faq_data
    faq_data = load_faq_data()
    logger.info(f"Loaded {len(faq_data)} FAQ items.")

# Calling initialization before start
initialize_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)