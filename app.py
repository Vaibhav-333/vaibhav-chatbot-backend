from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import requests
import os
import re
import logging
import sys
from typing import Optional, Dict, List
import time
from dotenv import load_dotenv

load_dotenv()

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

class Config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GEMINI_MODEL = 'gemini-1.5-flash'
    GEMINI_BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

    # FIX 1: __file__ instead of __name__
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FAQ_FILE = os.path.join(BASE_DIR, 'personal_faq.json')

    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    RATE_LIMIT_DELAY = 1

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

        if "project" in query_lower and ("all" in query_lower or "list" in query_lower):
            for item in faq_data:
                if "all" in item['question'].lower() and "project" in item['question'].lower():
                    return item

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
        # FIX 2: Check if API key exists before attempting call
        if not self.api_key:
            logger.error("GEMINI_API_KEY is not set! Check your Render environment variables.")
            return "My AI brain isn't configured yet. Please contact Vaibhav directly at awasthivaibhav333@gmail.com"

        global last_request_time
        current_time = time.time()
        if current_time - last_request_time < Config.RATE_LIMIT_DELAY:
            time.sleep(Config.RATE_LIMIT_DELAY)
        last_request_time = time.time()

        context = self._build_context(query, is_vaibhav_related)
        payload = {"contents": [{"parts": [{"text": context}]}]}
        target_url = f"{Config.GEMINI_BASE_URL}?key={self.api_key}"

        for attempt in range(Config.MAX_RETRIES):
            try:
                response = requests.post(target_url, json=payload, timeout=Config.REQUEST_TIMEOUT)

                # FIX 3: Log the actual error from Gemini instead of swallowing it
                if not response.ok:
                    logger.error(f"Gemini API error {response.status_code}: {response.text}")
                    response.raise_for_status()

                return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()

            except requests.exceptions.Timeout:
                logger.error(f"Gemini timeout on attempt {attempt + 1}")
                if attempt == Config.MAX_RETRIES - 1:
                    return "Request timed out. Please try again in a moment!"
            except Exception as e:
                logger.error(f"Gemini Attempt {attempt + 1} failed: {e}")
                if attempt == Config.MAX_RETRIES - 1:
                    return "I'm having trouble connecting right now. Please try again or contact Vaibhav directly!"
                time.sleep(2)

        return "Sorry, I can't answer right now."

    def _build_context(self, query: str, is_vaibhav_related: bool) -> str:
        if is_vaibhav_related:
            return (
                "You are Vaibhav Awasthi's personal AI assistant on his portfolio website. "
                "Vaibhav is a B.S. Computer Science & Data Analytics student at IIT Patna (CGPA 8.43, batch 2023-2027). "
                "He is skilled in Python, PyTorch, TensorFlow, OpenCV, Machine Learning, Deep Learning, NLP, SQL, Power BI. "
                "His projects include: RLHF pipeline for educational LLMs (IIT-GN), "
                "Telegram AI Chatbot (Python + OpenAI GPT), Sign Language Detection (OpenCV + Mediapipe, 90%+ accuracy), "
                "and Fraud Detection ML models. "
                "He has interned at Heleum (Data Science), Good Enough Energy (ML), and works as Subject Matter Expert at Chegg. "
                "He is an NTSE Scholar (2020, CRL under 500) and has organized IIT Patna events. "
                f"Answer this question about Vaibhav professionally and concisely: {query}"
            )
        return f"You are a helpful AI assistant. Answer this question concisely and helpfully: {query}"


def load_faq_data() -> List[Dict]:
    try:
        if os.path.exists(Config.FAQ_FILE):
            with open(Config.FAQ_FILE, "r", encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Successfully loaded {len(data)} FAQ items from {Config.FAQ_FILE}")
                return data
        else:
            logger.warning(f"FAQ file not found at: {Config.FAQ_FILE}")
    except Exception as e:
        logger.error(f"FAQ Load Error: {e}")
    return []


# Initialize
gemini_client = GeminiClient(Config.GEMINI_API_KEY)


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body"}), 400

        user_input = data.get("message", "").strip()
        if not user_input:
            return jsonify({"error": "Empty message"}), 400

        # FAQ first
        faq_match = FAQMatcher.find_best_match(user_input, faq_data)
        if faq_match:
            logger.info(f"FAQ match found for: {user_input[:50]}")
            return jsonify({"reply": faq_match["answer"], "source": "faq"})

        # Gemini second
        is_vaibhav = FAQMatcher.is_vaibhav_related(user_input)
        reply = gemini_client.generate_response(user_input, is_vaibhav)
        return jsonify({"reply": reply, "source": "gemini"})

    except Exception as e:
        logger.error(f"Route Error: {e}")
        return jsonify({"reply": "An error occurred. Please try again!"}), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "faq_items": len(faq_data),
        # FIX 4: Don't expose whether key exists in public endpoint
        "gemini_ready": bool(Config.GEMINI_API_KEY),
        "faq_path": Config.FAQ_FILE
    })


def initialize_app():
    global faq_data
    faq_data = load_faq_data()
    if not Config.GEMINI_API_KEY:
        logger.warning("⚠️  GEMINI_API_KEY is not set! Set it in Render dashboard → Environment tab.")
    else:
        logger.info("✅ Gemini API key loaded successfully.")


initialize_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)