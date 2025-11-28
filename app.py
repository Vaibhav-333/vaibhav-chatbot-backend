from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import difflib
import requests
import os
import re
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["*"])  # Configure CORS for all origins

# Configuration
class Config:
    # Gemini API Configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyBYtvnNZ67lWb4LRgBf24vFZq3IOg0yETc')  # Replace with your actual Gemini API key
    GEMINI_MODEL = 'gemini-2.5-flash'
    GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    
    # App Configuration
    FAQ_FILE = 'personal_faq.json'
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    RATE_LIMIT_DELAY = 1  # seconds between requests

# Global FAQ data storage
faq_data = []
last_request_time = 0

class FAQMatcher:
    """Enhanced FAQ matching with multiple strategies"""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for better matching"""
        if not text:
            return ""
        
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Remove extra spaces and strip
        text = ' '.join(text.split())
        return text
    
    @staticmethod
    def extract_keywords(text: str) -> List[str]:
        """Extract important keywords from text"""
        # Remove common stop words
        stop_words = {'is', 'are', 'was', 'were', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = FAQMatcher.normalize_text(text).split()
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    @staticmethod
    def calculate_similarity(query: str, question: str) -> float:
        """Calculate similarity between query and question"""
        query_keywords = set(FAQMatcher.extract_keywords(query))
        question_keywords = set(FAQMatcher.extract_keywords(question))
        
        if not query_keywords or not question_keywords:
            return 0.0
        
        # Jaccard similarity
        intersection = len(query_keywords.intersection(question_keywords))
        union = len(query_keywords.union(question_keywords))
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def is_vaibhav_related(query: str) -> bool:
        """Enhanced check if query is related to Vaibhav"""
        vaibhav_indicators = {
            'personal': ['vaibhav', 'awasthi', 'you', 'your', 'yourself', 'tell me about'],
            'skills': ['skill', 'technical', 'programming', 'language', 'technology', 'know'],
            'projects': ['project', 'work', 'built', 'created', 'developed', 'made'],
            'experience': ['intern', 'job', 'experience', 'company', 'employment', 'worked'],
            'education': ['study', 'student', 'school', 'college', 'iit', 'education', 'degree'],
            'location': ['live', 'location', 'where', 'kanpur', 'place', 'from'],
            'achievements': ['achievement', 'award', 'scholar', 'ntse', 'accomplishment'],
            'contact': ['contact', 'email', 'phone', 'reach', 'github', 'linkedin'],
            'future': ['future', 'plan', 'next', 'goal', 'pursue', 'want']
        }
        
        query_lower = query.lower()
        
        # Check for any category match
        for category, keywords in vaibhav_indicators.items():
            if any(keyword in query_lower for keyword in keywords):
                logger.info(f"Query classified as Vaibhav-related ({category}): {query}")
                return True
        
        return False
    
    @staticmethod
    def find_best_match(query: str, faq_data: List[Dict]) -> Optional[Dict]:
        """Find the best matching FAQ using multiple strategies"""
        if not faq_data or not query:
            return None
        
        logger.info(f"Searching FAQ for: '{query}'")
        
        # Strategy 1: Keyword-based pattern matching
        best_match = FAQMatcher._keyword_pattern_match(query, faq_data)
        if best_match:
            return best_match
        
        # Strategy 2: Similarity-based matching
        best_match = FAQMatcher._similarity_match(query, faq_data)
        if best_match:
            return best_match
        
        # Strategy 3: Fuzzy string matching
        best_match = FAQMatcher._fuzzy_match(query, faq_data)
        if best_match:
            return best_match
        
        logger.info("No FAQ match found")
        return None
    
    @staticmethod
    def _keyword_pattern_match(query: str, faq_data: List[Dict]) -> Optional[Dict]:
        """Keyword-based pattern matching"""
        patterns = {
            'identity': ['who is', 'about vaibhav', 'tell me about'],
            'skills': ['skill', 'technical', 'programming', 'technology'],
            'projects': ['project', 'built', 'created', 'developed'],
            'work': ['work', 'job', 'intern', 'experience', 'company'],
            'location': ['live', 'location', 'where', 'kanpur'],
            'achievements': ['achievement', 'award', 'scholar', 'accomplishment'],
            'certifications': ['certification', 'certificate', 'course'],
            'education': ['education', 'study', 'student', 'college', 'iit'],
            'future': ['future', 'plan', 'goal', 'pursue'],
            'contact': ['contact', 'email', 'phone', 'github', 'linkedin']
        }
        
        query_lower = query.lower()
        
        for pattern_type, keywords in patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                for item in faq_data:
                    question_lower = item['question'].lower()
                    
                    # Match based on pattern type
                    if (pattern_type == 'identity' and ('who is' in question_lower or 'vaibhav' in question_lower)) or \
                       (pattern_type == 'skills' and 'skill' in question_lower) or \
                       (pattern_type == 'projects' and 'project' in question_lower) or \
                       (pattern_type == 'work' and ('work' in question_lower or 'intern' in question_lower)) or \
                       (pattern_type == 'location' and ('live' in question_lower or 'where' in question_lower)) or \
                       (pattern_type == 'achievements' and 'achievement' in question_lower) or \
                       (pattern_type == 'certifications' and 'certification' in question_lower) or \
                       (pattern_type == 'education' and ('education' in question_lower or 'study' in question_lower)) or \
                       (pattern_type == 'future' and ('future' in question_lower or 'pursue' in question_lower)) or \
                       (pattern_type == 'contact' and any(c in question_lower for c in ['github', 'linkedin', 'email', 'phone'])):
                        
                        logger.info(f"Found keyword match ({pattern_type}): {item['question']}")
                        return item
        
        return None
    
    @staticmethod
    def _similarity_match(query: str, faq_data: List[Dict], threshold: float = 0.3) -> Optional[Dict]:
        """Similarity-based matching"""
        best_match = None
        best_score = 0
        
        for item in faq_data:
            score = FAQMatcher.calculate_similarity(query, item['question'])
            if score > best_score and score >= threshold:
                best_score = score
                best_match = item
        
        if best_match:
            logger.info(f"Found similarity match (score: {best_score:.2f}): {best_match['question']}")
        
        return best_match
    
    @staticmethod
    def _fuzzy_match(query: str, faq_data: List[Dict], cutoff: float = 0.6) -> Optional[Dict]:
        """Fuzzy string matching"""
        query_normalized = FAQMatcher.normalize_text(query)
        questions_normalized = [(FAQMatcher.normalize_text(item["question"]), item) for item in faq_data]
        
        matches = difflib.get_close_matches(
            query_normalized, 
            [q[0] for q in questions_normalized], 
            n=1, 
            cutoff=cutoff
        )
        
        if matches:
            for normalized_q, item in questions_normalized:
                if normalized_q == matches[0]:
                    logger.info(f"Found fuzzy match: {item['question']}")
                    return item
        
        return None

class GeminiClient:
    """Enhanced Gemini API client with error handling and rate limiting"""
    
    def __init__(self, api_key: str, model: str = None):
        self.api_key = api_key
        self.model = model if model else Config.GEMINI_MODEL
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
    
    def _rate_limit(self):
        """Simple rate limiting"""
        global last_request_time
        current_time = time.time()
        if current_time - last_request_time < Config.RATE_LIMIT_DELAY:
            time.sleep(Config.RATE_LIMIT_DELAY - (current_time - last_request_time))
        last_request_time = time.time()
    
    def generate_response(self, query: str, is_vaibhav_related: bool = False) -> str:
        """Generate response from Gemini with proper context"""
        self._rate_limit()
        
        context = self._build_context(query, is_vaibhav_related)
        
        payload = {
            "contents": [{"parts": [{"text": context}]}],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024,
                "stopSequences": []
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ]
        }
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Vaibhav-FAQ-Bot/1.0"
        }
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                logger.info(f"Requesting Gemini (attempt {attempt + 1}): {query[:50]}...")
                
                response = requests.post(
                    f"{self.base_url}?key={self.api_key}",
                    json=payload,
                    headers=headers,
                    timeout=Config.REQUEST_TIMEOUT
                )
                
                response.raise_for_status()
                response_data = response.json()
                
                return self._extract_response_text(response_data)
                
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP Error (attempt {attempt + 1}): {e}")
                if attempt == Config.MAX_RETRIES - 1:
                    return self._handle_http_error(e)
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except requests.exceptions.Timeout:
                logger.error(f"Timeout (attempt {attempt + 1})")
                if attempt == Config.MAX_RETRIES - 1:
                    return "I'm taking too long to respond. Please try again."
                time.sleep(2 ** attempt)
                
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}): {e}")
                if attempt == Config.MAX_RETRIES - 1:
                    return "I encountered an unexpected error. Please try again."
                time.sleep(2 ** attempt)
        
        return "I'm having trouble processing your request. Please try again later."
    
    def _build_context(self, query: str, is_vaibhav_related: bool) -> str:
        """Build appropriate context for the query"""
        if is_vaibhav_related:
            return f"""You are Vaibhav Awasthi's personal AI assistant. Provide accurate, helpful information about Vaibhav based on this profile:

PERSONAL INFORMATION:
- Name: Vaibhav Awasthi
- Location: Kanpur, Uttar Pradesh, India
- Current Status: B.S. Computer Science and Data Analytics student at IIT Patna
- CGPA: 8.26/10
- Previous Education: HSC (80%), SSC (87.8%) from Shivaji Group of Institutions

TECHNICAL SKILLS:
- Programming: Python, JavaScript
- Web Development: Flask, Streamlit, HTML, CSS
- Database: SQL, MySQL
- Data Science & AI: Computer Vision, Deep Learning, Machine Learning, NLP
- Tools: Power BI, Tableau, Git, OpenCV, TensorFlow, Pandas, NumPy
- Cloud: AWS (EC2, S3, RDS)

MAJOR PROJECTS:
1. Credit Card Fraud Detection - ML model using Logistic Regression and Random Forest
2. Sign Language Detection - Computer vision with Mediapipe + TensorFlow (90%+ accuracy)
3. AI Chatbot on Telegram - NLP integration with OpenAI GPT models
4. Cement Strength Prediction - Machine learning regression project
5. NLP-based Recommendation System - Content recommendation using natural language processing

PROFESSIONAL EXPERIENCE:
- Data Science Intern at Heleum: Developed fraud detection models, performed EDA, feature engineering
- Subject Matter Expert at Chegg: Provided 1000+ academic solutions in CS/DS/ML/DL
- Content Writer at Coursera: Created educational content for online courses

ACHIEVEMENTS & RECOGNITION:
- NTSE Scholar (National Talent Search Examination)
- ANTHE Scholar (Aakash National Talent Hunt Exam)
- Event Coordinator at IIT Patna for SPANDAN and sports competitions
- State-level debate competition winner

CERTIFICATIONS:
- Data Science Masters 2.0 (Physics Wallah)
- AWS Cloud Computing Fundamentals
- Excel Fundamentals for Finance (Corporate Finance Institute)

CONTACT INFORMATION:
- Email: www.awasthivaibhav333@gmail.com
- Phone: +91-639-626-3333
- GitHub: https://github.com/Vaibhav-333
- LinkedIn: https://www.linkedin.com/in/vaibhav-awasthi-17
- Twitter: https://x.com/httpsvaibhav

FUTURE ASPIRATIONS:
- Plans to pursue postgraduate degree abroad in Data Science
- Interested in advanced research in AI and Machine Learning

Answer this question about Vaibhav professionally and accurately: {query}"""
        else:
            return f"""You are a helpful AI assistant. Provide accurate, informative, and well-structured responses to general questions. Be concise but comprehensive.

Question: {query}"""
    
    def _extract_response_text(self, response_data: Dict) -> str:
        """Extract text from Gemini response"""
        try:
            if 'candidates' in response_data and len(response_data['candidates']) > 0:
                candidate = response_data['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    text = candidate['content']['parts'][0]['text']
                    logger.info(f"Gemini response received: {len(text)} characters")
                    return text.strip()
            
            logger.error(f"Unexpected response structure: {response_data}")
            return "I received an unexpected response format. Please try again."
            
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error parsing response: {e}")
            return "I had trouble understanding the response. Please try again."
    
    def _handle_http_error(self, error: requests.exceptions.HTTPError) -> str:
        """Handle HTTP errors with appropriate user messages"""
        status_code = error.response.status_code if error.response else 0
        
        error_messages = {
            400: "There was an issue with the request. Please try rephrasing your question.",
            401: "Authentication failed. Please check the API configuration.",
            403: "Access denied. The API key may be invalid or expired.",
            429: "Too many requests. Please wait a moment and try again.",
            500: "Server error occurred. Please try again later.",
            503: "Service temporarily unavailable. Please try again later."
        }
        
        return error_messages.get(status_code, f"API error occurred (Status: {status_code}). Please try again.")

def load_faq_data() -> List[Dict]:
    """Load FAQ data with comprehensive error handling"""
    try:
        if os.path.exists(Config.FAQ_FILE):
            with open(Config.FAQ_FILE, "r", encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} FAQ items from {Config.FAQ_FILE}")
            return data
        else:
            # Fallback data
            fallback_data = [
                {"question": "Who is Vaibhav Awasthi?", "answer": "Vaibhav Awasthi is a Computer Science and Data Analytics undergraduate student at IIT Patna with a CGPA of 8.26. He is from Kanpur, Uttar Pradesh, India."},
                {"question": "What are Vaibhav's technical skills?", "answer": "Vaibhav is skilled in Python, JavaScript, Flask, Streamlit, SQL, Computer Vision, Deep Learning, Power BI, Tableau, and Git. He also has experience with AWS cloud services."},
                {"question": "What are Vaibhav's major projects?", "answer": "His major projects include Credit Card Fraud Detection using ML, Sign Language Detection with 90%+ accuracy, AI Chatbot on Telegram, Cement Strength Prediction, and an NLP-based Recommendation System."},
                {"question": "Where has Vaibhav worked?", "answer": "Vaibhav has worked as a Data Science Intern at Heleum, Subject Matter Expert at Chegg (providing 1000+ solutions), and Content Writer at Coursera."},
                {"question": "What are Vaibhav's achievements?", "answer": "He is an NTSE Scholar, ANTHE Scholar, Event Coordinator at IIT Patna, and a state-level debate winner."},
                {"question": "What certifications does Vaibhav have?", "answer": "He has certifications in Data Science Masters from Physics Wallah, AWS Cloud Computing, and Excel for Finance from CFI."},
                {"question": "What is Vaibhav's educational background?", "answer": "He is currently pursuing B.S. in Computer Science and Data Analytics at IIT Patna with 8.26 CGPA. He completed HSC (80%) and SSC (87.8%) from Shivaji Group of Institutions."},
                {"question": "What are Vaibhav's future plans?", "answer": "Vaibhav plans to pursue a postgraduate degree abroad in Data Science after graduation."},
                {"question": "How to contact Vaibhav?", "answer": "Email: www.awasthivaibhav333@gmail.com, Phone: +91-639-626-3333, GitHub: https://github.com/Vaibhav-333, LinkedIn: https://www.linkedin.com/in/vaibhav-awasthi-17"}
            ]
            
            logger.info(f"Using fallback FAQ data with {len(fallback_data)} items")
            return fallback_data
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in FAQ file: {e}")
        return []
    except Exception as e:
        logger.error(f"Error loading FAQ data: {e}")
        return []

# Initialize components
gemini_client = GeminiClient(Config.GEMINI_API_KEY)

# Routes
@app.route("/chat", methods=["POST"])
def chat():
    """Main chat endpoint"""
    try:
        # Validate request
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Invalid request format. Please include 'message' field."}), 400
        
        user_input = data.get("message", "").strip()
        if not user_input:
            return jsonify({"error": "Message cannot be empty"}), 400
        
        if len(user_input) > 1000:
            return jsonify({"error": "Message too long. Please keep it under 1000 characters."}), 400
        
        logger.info(f"Processing query: '{user_input}'")
        
        # Step 1: Try FAQ matching
        faq_match = FAQMatcher.find_best_match(user_input, faq_data)
        
        if faq_match:
            logger.info("Returning FAQ answer")
            return jsonify({
                "reply": faq_match["answer"],
                "source": "faq",
                "confidence": "high",
                "timestamp": datetime.now().isoformat()
            })
        
        # Step 2: Check if Vaibhav-related for context
        is_vaibhav_related = FAQMatcher.is_vaibhav_related(user_input)
        
        # Step 3: Get Gemini response
        gemini_response = gemini_client.generate_response(user_input, is_vaibhav_related)
        
        return jsonify({
            "reply": gemini_response,
            "source": "gemini_vaibhav" if is_vaibhav_related else "gemini_general",
            "confidence": "medium",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return jsonify({"reply": "Sorry, I encountered an error. Please try again!"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Vaibhav's Personal AI Assistant",
        "version": "2.0",
        "timestamp": datetime.now().isoformat(),
        "faq_items": len(faq_data),
        "gemini_configured": bool(Config.GEMINI_API_KEY)
    })

@app.route("/stats", methods=["GET"])
def get_stats():
    """Get system statistics"""
    faq_topics = {}
    for item in faq_data:
        question_lower = item['question'].lower()
        if 'skill' in question_lower:
            faq_topics['Technical Skills'] = faq_topics.get('Technical Skills', 0) + 1
        elif 'project' in question_lower:
            faq_topics['Projects'] = faq_topics.get('Projects', 0) + 1
        elif 'work' in question_lower or 'intern' in question_lower:
            faq_topics['Experience'] = faq_topics.get('Experience', 0) + 1
        elif 'achievement' in question_lower:
            faq_topics['Achievements'] = faq_topics.get('Achievements', 0) + 1
        elif 'contact' in question_lower or any(c in question_lower for c in ['github', 'linkedin', 'email', 'phone']):
            faq_topics['Contact'] = faq_topics.get('Contact', 0) + 1
        else:
            faq_topics['General'] = faq_topics.get('General', 0) + 1
    
    return jsonify({
        "faq_statistics": {
            "total_items": len(faq_data),
            "topics": faq_topics,
            "sample_questions": [item["question"] for item in faq_data[:5]]
        },
        "configuration": {
            "gemini_model": Config.GEMINI_MODEL,
            "api_configured": bool(Config.GEMINI_API_KEY),
            "timeout": Config.REQUEST_TIMEOUT,
            "max_retries": Config.MAX_RETRIES
        }
    })

@app.route("/test", methods=["POST"])
def test_endpoint():
    """Test endpoint for debugging"""
    try:
        data = request.get_json()
        query = data.get("query", "What is 2+2?")
        
        # Test FAQ matching
        faq_match = FAQMatcher.find_best_match(query, faq_data)
        is_vaibhav_related = FAQMatcher.is_vaibhav_related(query)
        
        # Test Gemini (simple query)
        gemini_test = gemini_client.generate_response("What is 2+2?", False)
        
        return jsonify({
            "test_query": query,
            "faq_match": faq_match["question"] if faq_match else None,
            "is_vaibhav_related": is_vaibhav_related,
            "gemini_test": gemini_test,
            "all_systems": "operational" if faq_data and gemini_test else "issues_detected"
        })
        
    except Exception as e:
        logger.error(f"Test endpoint error: {e}")
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

# Initialize application
def initialize_app():
    """Initialize the application"""
    global faq_data
    
    logger.info("🚀 Starting Vaibhav's Personal AI Assistant v2.0")
    
    # Load FAQ data
    faq_data = load_faq_data()
    logger.info(f"📚 FAQ system ready with {len(faq_data)} items")
    
    # Test Gemini connection
    try:
        test_response = gemini_client.generate_response("Test connection", False)
        logger.info("🤖 Gemini API connection successful")
    except Exception as e:
        logger.warning(f"🚨 Gemini API connection issue: {e}")
    
    logger.info("✅ Application initialized successfully")
    logger.info("💡 Ready to handle personal FAQ and general queries")

if __name__ == "__main__":
    initialize_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
