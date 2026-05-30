"""
app.py
======
Flask web server for the Resume Classification System.
Run: python app.py
"""

import os
import re
import pickle
import io
import json
import traceback

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── NLTK setup ───────────────────────────────────────────────────────────────
for pkg in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
    nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ── Flask app ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
CORS(app)

# ── Load model artifacts ──────────────────────────────────────────────────────
MODEL_PATH = os.path.join(BASE_DIR, "resume_classifier.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

pipeline = None
label_encoder = None

def load_model():
    global pipeline, label_encoder
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        with open(MODEL_PATH, "rb") as f:
            pipeline = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)
        print("✓ Model loaded successfully")
        return True
    return False


# ── Text extraction ───────────────────────────────────────────────────────────
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception:
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            return "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        except Exception:
            return ""


def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception:
        return ""


def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in ("docx", "doc"):
        return extract_text_from_docx(file_bytes)
    else:
        try:
            return file_bytes.decode("utf-8")
        except Exception:
            return file_bytes.decode("latin-1", errors="ignore")


# ── Text cleaning (same as training) ─────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in STOP_WORDS and len(t) > 2
    ]
    return " ".join(tokens)


# ── Category metadata ─────────────────────────────────────────────────────────
CATEGORY_INFO = {
    "Data Science": {
        "icon": "📊",
        "color": "#6366f1",
        "skills": ["Python", "Pandas", "Scikit-learn", "SQL", "Statistics", "Visualization"],
        "description": "Analyzes complex data to extract actionable insights and build predictive models.",
    },
    "Machine Learning": {
        "icon": "🤖",
        "color": "#8b5cf6",
        "skills": ["TensorFlow", "PyTorch", "Deep Learning", "MLOps", "Computer Vision", "NLP"],
        "description": "Designs and deploys intelligent ML systems from research to production.",
    },
    "Web Development": {
        "icon": "🌐",
        "color": "#06b6d4",
        "skills": ["React", "Node.js", "TypeScript", "REST APIs", "CSS", "Databases"],
        "description": "Builds responsive, high-performance web applications end-to-end.",
    },
    "Java Developer": {
        "icon": "☕",
        "color": "#f59e0b",
        "skills": ["Spring Boot", "Microservices", "JVM", "Hibernate", "Maven", "CI/CD"],
        "description": "Develops robust enterprise-grade applications using the Java ecosystem.",
    },
    "Python Developer": {
        "icon": "🐍",
        "color": "#10b981",
        "skills": ["Django", "FastAPI", "Asyncio", "Docker", "PostgreSQL", "Testing"],
        "description": "Builds scalable backend services and automation tools with Python.",
    },
    "DevOps Engineer": {
        "icon": "⚙️",
        "color": "#ef4444",
        "skills": ["Kubernetes", "Terraform", "CI/CD", "AWS", "Docker", "Monitoring"],
        "description": "Bridges development and operations for faster, reliable software delivery.",
    },
    "HR": {
        "icon": "👥",
        "color": "#ec4899",
        "skills": ["Recruitment", "HRIS", "Payroll", "Compliance", "L&D", "Talent Management"],
        "description": "Manages people strategy, culture, and workforce development.",
    },
    "Advocate": {
        "icon": "⚖️",
        "color": "#64748b",
        "skills": ["Litigation", "Contract Law", "Legal Research", "Compliance", "IP Law"],
        "description": "Provides expert legal counsel and representation.",
    },
    "Arts": {
        "icon": "🎨",
        "color": "#f97316",
        "skills": ["Adobe Suite", "UI/UX", "Branding", "Typography", "Motion Graphics"],
        "description": "Creates compelling visual communications and artistic works.",
    },
    "Mechanical Engineer": {
        "icon": "🔧",
        "color": "#84cc16",
        "skills": ["SolidWorks", "AutoCAD", "FEA", "Thermodynamics", "Manufacturing"],
        "description": "Designs and develops mechanical systems and products.",
    },
    "Sales": {
        "icon": "💼",
        "color": "#14b8a6",
        "skills": ["CRM", "B2B", "Lead Generation", "Negotiation", "Pipeline Management"],
        "description": "Drives revenue growth through strategic client relationships.",
    },
    "Health and Fitness": {
        "icon": "💪",
        "color": "#22c55e",
        "skills": ["Fitness Training", "Nutrition", "Wellness", "Patient Care", "Assessment"],
        "description": "Promotes health, fitness, and wellbeing for individuals.",
    },
    "Civil Engineer": {
        "icon": "🏗️",
        "color": "#a16207",
        "skills": ["AutoCAD", "Structural Design", "Project Management", "Surveying", "Safety"],
        "description": "Plans and oversees construction of infrastructure and facilities.",
    },
    "Business Analyst": {
        "icon": "📈",
        "color": "#7c3aed",
        "skills": ["Requirements Analysis", "SQL", "Tableau", "BPMN", "Agile", "UML"],
        "description": "Bridges business needs and technical solutions through data-driven analysis.",
    },
    "Digital Marketing": {
        "icon": "📣",
        "color": "#db2777",
        "skills": ["SEO/SEM", "Google Ads", "Content Marketing", "Analytics", "Social Media"],
        "description": "Drives brand awareness and conversions through digital channels.",
    },
    "Testing": {
        "icon": "🧪",
        "color": "#0ea5e9",
        "skills": ["Selenium", "Pytest", "API Testing", "CI/CD", "BDD", "Test Planning"],
        "description": "Ensures software quality through comprehensive automated and manual testing.",
    },
    "PMO": {
        "icon": "📋",
        "color": "#6b7280",
        "skills": ["Agile", "Scrum", "Risk Management", "Jira", "Stakeholder Management"],
        "description": "Leads cross-functional projects on time, scope, and budget.",
    },
    "ETL Developer": {
        "icon": "🔄",
        "color": "#2563eb",
        "skills": ["Apache Spark", "Airflow", "dbt", "SQL", "Data Warehouse", "Kafka"],
        "description": "Designs and maintains robust data pipelines and warehousing solutions.",
    },
    "Operations Manager": {
        "icon": "🏭",
        "color": "#059669",
        "skills": ["Supply Chain", "ERP/SAP", "Process Optimization", "KPI", "Lean"],
        "description": "Optimizes business operations for efficiency and cost reduction.",
    },
    "Network Security Engineer": {
        "icon": "🔒",
        "color": "#dc2626",
        "skills": ["Firewall", "Penetration Testing", "SIEM", "Zero Trust", "Compliance"],
        "description": "Secures organizational infrastructure against cyber threats.",
    },
    "Blockchain": {
        "icon": "⛓️",
        "color": "#7c2d12",
        "skills": ["Solidity", "Ethereum", "Smart Contracts", "Web3", "DeFi", "IPFS"],
        "description": "Builds decentralized applications on blockchain platforms.",
    },
    "Database": {
        "icon": "🗄️",
        "color": "#1e40af",
        "skills": ["PostgreSQL", "MySQL", "MongoDB", "Performance Tuning", "Replication"],
        "description": "Architects, optimizes, and administers database systems.",
    },
}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "model_loaded": pipeline is not None,
        "categories": list(CATEGORY_INFO.keys()) if pipeline else [],
    })


@app.route("/api/classify", methods=["POST"])
def classify():
    if pipeline is None:
        return jsonify({"error": "Model not loaded. Run python train_model.py first."}), 503

    # Handle both file upload and raw text
    raw_text = ""
    filename = "text_input"

    if "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        filename = file.filename
        file_bytes = file.read()
        raw_text = extract_text(file_bytes, filename)
    elif request.is_json:
        data = request.get_json()
        raw_text = data.get("text", "")
    else:
        raw_text = request.form.get("text", "")

    if not raw_text.strip():
        return jsonify({"error": "Could not extract text. Please try a different file."}), 400

    # Clean and predict
    cleaned = clean_text(raw_text)
    if not cleaned.strip():
        return jsonify({"error": "No meaningful text found after preprocessing."}), 400

    # Temperature-scaled softmax on SVM decision scores
    # T < 1.0 sharpens the distribution → winner gets high confidence
    import numpy as np

    decision_scores = pipeline.decision_function([cleaned])[0]
    predicted_idx = int(decision_scores.argmax())
    T = 0.15  # low temperature = sharper, more confident predictions
    shifted = decision_scores - decision_scores.max()
    exp_scores = np.exp(shifted / T)
    probs = (exp_scores / exp_scores.sum() * 100).tolist()

    predicted_label = label_encoder.classes_[predicted_idx]

    # Top 5 predictions
    top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:5]
    top_predictions = [
        {
            "category": label_encoder.classes_[i],
            "confidence": round(probs[i], 2),
            "info": CATEGORY_INFO.get(label_encoder.classes_[i], {}),
        }
        for i in top_indices
    ]

    # Word count stats
    word_count = len(raw_text.split())
    token_count = len(cleaned.split())

    response = {
        "predicted_category": predicted_label,
        "confidence": round(probs[predicted_idx], 2),
        "info": CATEGORY_INFO.get(predicted_label, {}),
        "top_predictions": top_predictions,
        "stats": {
            "word_count": word_count,
            "token_count": token_count,
            "filename": filename,
            "char_count": len(raw_text),
        },
    }
    return jsonify(response)


@app.route("/api/categories")
def categories():
    return jsonify(CATEGORY_INFO)


if __name__ == "__main__":
    load_model()
    app.run(debug=True, port=5050, host="0.0.0.0")
