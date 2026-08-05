from dotenv import load_dotenv
load_dotenv()
import os
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from supabase import create_client, Client
import json
from datetime import datetime
import re
import hashlib
from io import BytesIO
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import spacy
import warnings
warnings.filterwarnings('ignore')

# ========== RAG & GROQ IMPORTS ==========
import ollama
from sentence_transformers import SentenceTransformer
import chromadb
import concurrent.futures
import groq

# ========== APP SETUP ==========
app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://mqjyvqhjhtrjymcriuvx.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "sb_publishable_FjqLD74vzT-Zj4-tg53ETA_XdECmPz2")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Groq client
groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

# ========== GLOBAL STATE ==========
class PaperStore:
    def __init__(self):
        self.text = ""
        self.filename = ""
        self.chunks = []
        self.collection = None
        self.embedding_model = None
        self.text_quality = {"valid": False, "issues": []}
        self.metadata = {}
    
    def reset(self):
        self.text = ""
        self.filename = ""
        self.chunks = []
        self.collection = None
        self.text_quality = {"valid": False, "issues": []}
        self.metadata = {}
    
    def set_paper(self, text, filename):
        self.text = text
        self.filename = filename
        self.metadata = {
            "word_count": len(text.split()),
            "char_count": len(text),
            "line_count": len(text.split('\n')),
            "uploaded_at": datetime.now().isoformat()
        }

PAPER_STORE = PaperStore()

# ========== NLP SETUP ==========
def load_nlp():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], capture_output=True)
        nlp = spacy.load("en_core_web_sm")
    return nlp

def load_nltk_data():
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords')
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except:
                pass
    return stopwords.words('english')

nlp = load_nlp()
stop_words = load_nltk_data()

KEEP_TOKENS = {'not', 'no', 'never', 'before', 'after', 'why', 'must', 'should', 'could', 'would'}
STOPWORDS_FILTERED = [w for w in stop_words if w not in KEEP_TOKENS]

# ========== ULTRA-AGGRESSIVE TEXT CLEANER ==========
import pdfplumber

class UltraAggressiveTextCleaner:
    def __init__(self):
        self.common_words = {
            'the','a','an','and','or','but','in','on','at','to','for',
            'of','with','from','by','as','is','are','was','were','be',
            'have','has','had','do','does','did','can','could','will',
            'would','should','may','might','must','shall','this','that',
            'these','those','i','you','he','she','it','we','they',
            'what','which','who','when','where','why','how','all','each',
            'every','both','neither','either','no','not','only','same',
            'such','so','than','then','now','here','there','about',
            'also','more','most','less','least','very','much','many',
            'few','some','any','several','another','other',
            'our','your','his','her','its','their','my','me','him',
            'us','them',
            'research','study','paper','analysis','results','method',
            'approach','theory','model','system','data','show',
            'demonstrate','found','suggest','propose','develop','improve',
            'achieve','performance','effectiveness','efficiency','impact',
            'project','management','success','practices','benefits',
            'framework','governance','implementation','conclusion',
            'abstract','introduction','discussion','evaluation'
        }
    
    def extract_from_pdf(self, file_bytes, max_pages=50):
        try:
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                full_text = []
                pages_to_read = min(len(pdf.pages), max_pages)
                for i, page in enumerate(pdf.pages[:pages_to_read]):
                    text = page.extract_text()
                    if text and len(text.strip()) > 20:
                        full_text.append(text)
                if not full_text:
                    raise Exception("No text extracted")
                return "\n\n".join(full_text)
        except Exception as e:
            raise Exception(f"PDF extraction failed: {e}")
    
    def clean_aggressively(self, text):
        text = self._step1_preprocess(text)
        text = self._step2_split_words(text)
        text = self._step3_word_boundaries(text)
        text = self._step4_punctuation(text)
        text = self._step5_whitespace(text)
        text = self._step6_junk_removal(text)
        text = self._step7_sections(text)
        return text
    
    def _step1_preprocess(self, text):
        text = re.sub(r'[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000]', ' ', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'([.!?,:;])([A-Za-z0-9])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z0-9])([.!?,:;])', r'\1\2', text)
        return text
    
    def _step2_split_words(self, text):
        def smart_split(word):
            if len(word) < 4 or ' ' in word:
                return word
            word_lower = word.lower()
            for i in range(2, len(word) - 1):
                left = word_lower[:i]
                right = word_lower[i:]
                if left in self.common_words:
                    right_split = smart_split(right)
                    return f"{word[:i]} {right_split}"
            return word
        pattern = r'\b[a-z]{3,}(?=[a-z]{2,})[a-z]*\b'
        text = re.sub(pattern, lambda m: smart_split(m.group(0)), text, flags=re.IGNORECASE)
        return text
    
    def _step3_word_boundaries(self, text):
        fixes = [
            (r'([a-z])and([A-Z])', r'\1 and \2'),
            (r'([a-z])or([A-Z])', r'\1 or \2'),
            (r'([a-z])the([A-Z])', r'\1 the \2'),
            (r'([a-z])in([A-Z])', r'\1 in \2'),
            (r'([a-z])is([A-Z])', r'\1 is \2'),
            (r'([a-z])are([A-Z])', r'\1 are \2'),
            (r'([a-z])was([A-Z])', r'\1 was \2'),
            (r'([a-z])were([A-Z])', r'\1 were \2'),
            (r'([a-z])be([A-Z])', r'\1 be \2'),
            (r'([a-z])et([a-z])', r'\1 et \2'),
        ]
        for pattern, replacement in fixes:
            text = re.sub(pattern, replacement, text, flags=re.I)
        return text
    
    def _step4_punctuation(self, text):
        text = re.sub(r'([.!?,:;])([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'\s+([.!?,:;)])', r'\1', text)
        return text
    
    def _step5_whitespace(self, text):
        text = text.replace('\t', ' ')
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\n +', '\n', text)
        text = re.sub(r' +\n', '\n', text)
        return text.strip()
    
    def _step6_junk_removal(self, text):
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if not stripped or len(stripped) < 3:
                continue
            if re.match(r'^[\d\-–—\s]*$', stripped):
                continue
            if not re.search(r'[a-zA-Z]', stripped):
                continue
            alpha = len(re.findall(r'[a-zA-Z0-9]', stripped))
            if alpha < len(stripped) * 0.6:
                continue
            if re.search(r'([^a-zA-Z0-9 ])\1{3,}', stripped):
                continue
            cleaned.append(line)
        return '\n'.join(cleaned)
    
    def _step7_sections(self, text):
        text = re.sub(
            r'\n\s*(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography|APPENDIX|Appendix|ACKNOWLEDGEMENTS).*',
            '',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )
        return text

# ========== TEXT VALIDATOR ==========
class TextValidator:
    @staticmethod
    def assess_quality(text):
        issues = []
        word_count = len(text.split())
        if word_count < 100:
            issues.append("Text too short (< 100 words)")
        english_words = sum(1 for w in text.split() if len(w) > 2)
        if english_words < word_count * 0.7:
            issues.append("Low English word ratio")
        avg_word_len = sum(len(w) for w in text.split()) / max(len(text.split()), 1)
        if avg_word_len < 3 or avg_word_len > 12:
            issues.append(f"Unusual average word length: {avg_word_len:.1f}")
        sentences = re.split(r'[.!?]', text)
        if len(sentences) < 3:
            issues.append("Very few sentences detected")
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "word_count": word_count,
            "quality_score": max(0, 100 - len(issues) * 20)
        }

# ========== SENTENCE SPLITTING ==========
def split_sentences(text):
    abbreviations = [
        r'\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|Ph\.D|U\.S|U\.K|U\.N|NATO|NASA|WHO)\b',
        r'\b(?:e\.g|i\.e|etc|vs|viz)\b'
    ]
    text_marked = text
    for abbr_pattern in abbreviations:
        def replace_abbr(match):
            return match.group(0).replace('.', '___DOT___')
        text_marked = re.sub(abbr_pattern, replace_abbr, text_marked, flags=re.IGNORECASE)
    sentences = re.split(r'(?<=[.!?])\s+', text_marked)
    sentences = [s.replace('___DOT___', '.').strip() for s in sentences]
    sentences = [s for s in sentences if s and len(s.split()) >= 5 and len(s) > 15]
    return sentences

# ========== SUMMARIZATION ENGINE ==========
class SummarizationEngine:
    def __init__(self):
        self.cleaner = UltraAggressiveTextCleaner()

    def _filter_sentences(self, sentences):
        filtered = []
        for sent in sentences:
            if len(re.findall(r'[a-zA-Z]', sent)) / max(len(sent), 1) < 0.5:
                continue
            if re.match(r'^\d+\.\s+[A-Z]', sent) or re.match(r'^(?:Figure|Fig|Table)\s+\d+', sent, re.I) or 'et al' in sent.lower():
                continue
            if len(sent.split()) > 100:
                continue
            filtered.append(sent)
        return filtered

    def textrank_summarize(self, text, num_sentences=8):
        sentences = split_sentences(text)
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        filtered = self._filter_sentences(sentences)
        if len(filtered) < num_sentences:
            filtered = sentences
        sentences = filtered[:150]
        # Scoring
        section_keywords = {
            'abstract': r'\b(abstract|overview|summary|purpose)\b',
            'results': r'\b(results|findings|outcomes|achieved|demonstrated|showed|revealed|found|indicate)\b',
            'conclusion': r'\b(conclusion|conclude|concluding|implications|summary)\b',
            'discussion': r'\b(discussion|discuss|suggests|propose)\b',
            'methods': r'\b(method|methods|approach|technique|proposed)\b'
        }
        importance_keywords = {
            'significant': 40, 'novel': 35, 'outperform': 35,
            'improve': 30, 'achieve': 30, 'demonstrate': 30,
            'superior': 35, 'effective': 25, 'efficient': 25,
            'performance': 25, 'empirical': 25, 'evidence': 25,
            'breakthrough': 40, 'unprecedented': 40
        }
        sentence_scores = {}
        for i, sent in enumerate(sentences):
            score = 0
            sent_lower = sent.lower()
            for section, pattern in section_keywords.items():
                if re.search(pattern, sent_lower):
                    score += 25 if section in ['abstract', 'results', 'conclusion'] else 15
                    break
            for keyword, boost in importance_keywords.items():
                if keyword in sent_lower:
                    score += boost
            score += (1.0 - i/len(sentences)) * 10
            if re.search(r'^\d+\.\s*[A-Z]', sent):
                score -= 30
            sentence_scores[i] = max(score, 0)

        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=500, ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(sentences)
            similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
            similarity_matrix = similarity_matrix / (np.max(similarity_matrix) + 1e-8)
        except:
            similarity_matrix = np.eye(len(sentences))

        scores = np.ones(len(sentences)) / len(sentences)
        damping = 0.85
        for _ in range(30):
            new_scores = np.zeros(len(sentences))
            for i in range(len(sentences)):
                weighted = sum(similarity_matrix[i][j] * scores[j] for j in range(len(sentences)) if i != j)
                new_scores[i] = weighted / (np.sum(similarity_matrix[:, i]) + 1e-8)
            scores = damping * new_scores + (1 - damping) / len(sentences)

        boost_scores = np.array([sentence_scores.get(i, 0) for i in range(len(sentences))])
        if np.max(boost_scores) > 0:
            boost_scores = boost_scores / np.max(boost_scores)
        scores = scores / (np.max(scores) + 1e-8)
        final_scores = 0.65 * scores + 0.35 * boost_scores

        top_indices = np.argsort(final_scores)[-num_sentences*2:][::-1]
        selected = []
        for idx in top_indices:
            sent = sentences[idx]
            is_dup = any(
                len(set(sent.split()) & set(s.split())) / len(set(sent.split()) | set(s.split())) > 0.65
                for s in selected
            )
            if not is_dup:
                selected.append(sent)
            if len(selected) >= num_sentences:
                break
        return ' '.join(selected) if selected else self.tfidf_summarize(text, num_sentences)

    def tfidf_summarize(self, text, num_sentences=5):
        sentences = split_sentences(text)
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
            tfidf = vectorizer.fit_transform(sentences)
            scores = tfidf.sum(axis=1).A1
            ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:num_sentences]
            return ' '.join(sentences[i] for i in sorted(ranked))
        except:
            return ' '.join(sentences[:num_sentences])

    def frequency_summarize(self, text, num_sentences=5):
        sentences = split_sentences(text)
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        filtered = self._filter_sentences(sentences)
        if len(filtered) < num_sentences:
            filtered = sentences
        extended_stops = set(stop_words)
        extended_stops.update({'et','al','paper','study','research','result','show','find','propose','use','would','could'})
        words = [w for w in re.findall(r'\b[a-zA-Z]+\b', text.lower()) if w not in extended_stops and len(w) > 3]
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=200)
            tfidf = vectorizer.fit_transform(filtered)
            word_weights = dict(zip(vectorizer.get_feature_names_out(), tfidf.sum(axis=0).A1))
        except:
            from collections import Counter
            word_weights = dict(Counter(words).most_common(200))
        importance_words = {'accuracy':50,'performance':40,'improvement':45,'efficient':40,'effective':40,
                            'superior':45,'significant':40,'novel':45,'outperform':45}
        scores = []
        for sent in filtered:
            sent_words = [w for w in re.findall(r'\b[a-zA-Z]+\b', sent.lower()) if len(w) > 3]
            score = sum(word_weights.get(w, 0) for w in sent_words) / (len(sent_words) + 1)
            for word, boost in importance_words.items():
                if word in sent.lower():
                    score += boost
            if re.search(r'\b(demonstrate|show|indicate|reveal|conclude|suggest)\b', sent.lower()):
                score *= 1.3
            if re.search(r'\bet al\.\b', sent, re.I):
                score *= 0.6
            scores.append((sent, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        selected = []
        for sent, _ in scores:
            is_dup = any(
                len(set(sent.split()) & set(s.split())) / len(set(sent.split()) | set(s.split())) > 0.65
                for s in selected
            )
            if not is_dup:
                selected.append(sent)
            if len(selected) >= num_sentences:
                break
        return ' '.join(selected) if selected else self.tfidf_summarize(text, num_sentences)

summarizer = SummarizationEngine()

# ========== EMBEDDING MODEL ==========
_embedding_model = None
def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model

def build_index(text):
    words = text.split()
    chunks = []
    chunk_size = 200
    overlap = 30
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i+chunk_size])
        if len(chunk.split()) > 20:
            chunks.append(chunk)
    if not chunks:
        return None
    try:
        model = get_embedding_model()
        embeddings = model.encode(chunks, batch_size=32)
        client = chromadb.Client()
        collection_name = f"paper_chunks_{hashlib.md5(text[:1000].encode()).hexdigest()[:8]}"
        try:
            client.delete_collection(collection_name)
        except:
            pass
        collection = client.create_collection(name=collection_name)
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            collection.add(documents=[chunk], embeddings=[emb.tolist()], ids=[str(i)])
        PAPER_STORE.chunks = chunks
        PAPER_STORE.collection = collection
        return collection
    except Exception as e:
        print(f"Index building failed: {e}")
        return None

# ========== PAPER STRUCTURE ==========
def extract_paper_structure(text):
    sections = {
        'title': '', 'abstract': '', 'introduction': '',
        'methodology': '', 'results': '', 'conclusion': '',
        'references': '', 'other': ''
    }
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        sections['title'] = lines[0][:200]
    section_markers = {
        'abstract': ['abstract', 'summary'],
        'introduction': ['introduction', 'background', 'related work'],
        'methodology': ['method', 'methodology', 'approach', 'proposed'],
        'results': ['result', 'evaluation', 'experiment'],
        'conclusion': ['conclusion', 'future work'],
        'references': ['reference', 'bibliography']
    }
    current_section = 'other'
    current_text = []
    for line in lines:
        line_lower = line.lower()
        section_found = False
        for section, markers in section_markers.items():
            for marker in markers:
                if re.match(rf'^{marker}[:\s]', line_lower) or line_lower == marker:
                    if current_text:
                        sections[current_section] += ' '.join(current_text)
                    current_section = section
                    current_text = []
                    section_found = True
                    break
            if section_found:
                break
        if not section_found:
            current_text.append(line)
    if current_text:
        sections[current_section] += ' '.join(current_text)
    for key in sections:
        sections[key] = sections[key][:2000]
    return sections

# ========== WINNING FEATURES ==========

# Feature 2: Surprising Insight Detector
def detect_surprising_insights(text):
    system_prompt = "You are a research assistant. Find the most counter-intuitive, surprising, or unexpected finding in this paper. If none found, say 'No surprising insights found.'"
    user_prompt = f"Paper:\n{text[:4000]}\n\nWhat is the most surprising or counter-intuitive finding?"
    try:
        response = ollama.chat(
            model='tinyllama:1.1b',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            options={'temperature': 0.3, 'max_tokens': 150}
        )
        return response['message']['content'].strip()
    except:
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except:
            return "Could not detect surprising insights."

# Feature 3: Comparative Analysis
def compare_papers(text1, text2):
    system_prompt = "You are a research analyst. Compare these two research papers. Highlight similarities, differences, and which one has stronger evidence."
    user_prompt = f"Paper 1:\n{text1[:3000]}\n\nPaper 2:\n{text2[:3000]}\n\nComparison:"
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except:
        return "Comparative analysis failed. Please try again."

# Feature 5: Download Full Report
def generate_full_report(text, summary, keywords, entities, method, filename):
    report = f"""# 📄 Research Paper Report

## 📌 Paper Information
- **File:** {filename}
- **Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
- **Method:** {method}

---

## 📊 Summary
{summary}

---

## 🏷️ Key Terms
{', '.join([kw[0] for kw in keywords[:15]]) if keywords else 'N/A'}

---

## 🏷️ Named Entities
"""
    for k, v in entities.items():
        if v:
            report += f"- **{k}**: {', '.join(v[:10])}\n"
    report += f"""
---

## 📈 Statistics
- Total words: {len(text.split())}
- Summary length: {len(summary.split())} words

---
*Generated by the NLP Research Assistant*
"""
    return report

# ========== ROUTES ==========
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if email == "test@demo.com" and password == "password123":
            session['user_id'] = "demo_user"
            session['email'] = "test@demo.com"
            return redirect(url_for('dashboard'))
        try:
            response = supabase.auth.sign_in_with_password({"email": email, "password": password})
            if response.user:
                session['user_id'] = response.user.id
                session['email'] = email
                return redirect(url_for('dashboard'))
        except Exception as e:
            return render_template('login.html', error="Invalid credentials. Try test@demo.com / password123")
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        try:
            supabase.auth.sign_up({"email": email, "password": password})
            return render_template('login.html', success="Account created! Please login.")
        except Exception as e:
            return render_template('signup.html', error=str(e))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    PAPER_STORE.reset()
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', email=session.get('email'))

@app.route('/summary')
def summary_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('summary.html')

@app.route('/chat')
def chat_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('chat.html')

# ========== API ==========
@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    filename = file.filename
    file_bytes = file.read()
    
    try:
        print(f"\n{'='*60}")
        print(f"📤 UPLOAD: {filename}")
        print(f"{'='*60}")
        
        cleaner = UltraAggressiveTextCleaner()
        
        if filename.lower().endswith('.pdf'):
            text = cleaner.extract_from_pdf(file_bytes)
        elif filename.lower().endswith('.docx'):
            import docx
            doc = docx.Document(BytesIO(file_bytes))
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            text = file_bytes.decode('utf-8', errors='ignore')
        
        print(f"🧹 Cleaning text ({len(text.split())} words)...")
        text = cleaner.clean_aggressively(text)
        
        if not text or len(text.strip()) < 100:
            return jsonify({"error": "Extracted text too short"}), 400
        
        print(f"✅ Cleaned: {len(text.split())} words")
        print(f"{'='*60}\n")
        
        PAPER_STORE.set_paper(text, filename)
        
        collection = build_index(text)
        if collection is None:
            return jsonify({"error": "Failed to build index"}), 500
        
        return jsonify({
            "success": True,
            "text": text[:500] + "..." if len(text) > 500 else text,
            "filename": filename,
            "word_count": len(text.split()),
            "cleaned": True
        })
    
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/get_paper', methods=['GET'])
def get_paper():
    return jsonify({
        "text": PAPER_STORE.text,
        "filename": PAPER_STORE.filename,
        "word_count": len(PAPER_STORE.text.split()),
        "metadata": PAPER_STORE.metadata,
        "quality": PAPER_STORE.text_quality
    })

@app.route('/api/summarize', methods=['POST'])
def summarize():
    data = request.json
    text = data.get('text', PAPER_STORE.text)
    method = data.get('method', 'textrank')
    num_sentences = data.get('num_sentences', 8)
    if not text:
        return jsonify({"error": "No text to summarize"}), 400
    try:
        if method == 'textrank':
            summary = summarizer.textrank_summarize(text, num_sentences)
        elif method == 'tfidf':
            summary = summarizer.tfidf_summarize(text, num_sentences)
        elif method == 'frequency':
            summary = summarizer.frequency_summarize(text, num_sentences)
        else:
            summary = summarizer.textrank_summarize(text, num_sentences)
        keywords = extract_keywords(text, 10)
        entities = extract_entities(text)
        user_id = session.get('user_id')
        if user_id and user_id != 'demo_user':
            try:
                supabase.table("summary_history").insert({
                    "user_id": user_id,
                    "paper_name": PAPER_STORE.filename,
                    "summary_method": method,
                    "summary": summary[:2000],
                    "created_at": datetime.now().isoformat()
                }).execute()
            except:
                pass
        return jsonify({
            "success": True,
            "summary": summary,
            "word_count": len(text.split()),
            "summary_word_count": len(summary.split()),
            "keywords": [kw[0] for kw in keywords],
            "entities": {k: len(v) for k, v in entities.items() if v},
            "method": method
        })
    except Exception as e:
        return jsonify({"error": f"Summarization failed: {str(e)}"}), 500

def extract_keywords(text, top_n=10):
    try:
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1, 2))
        tfidf = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf.toarray()[0]
        top_indices = scores.argsort()[-top_n:][::-1]
        return [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]
    except:
        from collections import Counter
        words = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', text.lower()) if w not in stop_words]
        return Counter(words).most_common(top_n)

def extract_entities(text):
    doc = nlp(text[:50000])
    entities = {
        'PERSON': [], 'ORG': [], 'GPE': [], 'DATE': [],
        'MONEY': [], 'PERCENT': [], 'PRODUCT': [], 'EVENT': []
    }
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    for key in entities:
        entities[key] = list(dict.fromkeys(entities[key]))[:10]
    return entities

# ========== HYBRID RAG CHAT ==========
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question', '')
    
    collection = PAPER_STORE.collection
    if collection is None:
        return jsonify({
            "answer": "⚠️ No paper uploaded yet. Please upload a paper first.",
            "sources": []
        })
    
    def _ollama_answer():
        model = get_embedding_model()
        q_emb = model.encode([question])[0]
        results = collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=3
        )
        chunks = results['documents'][0]
        context = "\n\n---\n\n".join(chunks)
        # Research assistant persona
        system_prompt = (
            "You are a senior research assistant with a PhD in the relevant field. "
            "Answer the user's question clearly and professionally, using ONLY the provided context. "
            "Cite specific findings from the paper when relevant. "
            "If the answer is not in the context, say 'I don't have that information.'"
        )
        response = ollama.chat(
            model='tinyllama:1.1b',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            options={'temperature': 0.1, 'max_tokens': 300}
        )
        answer = response['message']['content'].strip()
        return answer, chunks
    
    def _groq_answer():
        model = get_embedding_model()
        q_emb = model.encode([question])[0]
        results = collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=3
        )
        chunks = results['documents'][0]
        context = "\n\n---\n\n".join(chunks)
        system_prompt = (
            "You are a senior research assistant with a PhD. Answer clearly, cite sources from the context, and be helpful."
        )
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0.1,
            max_tokens=300
        )
        answer = response.choices[0].message.content.strip()
        return answer, chunks
    
    # ---- Hybrid logic ----
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_ollama_answer)
            answer, chunks = future.result(timeout=10)
            return jsonify({"success": True, "answer": answer, "sources": len(chunks)})
    except concurrent.futures.TimeoutError:
        try:
            answer, chunks = _groq_answer()
            return jsonify({"success": True, "answer": answer, "sources": len(chunks)})
        except Exception as e:
            try:
                model = get_embedding_model()
                q_emb = model.encode([question])[0]
                results = collection.query(
                    query_embeddings=[q_emb.tolist()],
                    n_results=1
                )
                chunk = results['documents'][0][0]
                return jsonify({"success": True, "answer": chunk, "sources": 1})
            except:
                return jsonify({"success": False, "answer": "⏳ All AI services timed out. Please try again."}), 504
    except Exception as e:
        try:
            answer, chunks = _groq_answer()
            return jsonify({"success": True, "answer": answer, "sources": len(chunks)})
        except:
            return jsonify({"success": False, "answer": f"Error: {str(e)}"}), 500

@app.route('/api/history')
def history():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    try:
        response = supabase.table("chat_history")\
            .select("question, answer, created_at")\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .limit(20)\
            .execute()
        return jsonify(response.data)
    except:
        return jsonify({"error": "Failed to fetch history"}), 500

@app.route('/api/debug_text')
def debug_text():
    return jsonify({
        "preview": PAPER_STORE.text[:1000] if PAPER_STORE.text else "No text",
        "length": len(PAPER_STORE.text),
        "word_count": len(PAPER_STORE.text.split()),
        "quality": PAPER_STORE.text_quality,
        "metadata": PAPER_STORE.metadata
    })

# ========== NEW ROUTES FOR WINNING FEATURES ==========

@app.route('/insights')
def insights_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('insights.html')

@app.route('/compare')
def compare_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('compare.html')

@app.route('/api/insights', methods=['POST'])
def insights_api():
    data = request.json
    text = data.get('text', PAPER_STORE.text)
    if not text:
        return jsonify({"error": "No text provided"}), 400
    insight = detect_surprising_insights(text)
    return jsonify({"insight": insight})

@app.route('/api/compare', methods=['POST'])
def compare_api():
    data = request.json
    text1 = data.get('text1', '')
    text2 = data.get('text2', '')
    if not text1 or not text2:
        return jsonify({"error": "Need two papers"}), 400
    comparison = compare_papers(text1, text2)
    return jsonify({"comparison": comparison})

@app.route('/api/report', methods=['POST'])
def report_api():
    data = request.json
    text = data.get('text', PAPER_STORE.text)
    method = data.get('method', 'textrank')
    summary = data.get('summary', '')
    if not text:
        return jsonify({"error": "No text"}), 400
    keywords = extract_keywords(text, 10)
    entities = extract_entities(text)
    report = generate_full_report(
        text=text,
        summary=summary or summarizer.textrank_summarize(text, 8),
        keywords=keywords,
        entities=entities,
        method=method,
        filename=PAPER_STORE.filename or "unknown"
    )
    return jsonify({"report": report})
# ========== RENDER TEMPLATES ==========
# (We will use Flask to render the HTML pages)
# The tabs and main UI are defined in the templates,
# but the features are in the Python code above.

@app.route('/history')
def history_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('history.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    
@app.route('/api/summary_history', methods=['GET'])
def summary_history():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    try:
        response = supabase.table("summary_history")\
            .select("paper_name, summary_method, summary, created_at")\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .limit(20)\
            .execute()
        return jsonify(response.data)
    except Exception as e:
        print(f"Summary history error: {e}")
        return jsonify({"error": str(e)}), 500
        
    
