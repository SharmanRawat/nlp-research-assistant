"""
Transparent Multi-Stage NLP Analysis Engine & Research Assistant
Advanced research paper analyzer with summarization, entity recognition, and export features.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import warnings
import json
from datetime import datetime
import tempfile
import os
from io import BytesIO
from wordcloud import WordCloud

warnings.filterwarnings('ignore')

# ---------- RAG & ML Imports ----------
try:
    import chromadb
    from chromadb.utils import embedding_functions
    from sentence_transformers import SentenceTransformer
    import ollama
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("RAG dependencies not installed. Install: pip install chromadb sentence-transformers ollama")

from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re

# ---------- File handling imports ----------
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
try:
    import pdfplumber
except ImportError:
    pdfplumber = None
try:
    import docx
except ImportError:
    docx = None

# ---------- Dependency Setup ----------
@st.cache_resource
def load_nlp():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    return nlp

@st.cache_resource
def load_nltk_data():
    """Download NLTK stopwords and tokenization resources."""
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
            except Exception as e:
                st.warning(f"Could not download NLTK resource '{pkg}': {e}")
    return stopwords.words('english')

# Global loading
try:
    nlp = load_nlp()
except Exception as e:
    st.error(f"Failed to load spaCy model: {e}")
    st.stop()

try:
    stop_words = load_nltk_data()
except Exception as e:
    st.error(f"Failed to load NLTK data: {e}")
    st.stop()

# Custom stopwords
KEEP_TOKENS = {'not', 'no', 'never', 'before', 'after', 'why', 'must', 'should', 'could', 'would'}
STOPWORDS_FILTERED = [w for w in stop_words if w not in KEEP_TOKENS]

# Emoji & Hashtag Mappings
EMOJI_MAP = {
    "😂": "<LAUGH_EMOJI>", "😊": "<POSITIVE_EMOJI>", "😡": "<ANGRY_EMOJI>",
    "😢": "<SAD_EMOJI>", "😏": "<SARCASM_EMOJI>", "❤️": "<HEART_EMOJI>",
    "🔥": "<FIRE_EMOJI>", "💯": "<HUNDRED_EMOJI>", "⭐": "<STAR_EMOJI>"
}
SARCASM_INDICATORS = ["🙄", "😒", "😑"]

# ---------- Improved Summarization Helpers ----------
def clean_text_for_sentences(text):
    """Clean text to improve sentence tokenization."""
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    return text
    
    
def split_sentences(text):
    """
    Robust sentence splitting with filters for:
    - Citations
    - Interview questions
    - Overly definitional sentences
    - Boosting for result‑bearing sentences
    """
    # 1. Clean up
    text = clean_text_for_sentences(text)

    # 2. Split into sentences
    try:
        sentences = sent_tokenize(text)
    except:
        sentences = []

    if len(sentences) < 2:
        sentences = re.split(r'(?<=[.!?])\s+', text)

    sentences = [s.strip() for s in sentences if len(s.split()) >= 5]

    if len(sentences) <= 1:
        sentences = re.split(r'\.\s*', text)
        sentences = [s.strip() + '.' for s in sentences if len(s.split()) >= 5]

    # 3. Helper filters
    def is_citation_sentence(sent):
        # Parenthetical citations count
        if len(re.findall(r'\([A-Z][a-z]+\s*[,;]?\s*\d{4}\)', sent)) >= 2:
            return True
        if re.match(r'^[A-Z][a-z]+,\s*[A-Z]\.?\s*\(\d{4}\)', sent):
            return True
        if re.search(r'et al\.\s*[,;]?\s*\d{4}', sent):
            return True
        if len(re.findall(r'[A-Z][a-z]+\s*[,;]?\s*[A-Z]\.', sent)) >= 3:
            return True
        if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s*,?\s*\d{4}[,.]', sent):
            return True
        return False

    def is_question_sentence(sent):
        # Starts with interrogative words
        if re.match(r'^(Do|Does|Did|Is|Are|Was|Were|Why|What|When|How|Where|Which)\s', sent):
            return True
        return False

    def is_definition_sentence(sent):
        # Contains "is defined as", "can be defined as", or many quoted fragments
        if re.search(r'\bis defined as\b|\bcan be defined as\b', sent, re.I):
            return True
        # More than two quoted segments (likely definitions)
        if len(re.findall(r'"[^"]+"', sent)) >= 2:
            return True
        if len(re.findall(r'\([^)]+\)', sent)) >= 3:  # heavy parentheticals
            return True
        return False

    # 4. Score and filter
    scored_sentences = []
    for sent in sentences:
        if is_citation_sentence(sent) or is_question_sentence(sent) or is_definition_sentence(sent):
            continue  # drop it

        # Base score
        score = 0

        # Boost for result keywords
        if re.search(r'\b(results|findings|showed|demonstrated|indicated|revealed|engaged|practiced|used|reported)\b', sent, re.I):
            score += 30
        # Boost for numbers/percentages (findings)
        if re.search(r'\b\d+%|\b\d+\s*(?:participants|students|subjects|interviewees)\b', sent, re.I):
            score += 20
        # Boost for sentences that mention the study itself
        if re.search(r'\b(study|research|paper|article)\b', sent, re.I):
            score += 10

        scored_sentences.append((sent, score))

    # Sort by score descending, then by original order (keep stable)
    scored_sentences.sort(key=lambda x: (-x[1], sentences.index(x[0])))

    # Return all sentences (you could limit to top N here, but TextRank will choose)
    return [s for s, _ in scored_sentences] if scored_sentences else sentences
    return sorted_sentences
def textrank_summarize(text, num_sentences=5):
    sentences = split_sentences(text)
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)
    
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
    except:
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        for i in range(len(sentences)):
            words_i = set(word_tokenize(sentences[i].lower()))
            for j in range(len(sentences)):
                if i != j:
                    words_j = set(word_tokenize(sentences[j].lower()))
                    if words_i and words_j:
                        similarity_matrix[i][j] = len(words_i & words_j) / (len(words_i | words_j) + 1e-8)
    
    scores = np.ones(len(sentences)) / len(sentences)
    for _ in range(10):
        new_scores = np.zeros(len(sentences))
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j and similarity_matrix[i][j] > 0:
                    new_scores[i] += similarity_matrix[i][j] * scores[j] / (np.sum(similarity_matrix[:, j]) + 1e-8)
        scores = 0.85 * new_scores + 0.15 / len(sentences)
    
    top_indices = np.argsort(scores)[-num_sentences:][::-1]
    summary = ' '.join(sentences[i] for i in sorted(top_indices))
    return summary

def generate_summary_tfidf(text, num_sentences=5):
    sentences = split_sentences(text)
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf = vectorizer.fit_transform(sentences)
        scores = tfidf.sum(axis=1).A1
        ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:num_sentences]
        return ' '.join(sentences[i] for i in sorted(ranked))
    except:
        return ' '.join(sentences[:num_sentences])

def generate_summary_frequency(text, num_sentences=5):
    sentences = split_sentences(text)
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    words = [w for w in words if w not in stop_words]
    from collections import Counter
    freq = Counter(words)
    scores = []
    for sent in sentences:
        sent_words = re.findall(r'\b[a-zA-Z]+\b', sent.lower())
        sent_score = sum(freq.get(w, 0) for w in sent_words)
        scores.append(sent_score / (len(sent_words) + 1))
    ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:num_sentences]
    return ' '.join(sentences[i] for i in sorted(ranked))

# ---------- Advanced PDF Extraction ----------
def extract_text_from_pdf_pypdf2(file_bytes):
    if PyPDF2 is None:
        return ""
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_text_from_pdf_pdfplumber(file_bytes):
    if pdfplumber is None:
        return ""
    text = ""
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                page_text = re.sub(r'\b\d+\s*$', '', page_text, flags=re.MULTILINE)
                page_text = re.sub(r'^\s*\d+\s*$', '', page_text, flags=re.MULTILINE)
                text += page_text + "\n"
    return text

def extract_text_from_file(uploaded_file, use_advanced=True):
    if uploaded_file is None:
        return ""
    
    file_type = uploaded_file.type
    file_bytes = uploaded_file.read()
    
    if file_type == "application/pdf":
        if use_advanced and pdfplumber is not None:
            try:
                text = extract_text_from_pdf_pdfplumber(file_bytes)
                if text and len(text.strip()) > 100:
                    return text
            except Exception as e:
                st.warning(f"pdfplumber extraction failed: {e}")
        
        if PyPDF2 is not None:
            try:
                text = extract_text_from_pdf_pypdf2(file_bytes)
                if text:
                    return text
            except Exception as e:
                st.warning(f"PyPDF2 extraction failed: {e}")
        
        return "Could not extract text from PDF. The file may be scanned or protected."
    
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        if docx is not None:
            doc = docx.Document(BytesIO(file_bytes))
            return "\n".join([para.text for para in doc.paragraphs])
        else:
            return "DOCX support not available. Install python-docx: pip install python-docx"
    
    else:
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return file_bytes.decode("latin-1")
            except:
                return "Could not decode file. Please ensure it's a valid text file."

# ---------- Preprocessing Functions ----------
def safe_preprocess(text: str) -> dict:
    original = text
    audit = []
    protected = {}

    def protect(text, pattern, placeholder):
        matches = re.findall(pattern, text)
        for i, m in enumerate(matches):
            key = f"__PROTECTED_{placeholder}_{i}__"
            protected[key] = m
            text = text.replace(m, key)
        return text

    text = protect(text, r'https?://\S+', 'URL')
    text = protect(text, r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'EMAIL')
    text = protect(text, r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', 'DATE')
    text = protect(text, r'\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM)?\b', 'TIME')
    text = protect(text, r'\b[A-Za-z]+\d+[A-Za-z]*\b', 'CODEID')
    text = protect(text, r'\bC\+\+\b', 'CODEID')

    for emoji, token in EMOJI_MAP.items():
        if emoji in text:
            text = text.replace(emoji, token)
            audit.append(f"Replaced {emoji} with {token}")
    for emoji in SARCASM_INDICATORS:
        if emoji in text:
            text = text.replace(emoji, "<SARCASM_EMOJI>")
            audit.append(f"Replaced {emoji} with <SARCASM_EMOJI>")

    def split_hashtag(match):
        tag = match.group(1)
        words = re.findall(r'[A-Z][a-z]*|[a-z]+', tag)
        return ' '.join(words)
    text = re.sub(r'#(\w+)', lambda m: split_hashtag(m), text)

    def lower_except_acronyms(t):
        if re.fullmatch(r'[A-Z]{2,}', t) or re.fullmatch(r'[A-Z]\.', t):
            return t
        if re.fullmatch(r'__PROTECTED_\w+__', t):
            return t
        return t.lower()
    words = re.findall(r'\S+', text)
    words = [lower_except_acronyms(w) for w in words]
    text = ' '.join(words)

    for key, val in protected.items():
        text = text.replace(key, val)

    doc = nlp(text)
    tokens = [token.text for token in doc]
    tokens_filtered = [t for t in tokens if t.lower() not in STOPWORDS_FILTERED]

    filtered_text = ' '.join(tokens_filtered)
    doc_filtered = nlp(filtered_text)
    lemmatized_filtered = []
    for token in doc_filtered:
        if token.text in protected.values() or (token.text.startswith('<') and token.text.endswith('>')):
            lemmatized_filtered.append(token.text)
        else:
            if token.pos_ in ('VERB', 'NOUN', 'ADJ', 'ADV'):
                lemmatized_filtered.append(token.lemma_)
            else:
                lemmatized_filtered.append(token.text)

    audit.append(f"Protected entities: {list(protected.keys())}")
    audit.append(f"Emojis mapped: {[e for e in EMOJI_MAP if e in original]}")
    audit.append(f"Hashtags split: {re.findall(r'#\w+', original)}")
    audit.append(f"Stopwords removed (kept exceptions: {KEEP_TOKENS})")

    return {
        'original': original,
        'protected': protected,
        'tokens_raw': tokens,
        'tokens_filtered': tokens_filtered,
        'lemmatized_filtered': lemmatized_filtered,
        'audit': audit
    }

def naive_preprocess(text: str) -> list:
    try:
        tokens = word_tokenize(text.lower())
    except LookupError:
        tokens = text.lower().split()
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    return tokens

# ---------- Research Paper Structure Extraction ----------
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

# ---------- Named Entity Recognition ----------
def extract_entities(text):
    doc = nlp(text)
    entities = {
        'PERSON': [], 'ORG': [], 'GPE': [], 'DATE': [],
        'MONEY': [], 'PERCENT': [], 'PRODUCT': [], 'EVENT': [],
        'WORK_OF_ART': [], 'LAW': [], 'LANGUAGE': [], 'FAC': []
    }
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    
    for key in entities:
        entities[key] = list(dict.fromkeys(entities[key]))[:20]
    
    return entities

# ---------- Advanced Keyword Extraction ----------
def extract_keywords(text, top_n=10):
    vectorizer = TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1, 2))
    try:
        tfidf = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf.toarray()[0]
        top_indices = scores.argsort()[-top_n:][::-1]
        keywords = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]
        return keywords
    except Exception:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        words = [w for w in words if w not in stop_words]
        from collections import Counter
        return Counter(words).most_common(top_n)

# ---------- Classification Data & Setup ----------
TECH_TEXTS = [
    "The algorithm uses a convolutional neural network for image classification.",
    "We propose a novel attention mechanism for transformer models.",
    "The system achieves 95% accuracy on the test set with a p-value less than 0.05.",
    "Our method reduces inference time by 30% using quantization.",
    "We evaluate on several benchmarks including GLUE and SQuAD."
]
NON_TECH_TEXTS = [
    "The new product is a game-changer for the industry.",
    "This groundbreaking innovation will revolutionize the market.",
    "Our unmatched performance sets a new standard.",
    "The solution is user-friendly and intuitive.",
    "We are proud to announce our latest version with amazing features."
]

def train_classifiers():
    corpus = TECH_TEXTS + NON_TECH_TEXTS
    labels = [1] * len(TECH_TEXTS) + [0] * len(NON_TECH_TEXTS)
    
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(corpus)
    
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Linear SVM': LinearSVC(max_iter=1000, random_state=42, dual=False)
    }
    
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_vec, labels)
        fitted_models[name] = model
        
    return fitted_models, vectorizer

# ---------- Hype vs Rigor ----------
HYPE_WORDS = [
    "groundbreaking", "revolutionary", "unmatched", "game-changing", "never-before-seen",
    "phenomenal", "extraordinary", "incredible", "amazing", "astonishing", "remarkable",
    "exceptional", "fantastic", "unprecedented", "paradigm-shift", "miracle", "magic", "breakthrough"
]
RIGOR_WORDS = [
    "quantitative", "p-value", "confidence interval", "sample size", "metric", "bound",
    "statistically significant", "correlation", "causal", "controlled", "experiment",
    "randomized", "reproducible", "robust", "validation", "test set", "training set",
    "baseline", "benchmark", "evaluation", "accuracy", "precision", "recall", "f1"
]

def compute_hype_rigor(text: str):
    text_lower = text.lower()
    hype_count = sum(text_lower.count(w) for w in HYPE_WORDS)
    rigor_count = sum(text_lower.count(w) for w in RIGOR_WORDS)
    total_words = max(1, len(text.split()))
    
    hype_score = min(100, (hype_count / total_words) * 1000)
    rigor_score = min(100, (rigor_count / total_words) * 1000)
    
    if re.findall(r'\b\d+(\.\d+)?\b', text):
        rigor_score = min(100, rigor_score + 10)
    if re.search(r'p\s*[<>=]\s*0\.\d+', text_lower):
        rigor_score = min(100, rigor_score + 15)
        
    return hype_score, rigor_score

# ---------- Concept Graph ----------
def build_concept_graph(text: str):
    doc = nlp(text)
    nodes = set()
    for chunk in doc.noun_chunks:
        nodes.add(chunk.text)
    for ent in doc.ents:
        nodes.add(ent.text)
    for token in doc:
        if token.pos_ in ('NOUN', 'PROPN'):
            nodes.add(token.text)
    nodes = list(nodes)[:30]

    sentences = [sent.text for sent in doc.sents]
    co_occurrence = {n: {m: 0 for m in nodes} for n in nodes}
    for sent in sentences:
        sent_tokens = [t.text for t in nlp(sent) if t.text in nodes]
        for i, a in enumerate(sent_tokens):
            for b in sent_tokens[i+1:]:
                co_occurrence[a][b] += 1
                co_occurrence[b][a] += 1

    adj_matrix = pd.DataFrame(co_occurrence, index=nodes, columns=nodes).fillna(0)
    G = nx.Graph()
    for node in nodes:
        G.add_node(node)
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if i < j and adj_matrix.iloc[i, j] > 0:
                G.add_edge(a, b, weight=adj_matrix.iloc[i, j])
    return G, adj_matrix, nodes

# ---------- Export Functions ----------
def export_results(text, results):
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'text_length': len(text),
        'word_count': len(text.split()),
        'results': results
    }
    return json.dumps(export_data, indent=2)

# ---------- Streamlit UI ----------
def main():
    st.set_page_config(layout="wide", page_title="NLP Research Assistant")
    st.title("🧠 Advanced NLP Research Paper Analyzer")
    st.markdown("Upload a research paper (PDF, DOCX, TXT) or paste text for comprehensive analysis.")

    # ---------- Main Input Section ----------
    st.header("📄 Input Your Research Paper")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            help="Upload a research paper to analyze. For best results, use PDF or DOCX."
        )
    
    with col2:
        use_advanced = st.checkbox("Use advanced PDF extraction (pdfplumber, better for complex layouts)", value=True)
        st.caption("Slower but handles scanned/layout-heavy PDFs.")
    
    if uploaded_file is None:
        st.info("No file uploaded. You can select a sample text or paste your own in the sidebar.")
        st.sidebar.header("Sample or Manual Input")
        sample_texts = {
            "Technical Abstract": "We introduce a novel transformer-based architecture for sentiment analysis. The model achieves 92.3% accuracy on the benchmark dataset with a p-value < 0.01. Our method is reproducible and robust to noise. #NLPResearch",
            "Hype Press Release": "Our groundbreaking AI solution is a game-changer! It's revolutionary and unmatched in the industry. Experience the future today! 💯🔥",
            "Mixed Text with Emojis": "This is amazing 😊! But seriously, we need to consider the p-value and sample size. No more errors! #NoMoreErrors",
            "Ambiguous Blog Post": "I think this could be important, but I'm not sure. The results are interesting. Maybe we need more data."
        }
        selected_sample = st.sidebar.selectbox("Choose a sample text", list(sample_texts.keys()))
        input_text = st.sidebar.text_area("Or paste your own text:", value=sample_texts[selected_sample], height=150)
    else:
        with st.spinner("Extracting text from file (this may take a moment)..."):
            input_text = extract_text_from_file(uploaded_file, use_advanced=use_advanced)
            if not input_text:
                st.error("Could not extract text from the file. Please try a different file or use the sample text.")
                sample_texts = {
                    "Technical Abstract": "We introduce a novel transformer-based architecture for sentiment analysis. The model achieves 92.3% accuracy on the benchmark dataset with a p-value < 0.01. Our method is reproducible and robust to noise. #NLPResearch"
                }
                input_text = st.sidebar.text_area("Fallback text:", value=sample_texts["Technical Abstract"], height=150)
            else:
                st.success(f"File loaded successfully! Extracted {len(input_text.split())} words.")
                with st.expander("Preview extracted text"):
                    st.text(input_text[:800] + "..." if len(input_text) > 800 else input_text)

    # ---------- Tabs ----------
    tabs = st.tabs([
    "🔒 Preprocessing",          # 0
    "📊 Vectorization",          # 1
    "🧪 Classifier",             # 2
    "📈 Hype vs Rigor",          # 3
    "🔗 Concept Graph",          # 4
    "📝 Summary",                # 5
    "🏷️ Entities",               # 6
    "🔑 Keywords",               # 7
    "📊 Paper Stats",            # 8
    "🤖 Paper Q&A",              # 9   <- RAG Chatbot
    "🔗 Citation Network",       # 10
    "🏷️ Classification",         # 11
    "📊 Readability Heatmap",    # 12
    "🔬 Methodology",            # 13
    "📤 Export"                  # 14
    ])

    # ---------- Tab 0: Preprocessing ----------
    with tabs[0]:
        st.header("Safer Preprocessing & Safety Audit Pipeline")
        if st.button("Run Preprocessing", key="preproc"):
            with st.spinner("Preprocessing..."):
                result = safe_preprocess(input_text)
                naive = naive_preprocess(input_text)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Naive Preprocessing")
                    st.write("Tokens:", naive)
                with col2:
                    st.subheader("Safe Preprocessing")
                    st.write("Filtered & Lemmatized Tokens:", result['lemmatized_filtered'])

                st.subheader("Side-by-Side Diff")
                diff_df = pd.DataFrame({
                    "Naive": [", ".join(naive[:20]) + ("..." if len(naive)>20 else "")],
                    "Safe": [", ".join(result['lemmatized_filtered'][:20]) + ("..." if len(result['lemmatized_filtered'])>20 else "")]
                })
                st.dataframe(diff_df)

                with st.expander("Audit Log"):
                    for entry in result['audit']:
                        st.write(f"- {entry}")
        else:
            st.info("Click the button above to run preprocessing.")

    # ---------- Tab 1: Vectorization ----------
    with tabs[1]:
        st.header("Dual Vectorization Layer")
        preprocessed_text = ' '.join(safe_preprocess(input_text)['lemmatized_filtered'])
        corpus = [preprocessed_text]

        tfidf_vec = TfidfVectorizer(max_features=20)
        tfidf_matrix = tfidf_vec.fit_transform(corpus)
        feature_names = tfidf_vec.get_feature_names_out()
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

        st.subheader("TF-IDF Feature Weights")
        st.dataframe(tfidf_df)

        if len(feature_names) > 0:
            top_n = min(10, len(feature_names))
            top_indices = tfidf_matrix.toarray()[0].argsort()[-top_n:][::-1]
            top_features = feature_names[top_indices]
            top_weights = tfidf_matrix.toarray()[0][top_indices]
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(top_features, top_weights, color='skyblue')
            ax.set_xlabel("TF-IDF Score")
            ax.set_title(f"Top {top_n} TF-IDF Features")
            st.pyplot(fig)

        doc = nlp(input_text)
        words = [token.text for token in doc if token.is_alpha and not token.is_stop]
        if len(words) >= 2:
            word_pairs = st.multiselect("Select two words to compare via spaCy vectors", words, default=words[:2] if len(words)>=2 else [])
            if len(word_pairs) == 2:
                w1, w2 = word_pairs
                vec1 = nlp(w1).vector
                vec2 = nlp(w2).vector
                if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                    sim = cosine_similarity([vec1], [vec2])[0][0]
                    st.metric(f"Cosine Similarity: {w1} vs {w2}", f"{sim:.4f}")
                    st.write(f"Vector distance (Euclidean): {np.linalg.norm(vec1 - vec2):.4f}")

    # ---------- Tab 2: Classifier ----------
    with tabs[2]:
        st.header("Technical vs Non-Technical Classification")
        models, vectorizer = train_classifiers()
        
        vec_input = vectorizer.transform([input_text])
        st.subheader("Predictions for Current Input")
        
        preds = {}
        for name, model in models.items():
            pred = model.predict(vec_input)[0]
            preds[name] = "Technical" if pred == 1 else "Non-Technical"
        
        st.write(pd.DataFrame([preds]))

    # ---------- Tab 3: Hype vs Rigor ----------
    with tabs[3]:
        st.header("Hype vs Rigor Analysis")
        hype, rigor = compute_hype_rigor(input_text)
        col1, col2 = st.columns(2)
        col1.metric("Hype Score", f"{hype:.1f} / 100")
        col2.metric("Rigor Score", f"{rigor:.1f} / 100")

    # ---------- Tab 4: Concept Graph ----------
    with tabs[4]:
        st.header("Concept Co-occurrence Network")
        if st.button("Generate Concept Graph"):
            with st.spinner("Building graph..."):
                G, adj_matrix, nodes = build_concept_graph(input_text)
                if len(G.edges()) > 0:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    pos = nx.spring_layout(G)
                    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=500)
                    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)
                    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
                    st.pyplot(fig)
                else:
                    st.info("Not enough entity/noun connections found to build a graph.")

    # ---------- Tab 5: Summary ----------
        with tabs[5]:
            st.header("Multi-Algorithm Text Summarization")
            
            summary_method = st.radio(
                "Select summarization method:",
                ["TextRank (Graph-based)", "TF-IDF Extractive", "Sentence Frequency"]
            )
            
            num_sents = st.slider("Select Summary Length (Sentences)", 1, 10, 5)
            
            # Debug info
            with st.expander("📊 Text Statistics (Debug)"):
                sentences = split_sentences(input_text)
                st.write(f"**Total characters:** {len(input_text)}")
                st.write(f"**Word count:** {len(input_text.split())}")
                st.write(f"**Sentences detected:** {len(sentences)}")
                if sentences:
                    st.write("**First 3 sentences:**")
                    for i, s in enumerate(sentences[:3]):
                        st.write(f"{i+1}. {s[:150]}...")
            
            if st.button("Generate Summary", key="summarize"):
                with st.spinner("Generating summary..."):
                    if summary_method == "TextRank (Graph-based)":
                        summary = textrank_summarize(input_text, num_sents)
                    elif summary_method == "TF-IDF Extractive":
                        summary = generate_summary_tfidf(input_text, num_sents)
                    else:
                        summary = generate_summary_frequency(input_text, num_sents)
                    
                    # ----- Post-filter summary sentences -----
                    def is_bad_sentence(sent):
                        if '?' in sent:  # questions
                            return True
                        if re.match(r'^[A-Z][a-z]+,\s*[A-Z]\.?\s*\(\d{4}\)', sent):  # references
                            return True
                        if re.match(r'^(Moving bravely|Left to my own devices|Mobile technology|Autonomy and|Profiling mobile|Computer Assisted)', sent, re.I):
                            return True
                        if len(re.findall(r'[A-Z][a-z]+\s*[,;]?\s*[A-Z]\.', sent)) >= 2:
                            return True
                        return False
                    
                    raw_sentences = split_sentences(summary)
                    summary_sentences = [s for s in raw_sentences if not is_bad_sentence(s)]
                    if not summary_sentences:
                        summary_sentences = [summary]  # fallback
                    
                    st.markdown("### 📌 Key Takeaways")
                    for sent in summary_sentences:
                        # Bold numbers and percentages
                        sent = re.sub(r'(\d+%|\d+\s*(?:participants|students|years))', r'**\1**', sent)
                        # Bold key verbs
                        for word in ['showed', 'demonstrated', 'revealed', 'indicated', 'reported', 'found']:
                            sent = re.sub(rf'\b({word})\b', r'**\1**', sent, flags=re.I)
                        st.markdown(f"- {sent}")
                    
                    st.caption(f"Original: {len(input_text.split())} words | Summary: {len(summary.split())} words")
                    
                    # Download button
                    summary_md = f"""# Summary of {uploaded_file.name if uploaded_file else 'your text'}

    **Method**: {summary_method}  
    **Number of sentences**: {num_sents}  

    ## Key Takeaways
    {chr(10).join('- ' + s for s in summary_sentences)}

    ---
    *Generated by the Transparent NLP Analysis Engine*
    """
                    st.download_button(
                        label="📥 Download Summary as Markdown",
                        data=summary_md,
                        file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )    
                
    # ---------- Tab 6: Entities ----------
    with tabs[6]:
        st.header("Named Entity Recognition")
        entities = extract_entities(input_text)
        for label, ent_list in entities.items():
            if ent_list:
                st.write(f"**{label}**: {', '.join(ent_list)}")

    # ---------- Tab 7: Keywords ----------
    with tabs[7]:
        st.header("Top Keywords & Keyphrases")
        keywords = extract_keywords(input_text, top_n=10)
        kw_df = pd.DataFrame(keywords, columns=["Keyword", "Score/Count"])
        st.dataframe(kw_df)

    # ---------- Tab 8: Export ----------
    with tabs[8]:
        st.header("Export Analysis Data")
        summary_res = textrank_summarize(input_text, num_sentences=3)
        entities_res = extract_entities(input_text)
        hype, rigor = compute_hype_rigor(input_text)
        
        results_payload = {
            "summary": summary_res,
            "entities": entities_res,
            "hype_score": hype,
            "rigor_score": rigor
        }
        
        json_str = export_results(input_text, results_payload)
        st.download_button(
            label="Download JSON Report",
            data=json_str,
            file_name="nlp_analysis_report.json",
            mime="application/json"
        )
    
        # ---------- Tab 9: RAG Chatbot ----------
    with tabs[9]:
        st.header("🤖 Research Paper Q&A with Local LLM (Ollama)")
        st.markdown("**Powered by Qwen2.5-Coder-1.5B** (running via Ollama) – 100% offline and private.")

        if not RAG_AVAILABLE:
            st.error("Missing dependencies. Run: pip install chromadb sentence-transformers ollama")
        else:
            if not input_text:
                st.info("Please upload a research paper first.")
            else:
                # Load embedding model (cached)
                @st.cache_resource
                def load_embedding_model():
                    return SentenceTransformer('all-MiniLM-L6-v2')

                with st.spinner("Loading AI models..."):
                    embedding_model = load_embedding_model()

                # Compute a unique collection name based on the paper content
                import hashlib
                paper_hash = hashlib.md5(input_text[:1000].encode()).hexdigest()[:8]
                collection_name = f"paper_chunks_{paper_hash}"

                # Check if we need to (re)index
                if ('rag_ready' not in st.session_state or 
                    st.session_state.get('paper_hash') != paper_hash):
                    
                    with st.spinner("Indexing your paper for Q&A..."):
                        def chunk_text(text, chunk_size=500, overlap=50):
                            words = text.split()
                            chunks = []
                            for i in range(0, len(words), chunk_size - overlap):
                                chunk = ' '.join(words[i:i+chunk_size])
                                if chunk:
                                    chunks.append(chunk)
                            return chunks

                        chunks = chunk_text(input_text)
                        embeddings = embedding_model.encode(chunks)
                        
                        client = chromadb.Client()
                        # Delete collection if it exists (from a previous run)
                        try:
                            client.delete_collection(collection_name)
                        except:
                            pass  # collection doesn't exist
                        
                        # Create fresh collection
                        collection = client.create_collection(name=collection_name)
                        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                            collection.add(
                                documents=[chunk],
                                embeddings=[emb.tolist()],
                                ids=[str(i)]
                            )
                        
                        st.session_state.rag_collection = collection
                        st.session_state.rag_chunks = chunks
                        st.session_state.paper_text = input_text
                        st.session_state.paper_hash = paper_hash
                        st.session_state.rag_ready = True
                    st.success(f"✅ Paper indexed! {len(chunks)} chunks ready.")
                else:
                    collection = st.session_state.rag_collection

                # Chat interface
                st.markdown("### 💬 Ask Questions About the Paper")
                
                if 'rag_messages' not in st.session_state:
                    st.session_state.rag_messages = []

                for msg in st.session_state.rag_messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                        if "sources" in msg:
                            st.caption(f"📚 Sources: {', '.join(msg['sources'])}")

                if prompt := st.chat_input("Ask about the research paper..."):
                    st.session_state.rag_messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            try:
                                # Retrieve relevant chunks
                                query_embedding = embedding_model.encode([prompt])[0]
                                results = collection.query(
                                    query_embeddings=[query_embedding.tolist()],
                                    n_results=3
                                )
                                context_chunks = results['documents'][0]
                                context = "\n\n---\n\n".join(context_chunks)
                                
                                # Generate with Ollama
                                response = ollama.chat(
                                    model='qwen2.5-coder:1.5b',
                                    messages=[{
                                        'role': 'user',
                                        'content': f"You are a research assistant. Answer the question based ONLY on the provided context.\n\nContext:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
                                    }],
                                    options={'temperature': 0.3, 'max_tokens': 512}
                                )
                                answer = response['message']['content'].strip()
                                st.markdown(answer)
                                st.caption("📚 Sources: Chunk 1, 2, 3")
                                with st.expander("📖 View Source Chunks"):
                                    for i, ctx in enumerate(context_chunks):
                                        st.markdown(f"**Source {i+1}:**")
                                        st.text(ctx[:500] + "..." if len(ctx) > 500 else ctx)
                                        st.divider()
                                st.session_state.rag_messages.append({
                                    "role": "assistant",
                                    "content": answer,
                                    "sources": ["Chunk 1", "Chunk 2", "Chunk 3"]
                                })
                            except Exception as e:
                                st.error(f"Error: {e}\nMake sure Ollama is running (`ollama serve`) and model is pulled (`ollama pull qwen2.5-coder:1.5b`).")
                                
        # ---------- Tab 10: Citation Network ----------
    with tabs[10]:
        st.header("🔗 Citation Network Analysis")
        st.markdown("Shows the most frequently cited authors in the paper.")

        if st.button("Build Citation Network", key="citation_network"):
            with st.spinner("Extracting citations..."):
                # Extract all (Author, Year) citations
                citations = re.findall(r'\(([A-Z][a-z]+\s*[,;]?\s*(?:et al\.)?\s*\d{4}[a-z]?)\)', input_text)
                citations = [c.strip() for c in citations]
                if not citations:
                    st.info("No citations found in the text.")
                else:
                    freq = Counter(citations)
                    top_citations = freq.most_common(20)
                    
                    st.subheader("Top 20 Cited References")
                    df_citations = pd.DataFrame(top_citations, columns=["Citation", "Frequency"])
                    st.dataframe(df_citations)
                    
                    # Bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    authors = [c[:30] + "..." if len(c) > 30 else c for c, _ in top_citations]
                    counts = [count for _, count in top_citations]
                    ax.barh(authors, counts, color='coral')
                    ax.set_xlabel("Number of Mentions")
                    ax.set_title("Citation Frequency")
                    st.pyplot(fig)
                    
        # ---------- Tab 11: Research Field Classification ----------
    with tabs[11]:
        st.header("🏷️ Research Field Classification")
        st.markdown("Automatically identifies the primary research field based on keyword matches.")

        if st.button("Classify Paper", key="classify"):
            with st.spinner("Analyzing paper..."):
                fields = {
                    "Computer Science": ["algorithm", "neural", "network", "data", "machine learning", "AI", "deep learning", "transformer", "model", "training"],
                    "Linguistics/Education": ["language", "learner", "autonomy", "teaching", "student", "classroom", "vocabulary", "pronunciation", "mobile", "EFL", "ESL"],
                    "Medicine/Biology": ["patient", "disease", "gene", "protein", "cell", "clinical", "treatment", "drug", "cancer", "COVID-19"],
                    "Physics/Engineering": ["quantum", "particle", "mechanical", "circuit", "thermal", "energy", "force", "velocity"],
                    "Social Sciences": ["society", "survey", "interview", "participant", "behavior", "gender", "race", "economy", "policy"],
                    "Psychology": ["behavior", "cognition", "brain", "amygdala", "hormone", "stress", "depression", "anxiety"]
                }
                scores = {}
                text_lower = input_text.lower()
                for field, keywords in fields.items():
                    score = sum(text_lower.count(kw) for kw in keywords)
                    scores[field] = score
                
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                
                st.markdown("### 📊 Classification Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Primary Field", sorted_scores[0][0])
                    st.caption(f"*Confidence: {sorted_scores[0][1]} keyword matches*")
                with col2:
                    st.metric("Secondary Field", sorted_scores[1][0] if len(sorted_scores) > 1 else "N/A")
                
                fig, ax = plt.subplots(figsize=(10, 4))
                fields_list = [f for f, _ in sorted_scores]
                scores_list = [s for _, s in sorted_scores]
                colors = plt.cm.viridis(np.linspace(0, 1, len(fields_list)))[::-1]
                ax.barh(fields_list, scores_list, color=colors)
                ax.set_xlabel("Keyword Matches")
                ax.set_title("Paper Classification Scores")
                st.pyplot(fig)
                
        # ---------- Tab 12: Readability Heatmap ----------
    with tabs[12]:
        st.header("📊 Section Readability Heatmap")
        st.markdown("Visualizes which sections are easiest or hardest to read.")

        # Helper functions (reuse from Paper Stats)
        def flesch(text):
            words = re.findall(r'\b[a-zA-Z]+\b', text)
            sentences = split_sentences(text)
            if len(sentences) == 0 or len(words) == 0:
                return 0
            syllables = sum(len(re.findall(r'[aeiouy]', w)) for w in words)
            score = 206.835 - 1.015*(len(words)/len(sentences)) - 84.6*(syllables/len(words))
            return max(0, min(100, score))

        def gunning_fog(text):
            words = re.findall(r'\b[a-zA-Z]+\b', text)
            sentences = split_sentences(text)
            if len(sentences) == 0 or len(words) == 0:
                return 0
            complex_words = sum(1 for w in words if len(re.findall(r'[aeiouy]', w)) >= 3)
            return 0.4 * ((len(words)/len(sentences)) + 100*(complex_words/len(words)))

        if st.button("Generate Readability Heatmap", key="readability_heatmap"):
            with st.spinner("Calculating..."):
                sections = extract_paper_structure(input_text)
                section_data = []
                for section_name, content in sections.items():
                    if content and len(content.split()) > 10:
                        section_data.append({
                            "Section": section_name.capitalize(),
                            "Flesch Score": flesch(content),
                            "Gunning Fog": gunning_fog(content),
                            "Words": len(content.split())
                        })
                if section_data:
                    df_sections = pd.DataFrame(section_data)
                    st.subheader("Section Readability Scores")
                    st.dataframe(df_sections)

                    # Heatmap
                    fig, ax = plt.subplots(figsize=(10, 4))
                    heatmap_data = df_sections[["Flesch Score", "Gunning Fog"]].T
                    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r',
                               xticklabels=df_sections["Section"], ax=ax)
                    ax.set_title("Section Readability (Green = Easier)")
                    st.pyplot(fig)

                    # Hardest and easiest sections
                    hardest = df_sections.loc[df_sections["Gunning Fog"].idxmax()]
                    easiest = df_sections.loc[df_sections["Flesch Score"].idxmax()]
                    st.warning(f"**Hardest Section**: {hardest['Section']} (Gunning Fog: {hardest['Gunning Fog']:.1f})")
                    st.success(f"**Easiest Section**: {easiest['Section']} (Flesch Score: {easiest['Flesch Score']:.1f})")
                else:
                    st.info("Not enough content to compute readability scores.")
                    
        # ---------- Tab 13: Methodology Extractor ----------
    with tabs[13]:
        st.header("🔬 Study Methodology Extractor")
        st.markdown("Automatically extracts key methodological details from the paper.")

        if st.button("Extract Methodology", key="methodology"):
            with st.spinner("Extracting methodology..."):
                sections = extract_paper_structure(input_text)
                method_text = sections.get('methodology', '')
                if not method_text:
                    method_text = input_text[:3000]  # fallback

                methodology = {}
                # Sample size
                sample_match = re.search(r'(\d+)\s*(?:participants?|subjects?|students?)', method_text, re.I)
                if sample_match:
                    methodology['Sample Size'] = sample_match.group(1)
                # Study design
                design_keywords = ['semi-structured', 'structured', 'unstructured', 'interview', 'survey',
                                   'questionnaire', 'experiment', 'case study', 'longitudinal', 'cross-sectional']
                found_design = [kw for kw in design_keywords if kw in method_text.lower()]
                if found_design:
                    methodology['Study Design'] = ', '.join(found_design)
                # Data collection
                if 'interview' in method_text.lower():
                    methodology['Data Collection'] = 'Interview'
                elif 'survey' in method_text.lower() or 'questionnaire' in method_text.lower():
                    methodology['Data Collection'] = 'Survey/Questionnaire'
                elif 'experiment' in method_text.lower():
                    methodology['Data Collection'] = 'Experiment'
                else:
                    methodology['Data Collection'] = 'Not specified'
                # Analysis
                analysis_keywords = ['qualitative', 'quantitative', 'mixed methods', 'statistical', 'thematic', 'content analysis']
                found_analysis = [kw for kw in analysis_keywords if kw in method_text.lower()]
                if found_analysis:
                    methodology['Analysis'] = ', '.join(found_analysis)

                if methodology:
                    st.markdown("### 📋 Extracted Methodology")
                    for key, value in methodology.items():
                        st.metric(key, value)
                else:
                    st.info("Could not extract methodology. This works best with structured research papers.")

                with st.expander("📄 Methodology Section Snippet"):
                    st.text(method_text[:1000] + "..." if len(method_text) > 1000 else method_text)                
        # ---------- Tab: Paper Stats ----------
    with tabs[-1]:  # adjust index based on your list order
        st.header("📊 Paper Statistics & Readability")
        
        # Helper functions
        def flesch(text):
            words = re.findall(r'\b[a-zA-Z]+\b', text)
            sentences = split_sentences(text)
            if len(sentences) == 0 or len(words) == 0:
                return 0
            # Rough syllable count
            syllables = sum(len(re.findall(r'[aeiouy]', w)) for w in words)
            score = 206.835 - 1.015*(len(words)/len(sentences)) - 84.6*(syllables/len(words))
            return max(0, min(100, score))
        
        def gunning_fog(text):
            words = re.findall(r'\b[a-zA-Z]+\b', text)
            sentences = split_sentences(text)
            if len(sentences) == 0 or len(words) == 0:
                return 0
            complex_words = sum(1 for w in words if len(re.findall(r'[aeiouy]', w)) >= 3)
            return 0.4 * ((len(words)/len(sentences)) + 100*(complex_words/len(words)))
        
        # Compute metrics
        words = re.findall(r'\b[a-zA-Z]+\b', input_text)
        unique_words = set(w.lower() for w in words)
        citations = len(re.findall(r'\([A-Z][a-z]+\s*[,;]?\s*\d{4}\)', input_text))
        sentences = split_sentences(input_text)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Flesch Reading Ease", f"{flesch(input_text):.1f}", help="Higher = easier to read")
        col2.metric("Gunning Fog Index", f"{gunning_fog(input_text):.1f}", help="Grade level needed")
        col3.metric("Citations Found", citations)
        
        col4, col5, col6 = st.columns(3)
        col4.metric("Total Words", len(words))
        col5.metric("Unique Words", len(unique_words))
        col6.metric("Vocabulary Richness", f"{len(unique_words)/len(words):.3f}" if words else "0")
        
        # Frequency of content words
        from collections import Counter
        freq = Counter(w.lower() for w in words if w.lower() not in stop_words)
        top_words = freq.most_common(15)
        
        if top_words:
            st.subheader("🔝 Top 15 Most Frequent Content Words")
            fig, ax = plt.subplots(figsize=(10, 5))
            words_list, counts = zip(*top_words)
            ax.barh(words_list, counts, color='skyblue')
            ax.set_xlabel("Frequency")
            st.pyplot(fig)
        
        # Word Cloud (optional, requires wordcloud library)
        try:
            text_to_cloud = ' '.join(w for w in words if w.lower() not in stop_words and len(w) > 2)
            if text_to_cloud:
                st.subheader("☁️ Word Cloud")
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_to_cloud)
                fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
                ax_wc.imshow(wordcloud, interpolation='bilinear')
                ax_wc.axis('off')
                st.pyplot(fig_wc)
        except NameError:
            pass  # wordcloud not installed, skip silently

if __name__ == "__main__":
    main()
