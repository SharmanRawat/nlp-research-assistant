# 🧠 NLP Research Paper Analyzer

**Upload a research paper → Get a summary, insights, and chat with an AI assistant.**

A complete research assistant tool that extracts, summarizes, and analyzes academic papers with **local RAG**, **abstractive summarization**, **surprising insights**, **comparative analysis**, and **history tracking**.

---

## ✨ Features

- 📄 **Upload PDF/DOCX/TXT** – with aggressive text cleaning for corrupted PDFs
- 📝 **Multi‑method summarization** – TextRank, TF‑IDF, Frequency‑based
- 🤖 **Hybrid AI Chatbot** – TinyLlama (local, fast) + Groq fallback, with context‑aware RAG
- 💡 **Surprising Insights** – find counter‑intuitive findings automatically
- 🔬 **Comparative Analysis** – compare two papers side‑by‑side
- 📊 **Paper structure viewer** – outline of abstract, methods, results, conclusion
- 📥 **Full report download** – export summary, keywords, entities as Markdown
- 📜 **History** – all chats and summaries saved to Supabase
- 🧹 **Advanced text cleaning** – fixes concatenated words, page numbers, and PDF artifacts
- 🎨 **Floating chat widget** – ask questions while reading the summary
- 🔐 **User login/signup** – demo account `test@demo.com` / `password123`
- 📈 **Readability metrics** – Flesch Reading Ease, Gunning Fog Index
- 🔗 **Citation network** – shows most cited references
- 🏷️ **Named entity recognition** – extracts persons, organizations, dates, etc.
- 📊 **Word cloud & top keywords** – visual frequency analysis

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) (for local LLM)
- [Groq](https://groq.com) API key (optional, fallback)
- [Supabase](https://supabase.com) project (for history)

### 1. Clone the repo

git clone https://github.com/SharmanRawat/nlp-research-assistant.git
cd nlp-research-assistant

2. Create & activate Conda environment

conda env create -f environment.yml
conda activate nlp_bootcamp
If you don’t have Conda, use pip + virtualenv:


python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

3. Set up environment variables
Create a .env file in the root:

env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key
GROQ_API_KEY=your_groq_api_key   # optional
HF_TOKEN=your_hf_token           # optional

4. Download spaCy model
python -m spacy download en_core_web_sm

5. Pull the local LLM (TinyLlama)
ollama pull tinyllama:1.1b

6. Run the app
python app.py
Open http://localhost:5000 in your browser.

🧪 Demo Login
Use the demo account:

Email: test@demo.com

Password: password123

🛠️ Tech Stack
Backend: Flask, Supabase (PostgreSQL), ChromaDB, SpaCy, NLTK, Scikit‑learn
LLM: TinyLlama (Ollama) + Groq (Llama 3.1)
Embeddings: Sentence‑Transformers (all-MiniLM-L6-v2)
Frontend: HTML, Bootstrap, JavaScript
PDF extraction: pdfplumber, PyPDF2
Summarization: TextRank, TF‑IDF, Frequency-based

📁 Project Structure
text
nlp_project/
├── app.py                 # main Flask app
├── environment.yml        # Conda environment
├── requirements.txt       # pip dependencies
├── .env                   # secrets (ignored by git)
├── .gitignore
├── README.md
├── templates/             # HTML templates
│   ├── base.html
│   ├── login.html
│   ├── signup.html
│   ├── dashboard.html
│   ├── summary.html
│   ├── chat.html
│   ├── history.html
│   ├── insights.html
│   └── compare.html
└── static/
    └── style.css
🧠 How It Works
Upload – extracts text from PDF/DOCX/TXT, applies aggressive cleaning (fixes concatenated words, removes page numbers, journal headers).

Summarization – uses TextRank, TF‑IDF, or Frequency-based extraction; also offers abstractive summarization with PEGASUS (optional).

Chat – hybrid RAG: tries TinyLlama first (fast local), falls back to Groq if slow, then returns the most relevant chunk.

History – all chats and summaries are saved to Supabase and displayed in the History tab.

Insights – uses the LLM to find counter‑intuitive findings.

Compare – compares two papers side‑by‑side using Groq.

🧪 Troubleshooting
PDF extraction fails – ensure the PDF is not scanned; pdfplumber requires text-based PDFs.

Ollama takes too long – reduce timeout or let it fallback to Groq.

Groq API key missing – the app falls back to local TinyLlama.

Supabase history not saving – check your .env credentials and table schema.

🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first.

📄 License
MIT

🙏 Acknowledgements
Built for the NLP Bootcamp competition.
