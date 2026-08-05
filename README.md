# 🧠 Transparent NLP Research Paper Analyzer

An interactive, multi‑stage NLP analysis tool for research papers, with local LLM Q&A, summarization, readability metrics, citation network, and more.

## ✨ Features

- 🔒 **Safer Preprocessing** – preserves acronyms, URLs, emojis, hashtags, with audit log.
- 📊 **Vectorization** – TF‑IDF, word embeddings, cosine similarity heatmap.
- 🧪 **Classifier** – Naive Bayes, Logistic Regression, SVM with feature weights.
- 📈 **Hype vs Rigor** – scores marketing buzzwords vs. quantitative claims.
- 🔗 **Concept Graph** – co‑occurrence graph of key terms.
- 📝 **Summarization** – TextRank, TF‑IDF, Frequency‑based with visual bullet points.
- 🏷️ **Entities & Keywords** – NER and TF‑IDF keyword extraction.
- 📊 **Paper Stats** – readability (Flesch, Gunning Fog), word cloud, frequency chart.
- 🤖 **Paper Q&A** – local LLM (Qwen2.5‑Coder via Ollama) with RAG – 100% offline.
- 🔗 **Citation Network** – most cited authors.
- 🏷️ **Research Field Classification** – automatically detects domain.
- 📊 **Section Readability Heatmap** – highlights hardest sections.
- 🔬 **Methodology Extractor** – sample size, design, analysis.
- 📤 **Export** – download summary as Markdown or full text.

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Pull the model: `ollama pull qwen2.5-coder:1.5b`

# 🧠 Transparent NLP Research Paper Analyzer

## Installation (with Conda)

1. Clone the repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/nlp-research-assistant.git
   cd nlp-research-assistant
