# 🔍 Multi-Model RAG Data Validator

> AI-Powered Data Quality Validation using RAG with HuggingFace, Google Gemini, and Grok

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📖 Overview

An intelligent data validation tool that uses **Retrieval-Augmented Generation (RAG)** and multiple Large Language Models to provide AI-powered insights into your data quality. Upload CSV, Excel, JSON, or TXT files and get comprehensive validation reports with actionable recommendations.

### Key Features

- 🤖 **Multi-Model AI Support** - HuggingFace, Google Gemini, and Grok (xAI)
- 📊 **7 Validation Techniques** - Type checking, null detection, duplicate detection, outlier detection, and more
- 📁 **Multiple File Formats** - CSV, Excel, JSON, TXT with auto-encoding detection
- 🎯 **Quality Scoring** - Production-ready quality assessment (0-100 scale)
- 📥 **Export Reports** - JSON, TXT, and CSV format reports

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/DheerajKumar97/multi-model-rag-validator.git
cd multi-model-rag-validator

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Requirements

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
huggingface-hub>=0.17.0
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
scipy>=1.10.0
fuzzywuzzy>=0.18.0
chardet>=5.1.0
openpyxl>=3.1.0
groq>=0.4.0
google-generativeai>=0.3.0
```

---

## 📋 Usage

1. **Configure AI Model** - Select provider (HuggingFace/Gemini/Grok) and enter API key
2. **Upload File** - Choose CSV, Excel, JSON, or TXT file
3. **Select Validations** - Pick which validation checks to run
4. **Run Validation** - Click "Run Validation" and wait for results
5. **Review & Export** - Check quality score and download reports

---

## 🔍 Validation Techniques

| Validation | Description |
|------------|-------------|
| **Data Type Check** | Validates column data types and consistency |
| **Range Check** | Detects statistical outliers (4σ rule) |
| **Null Value Check** | Identifies missing data |
| **Duplicate Detection** | Finds duplicate rows |
| **Outlier Detection** | IQR-based anomaly detection |
| **Data Integrity** | Business logic validation (negatives, infinites) |
| **Quality Scoring** | Overall quality score (0-100) with penalty breakdown |

---

## 🔑 API Keys Setup

### HuggingFace
1. Visit [huggingface.co](https://huggingface.co)
2. Go to Settings → Access Tokens
3. Create token with "Read" permission

### Google Gemini
1. Visit [ai.google.dev](https://ai.google.dev)
2. Get API key from Google AI Studio

### Grok (via Groq)
1. Visit [console.groq.com](https://console.groq.com)
2. Create API key in dashboard

---

## 📊 Quality Scoring

- **95-100** 🟢 Perfect - Production-ready
- **90-94** 🟢 Excellent - High quality
- **80-89** 🟡 Good - Minor improvements needed
- **70-79** 🟠 Fair - Attention required
- **60-69** 🔴 Poor - Not production-ready
- **<60** 🔴 Critical - Major issues

---

## 📁 Project Structure

```
multi-model-rag-validator/
│
├── app.py                 # Streamlit UI (Frontend)
├── main.py                # Backend Logic (FileHandler, LLMClient, Validator)
├── requirements.txt       # Python dependencies
└── README.md             # Documentation
```

### Core Components

**app.py** - All UI components and user interactions

**main.py** - Three main classes:
- `FileHandler` - Handles file parsing with encoding detection
- `MultiModelLLMClient` - Unified interface for HuggingFace, Gemini, Grok
- `ComprehensiveRAGValidator` - RAG-based validation with 7 techniques

---

## 🛠 Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **AI/ML**: HuggingFace Hub, Google Generative AI, Groq
- **Vector Search**: FAISS, Sentence Transformers
- **Utilities**: CharDet, OpenPyXL, FuzzyWuzzy

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 📞 Contact

**Dheeraj Kumar K**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/dheerajkumar1997/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/DheerajKumar97)
[![Website](https://img.shields.io/badge/Website-FF7139?style=flat&logo=Firefox&logoColor=white)](https://dheeraj-kumar-k.lovable.app/)

---

<div align="center">

**Made with ❤️ by Dheeraj Kumar K**

© 2025 All Rights Reserved

</div>