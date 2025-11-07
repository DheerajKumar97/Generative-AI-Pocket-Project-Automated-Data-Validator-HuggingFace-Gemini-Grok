# 🔍 Multi-Model RAG Data Validator

**AI-Powered Data Quality Validation with HuggingFace, Gemini, and Grok (xAI)**  

![Streamlit App](https://img.shields.io/badge/Built%20with-Streamlit-red?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 🧠 Overview  

**Multi-Model RAG Data Validator** is an **AI-driven Streamlit application** that performs advanced **data quality validation** using **Retrieval-Augmented Generation (RAG)** and **multiple LLM providers** — **HuggingFace**, **Google Gemini**, and **Grok (xAI)**.

It validates your dataset for **missing values, duplicates, outliers, integrity issues**, and assigns an **AI-generated quality score** using LLM reasoning and context retrieval.

---

## ✨ Key Features  

### 🤖 Multi-Model AI Support  
- **HuggingFace:** Qwen, LLaMA, Mixtral  
- **Google Gemini:** Gemini-2.0-Flash, Gemini-1.5-Pro  
- **Grok (xAI):** LLaMA-3.3, Mixtral-8x7B  

### 📊 Comprehensive Data Validations  
| Validation | Description |
|-------------|-------------|
| 🧩 **Data Type Check** | Ensures correct schema & column types |
| 📉 **Range Check** | Detects out-of-range or constant numeric values |
| 🧱 **Null Value Check** | Identifies missing and incomplete data |
| 🔁 **Duplicate Detection** | Detects duplicate or redundant records |
| 🚨 **Outlier Detection** | Flags statistical anomalies (IQR & Z-score) |
|⚙️ **Data Integrity Check** | Ensures logical consistency and valid business rules |
| 🧮 **Quality Scoring** | AI-based scoring of completeness, validity & consistency |

---

## 🧩 Tech Stack  

- **Frontend:** [Streamlit](https://streamlit.io)  
- **Language:** Python 3.10+  
- **AI APIs:**  
  - 🤗 [HuggingFace Inference API](https://huggingface.co/inference-api)  
  - ✨ [Google Gemini API](https://ai.google.dev)  
  - 🚀 [Groq (xAI)](https://console.groq.com)  
- **Libraries:**  
  `pandas`, `numpy`, `faiss`, `sentence-transformers`, `scipy`, `chardet`,  
  `google-generativeai`, `huggingface_hub`, `groq`, `openpyxl`, `fuzzywuzzy`

---

## ⚙️ Installation  

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/MultiModel-RAG-Data-Validator.git
cd MultiModel-RAG-Data-Validator
