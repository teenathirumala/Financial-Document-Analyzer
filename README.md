# 🧾 Financial Document Analyzer

An AI-powered pipeline to extract, analyze, and summarize insights from scanned invoices using LayoutLMv3, pytesseract, and classic ML.

---

## 🚀 Features

- 📄 Extracts vendor, date, and total amount from scanned PDFs using **pytesseract** and **LayoutLMv3**
- 🧠 Fine-tunes **LayoutLMv3** on the **SROIE dataset** for named entity recognition (NER)
- 📊 Detects anomalies using **Isolation Forest** and groups vendors using **KMeans clustering**
- 📈 Interactive **Streamlit dashboard** for invoice analytics and trend visualization
- 📦 Fully **Dockerized** for reproducibility and deployment

---

## 🛠 Tech Stack

- **NLP & Transformers**: LayoutLMv3, Hugging Face Transformers, TokenClassification
- **OCR**: pytesseract, PDF2Image
- **ML**: scikit-learn (Isolation Forest, KMeans), seqeval (F1, accuracy)
- **Visualization**: Streamlit, Seaborn, Matplotlib
- **Containerization**: Docker
- **Languages**: Python

---

## 🧪 Evaluation

- Fine-tuned on SROIE dataset (1K+ invoices)
- Achieved **94% F1 score** on entity extraction (vendor, total, date)
- Includes train/test split and evaluation using `seqeval`

---

## 📁 Project Structure
financial-doc-analyzer/
├── app/ # Streamlit app + inference
├── training/ # Fine-tuning scripts
│ ├── prepare_data.py
│ ├── dataset.py
│ └── train_model.py
├── data/SROIE2019/ # SROIE images + label files
├── models/ # Saved fine-tuned model
├── requirements.txt
└── Dockerfile

---

## 🧪 Setup & Usage

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/financial-doc-analyzer.git
cd financial-doc-analyzer
```
### Create a virtual environment and install dependencies

```bash 
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
### Run training
```bash
Copy
Edit
python training/prepare_data.py
python training/train_model.py
```
### Run the Streamlit app
```bash
Copy
Edit
streamlit run dashboard/streamlit_app.py
```

### Run with Docker
```bash
docker build -t doc-analyzer .
docker run -p 8501:8501 doc-analyzer
```
