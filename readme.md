# Enhancing RAG System Performance Through Semantic Layout Chunking
This repository contains the implementation and datasets for "Enhancing RAG System Performance Through Semantic Layout Chunking".

## 🎯 Overview
Retrieval-Augmented Generation (RAG) systems rely heavily on effective document chunking strategies. While most existing methods focus on either semantic coherence or structural boundaries, our approach combines both by preserving semantic layout information during the chunking process.

- **Novel Chunking Strategy:** Introduces semantic layout-based chunking that preserves both structural integrity and semantic flow.
- **Comprehensive Evaluation:** Systematic comparison across three document domains (literature papers, financial reports, Wikipedia pages).
- **Annotated Dataset:** 70 documents with semantic layout labels across multiple domains.

## 📊 Results
| Domain | Method | Retrieval Accuracy |
|:------:|:------:|:------------------:|
| Literature Paper | FixedToken | |
| | FixedSentence | |
| | FiexedParagraph | |


## 🏗️ Architecture
Our approach consists of four main stages:
![Overview](scr/ExperimentOverview.png)

1. Document Layout Detection: Using DocLayout-YOLO to identify structural elements
2. Use a human annotation tool Doc2KG (https://app.ai4wa.com/) to verify the results.
3. Semantic Layout Annotation: Mapping structural elements to semantic roles
4. Text Extraction & Post-processing: OCR(Tesseract-OCR) and PDFPlumber extraction with semantic label preservation.


## 🚀 Quick Start
**Installation**
```bash
git clone https://github.com/Lumanman9/Semantic_Layout_Chunking.git
cd Semantic_Layout_Chunking
pip install -r requirements.txt
```
**Running Experiments**
```bash
python experiment.py
```

## 📁 Dataset

**UDA Benchmark Subset** 

Original Dataset Link: https://github.com/qinchuanhui/UDA-Benchmark




