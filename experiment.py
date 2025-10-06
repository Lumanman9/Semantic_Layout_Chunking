import llm_qa_system
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llm_qa_system.pdf_to_text import PDFConverter
from loguru import logger
from llm_qa_system.embedding import Embedding
from loguru import logger
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

#from retreive_result import dir_path
from test.semantic_embedding_test import get_chunks_as_list
from semantic_chunking.semantic_BERT import SemanticChunkerBert
from OCR.OCR import PDFTextExtractor
from evaluation import QAAccuracyEvaluator
from llm_qa_system.llm_qa import RAG_llm
from llm_qa_system.retreival import Retrieve
from qa_preprocessing import extract_qa_by_document
import openai
import os
import numpy as np
import spacy
from pathlib import Path
from read_csv import read_csv_by_tuple
from evaluation import evaluate_and_average_metrics, calculate_and_print_averages
from pathlib import Path


load_dotenv()


#Input PDF, output texts with (name,text), name is the file_name, text is the extracted text
def load_pdfs(path):
    pdfconverter = PDFConverter()
    texts = pdfconverter.convert_pdf_directory(dir_path,recursive=True)
    logger.info(texts.keys())
    return texts

def paragraph_chunks(texts,num,collection_name = None):
    paragraph_chunker = FixLengthChunker(length=num, granularity='paragraph')
    for name, text in texts.items():
        logger.info(name)
        chunks_paragraph = paragraph_chunker.split_text(text)
        embedder = Embedding(chunks=chunks_paragraph, collection_name=f'fix_length_paragraph_{collection_name}{num}')
        embedder.store_chunks(metadata=[{'name': name}] * len(chunks_paragraph))
    return None

def token_chunks(texts,num,collection_name = None):
    token_chunker = FixLengthChunker(length=num, granularity='token')
    for name, text in texts.items():
        logger.info(name)
        chunks_token = token_chunker.split_text(text)
        embedder = Embedding(chunks=chunks_token, collection_name=f'fix_length_token_{collection_name}{num}')
        embedder.store_chunks(metadata=[{'name': name}] * len(chunks_token))
    return None

def sentence_chunks(texts,num,collection_name = None):
    sentence_chunker = FixLengthChunker(length=num, granularity='sentence')
    for name, text in texts.items():
        logger.info(name)
        chunks_sentence = sentence_chunker.split_text(text)
        embedder = Embedding(chunks=chunks_sentence, collection_name=f'fix_length_sentence_{collection_name}{num}')
        embedder.store_chunks(metadata=[{'name': name}] * len(chunks_sentence))
    return None

def semantic_layout_chunks(texts,schema,num,base_dir,collection_name = None):

    # Generate paths for each directory
    for dir_num in range(1, num+1):
        dir_path = os.path.join(base_dir, str(dir_num))

        # Find the PDF and JSON files in this directory
        pdf_files = list(Path(dir_path).glob("*.pdf"))
        json_files = list(Path(dir_path).glob("*.json"))

        # Get the first PDF and JSON file (assuming there's one of each)
        pdf_path = str(pdf_files[0]) if pdf_files else None
        json_path = str(json_files[0]) if json_files else None
        pdf_filename = os.path.basename(pdf_path) if pdf_path else None
        print(pdf_path)
        print(json_path)
        print(pdf_filename)

        extractor = PDFTextExtractor(
            pdf_path=pdf_path,
            json_path=json_path,
            schema=schema,
            ocr_engine="tesseract"
        )
        layout_data = extractor.process()

        chunks = [doc['text'] for doc in layout_data]
        metadata = [
            {'name': pdf_filename, 'id': doc['id'], 'page': doc['page'], 'bbox': doc['bbox'], 'label': doc['label']} for
            doc in layout_data]

        Embedder = Embedding(chunks=chunks, collection_name=f'semantic_layout{collection_name}')
        Embedder.store_chunks(chunks=chunks, metadata=metadata)
    return None


def semantic_layout(texts,collection_name = None):
    for name, text in texts.items():
        chunks_embedding = get_chunks_as_list(text)
        embedder = Embedding(chunks=chunks_embedding, collection_name=f'semantic_embedding{collection_name}')
        embedder.store_chunks(metadata=[{'name': name}] * len(chunks_embedding))

def recursive_layout(texts,num,overlap=0.1,collection_name = None):
    nlp = spacy.load("en_core_web_sm")

    def spacy_token_count(text):
        doc = nlp(text)
        return len(doc)

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=num,
        chunk_overlap=num*overlap,
        length_function=spacy_token_count,
        separators=["\n", "。", ".", ""]
    )
    for name, text in texts.items():
        chunks_recursive = [chunk.page_content for chunk in recursive_splitter.create_documents([text])]
        logger.info(chunks_recursive)
        embedder = Embedding(chunks=chunks_recursive, collection_name=f'recursive_{collection_name}{num}')
        embedder.store_chunks(metadata=[{'name': name}] * len(chunks_recursive))

# Decorator for chunking
CHUNKING_METHODS = {}
def chunking_method(method_name):
    def decorator(func):
        func._method_name = method_name
        # Auto-register the method
        CHUNKING_METHODS[method_name] = func
        return func
    return decorator
def run_chunking_method(method_name = 'token_chunker'):
    if method_name not in CHUNKING_METHODS:
        raise ValueError(f"unknown chunking method {method_name}. Available: {list(CHUNKING_METHODS.keys())}")
    return CHUNKING_METHODS[method_name]()

@chunking_method("token_chunker")
def token_chunker(texts, length=200, qa_path):
    from fix_length_chunking.fix_length import FixLengthChunker
    token_chunker = FixLengthChunker(length=length, granularity='token')
    for name, text in texts.items():
        logger.info(name)
        chunks_token = token_chunker.split_text(text)
        embedder = Embedding(chunks=chunks_token, collection_name='fix_length_token')
        embedder.store_chunks(metadata=[{'name': name}] * len(chunks_token))
    all_results = evaluate_and_average_metrics(texts, 'fix_length_token', qa_path=qa_path)
    return all_results

@chunking_method("sentence_chunker")
def sentence_chunker(texts, length=5):
    from fix_length_chunking.fix_length import FixLengthChunker
    sentence_chunker = FixLengthChunker(length=length, granularity='sentence')
    for name, text in texts.items():
        logger.info(name)
        chunks_sentence = sentence_chunker.split_text(text)
        embedder = Embedding(chunks=chunks_sentence, collection_name='fix_length_sentence')
        embedder.store_chunks(metadata=[{'name': name}] * len(chunks_sentence))
    all_results = evaluate_and_average_metrics(texts, 'fix_length_sentence', qa_path=qa_path)
    return all_results

@chunking_method("paragraph_chunker")
def paragraph_chunker(texts, length=2):
    from fix_length_chunking.fix_length import FixLengthChunker
    paragraph_chunker = FixLengthChunker(length=length, granularity='paragraph')
    for name, text in texts.items():
        logger.info(name)
        chunks_paragraph = paragraph_chunker.split_text(text)
        embedder = Embedding(chunks=chunks_paragraph, collection_name='fix_length_paragraph')
        embedder.store_chunks(metadata=[{'name': name}] * len(chunks_paragraph))
    all_results = evaluate_and_average_metrics(texts, 'fix_length_paragraph', qa_path=qa_path)
    return all_results

@chunking_method("recursive_chunker")
def recursive_chunker(texts, chunk_size= 500, chunk_overlap= 50):
    # Load a spaCy model
    nlp = spacy.load("en_core_web_sm")
    def spacy_token_count(text):
        doc = nlp(text)
        return len(doc)

    recursive_chunker = RecursiveCharacterTextSplitter(
        chunk_size= chunk_size,
        chunk_overlap= chunk_overlap,
        length_function= spacy_token_count,
        separators=["\n", "。", ".", ""]
    )
    for name, text in texts.items():
        chunks_recursive = [chunk.page_content for chunk in recursive_chunker.create_documents([text])]
        logger.info(chunks_recursive)
        embedder = Embedding(chunks=chunks_recursive, collection_name='recursive')
        embedder.store_chunks(metadata=[{'name': name}] * len(chunks_recursive))

@chunking_method("semantic_chunker")
def semantic_chunker(texts):
    from llm_qa_system.embedding import Embedding
    for name,text in texts.items():
        chunks_embedding = get_chunks_as_list(text)
        embedder = Embedding(chunks=chunks_embedding, collection_name='semantic_chunking')
        embedder.store_chunks(metadata=[{'name':name}]*len(chunks_embedding))
    all_results = evaluate_and_average_metrics(texts, 'semantic_chunking', qa_path=qa_path)
    return all_results

@chunking_method("semantic_layout_chunker")
def semantic_layout_chunker(texts, schema, base_dir, qa_path, label_path):
    from llm_qa_system.embedding import Embedding
    # Find all subdirectories that contain JSON files matching the pattern
    for root, dirs, files in os.walk(base_dir):
        # Check if any file matches the pattern annotations_layout_*.json
        json_files = [f for f in files if f.startswith('annotations_layout_') and f.endswith('.json') and len(
            f.split('_')[2].split('.')[0]) == 3]

        if json_files:
            # Get the first matching JSON file
            json_filename = json_files[0]
            json_path = os.path.join(root, json_filename)

            # Find PDF files in the same directory
            pdf_files = list(Path(root).glob("*.pdf"))
            pdf_path = str(pdf_files[0]) if pdf_files else None
            pdf_filename = os.path.basename(pdf_path) if pdf_path else None

            print(f"Processing directory: {root}")
            print(f"PDF path: {pdf_path}")
            print(f"JSON path: {json_path}")
            print(f"PDF filename: {pdf_filename}")

            if pdf_path and json_path:
                extractor = PDFTextExtractor(
                    pdf_path=pdf_path,
                    json_path=json_path,
                    schema=schema,
                    ocr_engine="tesseract"
                )
                layout_data = extractor.process()

                chunks = [doc['text'] for doc in layout_data]
                metadata = [
                    {'name': pdf_filename, 'id': doc['id'], 'page': doc['page'], 'bbox': doc['bbox'],
                     'label': doc['label']} for doc in layout_data
                ]

                Embedder = Embedding(chunks=chunks, collection_name='semantic_layout')
                Embedder.store_chunks(chunks=chunks, metadata=metadata)
    all_results = evaluate_and_average_metrics(texts, 'semantic_layout', qa_path=qa_path,
                        label_path=label_path, agentic=True)
    return all_results
def experiment(chunking_method, dataset, length):
    base_dirs = {
        'Literature_Paper': 'dataset/Literature_Paper',
        'Financial_Report': 'dataset/Financial_Report',
        'Wikiperson': 'dataset/Wikiperson_data'
    }

    schemas = {
        'Literature_Paper': ['author','abstract','title','introduction','methodology','related_work','experiment','background','dataset',
          'acknowledge','conclusion','result_discussion'],
        'Financial_Report': ['company_overview', 'manager_discussion', 'performance_summary', 'strategy_outlook', 'segment_business_review', 'risk_factors',
          'governance_and_leadership', 'sustainability', 'financial_statement', 'financial_statement_note', 'auditors_report', 'shareholder_information',
          'legal_compliance', 'miscellaneous'],
        'Wikiperson': ['introduction', 'early_fife', 'career', 'contribution', 'award', 'legal_trouble', 'personal_experience','miscellaneous']
    }

    qa_paths = {
        'Literature_Paper': 'dataset/Literature_Paper/qa.csv',
        'Financial_Report': 'dataset/Financial_Report/qa_csv',
        'Wikiperson': 'dataset/Wikiperson_data/qa_csv'
    }

    label_paths = {
        'Literature_Paper': 'dataset/Literature_Paper/qa_with_label.csv',
        'Financial_Report': 'dataset/Financial_Report/qa_with_label.csv',
        'Wikiperson': 'dataset/Wikiperson_data/qa_with_label.csv'
    }

    base_dir = base_dirs[dataset]
    schema = schemas[dataset]
    qa_path = qa_paths[dataset]
    label_path = label_paths[dataset]
    texts = load_pdfs(dir_path)

    chunker=run_chunking_method(chunking_method)
    if chunking_method == 'semantic_layout_chunker':
        chunker(texts, schema, base_dir, qa_path, label_path)
    elif chunking_method in ['token_chunker', 'sentence_chunker', 'paragraph_chunker']:
        chunker(texts, length)
    elif chunking_method == 'recursive_chunker':
        chunker(texts, length, int(length * 0.1))  # chunk_size and chunk_overlap
    elif chunking_method == 'semantic_chunker':
        chunker(texts)

    print(all_results)



def main():
    import argparse
    paser = argparse.ArgumentParser()
    paser.add_argument('--chunking_method', type=str, default= 'token_chunker', choices=['token_chunker',
        'sentence_chunker', 'paragraph_chunker', 'recursive_chunker', 'semantic_chunker', 'semantic_layout_chunker'],
        help= 'choose chunking methods from token_chunker, sentence_chunker, paragraph_chunker, recursive_chunker,'
        'semantic_chunker, semantic_layout_chunker')
    paser.add_argument('--dataset', type= str, default= 'Literature_Paper', choices=['Literature_Paper',
        'Financial_Report', 'Wikiperson'], help= 'choose dataset name to test')
    paser.add_argument('--length', type=int, default=None, help='The length used in boundary_aware and recursive chunking')
    paser.add_argument('--output_path', type=str, default='./output/', help='The output path showing the results')

if __name__ == '__main__':
    main()









