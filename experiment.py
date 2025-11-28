from langchain.text_splitter import RecursiveCharacterTextSplitter
from llm_qa_system.pdf_to_text import PDFConverter
from llm_qa_system.embedding import Embedding
from loguru import logger
from dotenv import load_dotenv
from OCR.OCR import PDFTextExtractor
import os
import spacy
from evaluation import evaluate_and_average_metrics
from pathlib import Path
from qdrant_client import QdrantClient


load_dotenv()


#Input PDF, output texts with (name,text), name is the file_name, text is the extracted text
def load_pdfs(path):
    pdfconverter = PDFConverter()
    texts = pdfconverter.convert_pdf_directory(path,recursive=True)
    logger.info(texts.keys())
    return texts

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
    return CHUNKING_METHODS[method_name]

@chunking_method("token_chunker")
def token_chunker(texts, length=200, qa_path=None, output_dir='./output'):
    from fix_length_chunking.fix_length import FixLengthChunker
    token_chunker = FixLengthChunker(length=length, granularity='token')
    collection_name = 'fix_length_token'
    
    # Delete collection if it exists
    qdrant = QdrantClient(os.getenv('qdrant_url'), port=os.getenv('qdrant_port'))
    collections = qdrant.get_collections().collections
    if any(c.name == collection_name for c in collections):
        logger.info(f"Deleting existing collection '{collection_name}'")
        qdrant.delete_collection(collection_name=collection_name)
    
    # Process and store chunks for each document
    for name, text in texts.items():
        logger.info(name)
        chunks_token = token_chunker.split_text(text)
        embedder = Embedding(chunks=chunks_token, collection_name=collection_name)
        embedder.store_chunks(metadata=[{'name': name}] * len(chunks_token))
    all_results = evaluate_and_average_metrics(texts, collection_name, qa_path=qa_path, output_dir=output_dir)
    return all_results

@chunking_method("sentence_chunker")
def sentence_chunker(texts, length=5, qa_path=None, output_dir='./output'):
    from fix_length_chunking.fix_length import FixLengthChunker
    sentence_chunker = FixLengthChunker(length=length, granularity='sentence')
    collection_name = 'fix_length_sentence'
    
    # Delete collection if it exists
    qdrant = QdrantClient(os.getenv('qdrant_url'), port=os.getenv('qdrant_port'))
    collections = qdrant.get_collections().collections
    if any(c.name == collection_name for c in collections):
        logger.info(f"Deleting existing collection '{collection_name}'")
        qdrant.delete_collection(collection_name=collection_name)
    
    # Process and store chunks for each document
    for name, text in texts.items():
        logger.info(name)
        chunks_sentence = sentence_chunker.split_text(text)
        embedder = Embedding(chunks=chunks_sentence, collection_name=collection_name)
        embedder.store_chunks(metadata=[{'name': name}] * len(chunks_sentence))
    all_results = evaluate_and_average_metrics(texts, collection_name, qa_path=qa_path, output_dir=output_dir)
    return all_results

@chunking_method("paragraph_chunker")
def paragraph_chunker(texts, length=2, qa_path=None, output_dir='./output'):
    from fix_length_chunking.fix_length import FixLengthChunker
    paragraph_chunker = FixLengthChunker(length=length, granularity='paragraph')
    collection_name = 'fix_length_paragraph'
    
    # Delete collection if it exists
    qdrant = QdrantClient(os.getenv('qdrant_url'), port=os.getenv('qdrant_port'))
    collections = qdrant.get_collections().collections
    if any(c.name == collection_name for c in collections):
        logger.info(f"Deleting existing collection '{collection_name}'")
        qdrant.delete_collection(collection_name=collection_name)
    
    # Process and store chunks for each document
    for name, text in texts.items():
        logger.info(name)
        chunks_paragraph = paragraph_chunker.split_text(text)
        embedder = Embedding(chunks=chunks_paragraph, collection_name=collection_name)
        embedder.store_chunks(metadata=[{'name': name}] * len(chunks_paragraph))
    all_results = evaluate_and_average_metrics(texts, collection_name, qa_path=qa_path, output_dir=output_dir)
    return all_results

@chunking_method("recursive_chunker")
def recursive_chunker(texts, chunk_size= 500, chunk_overlap= 50, qa_path=None, output_dir='./output'):
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
    collection_name = 'recursive'
    
    # Delete collection if it exists
    qdrant = QdrantClient(os.getenv('qdrant_url'), port=os.getenv('qdrant_port'))
    collections = qdrant.get_collections().collections
    if any(c.name == collection_name for c in collections):
        logger.info(f"Deleting existing collection '{collection_name}'")
        qdrant.delete_collection(collection_name=collection_name)
    
    # Process and store chunks for each document
    for name, text in texts.items():
        chunks_recursive = [chunk.page_content for chunk in recursive_chunker.create_documents([text])]
        logger.info(chunks_recursive)
        embedder = Embedding(chunks=chunks_recursive, collection_name=collection_name)
        embedder.store_chunks(metadata=[{'name': name}] * len(chunks_recursive))
    all_results = evaluate_and_average_metrics(texts, collection_name, qa_path=qa_path, output_dir=output_dir)
    return all_results

@chunking_method("semantic_chunker")
def semantic_chunker(texts, qa_path, output_dir='./output'):
    from llm_qa_system.embedding import Embedding
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_openai.embeddings import OpenAIEmbeddings
    def get_chunks_as_list(text):
        text_splitter = SemanticChunker(OpenAIEmbeddings())
        docs = text_splitter.create_documents([text])
        chunks = [doc.page_content for doc in docs]
        return chunks
    collection_name = 'semantic_chunking'
    
    # Delete collection if it exists
    qdrant = QdrantClient(os.getenv('qdrant_url'), port=os.getenv('qdrant_port'))
    collections = qdrant.get_collections().collections
    if any(c.name == collection_name for c in collections):
        logger.info(f"Deleting existing collection '{collection_name}'")
        qdrant.delete_collection(collection_name=collection_name)
    
    # Process and store chunks for each document
    for name,text in texts.items():
        chunks_embedding = get_chunks_as_list(text)
        embedder = Embedding(chunks=chunks_embedding, collection_name=collection_name)
        embedder.store_chunks(metadata=[{'name':name}]*len(chunks_embedding))
    all_results = evaluate_and_average_metrics(texts, collection_name, qa_path=qa_path, output_dir=output_dir)
    return all_results

@chunking_method("semantic_layout_chunker")
def semantic_layout_chunker(texts, schema, base_dir, qa_path, label_path, output_dir='./output'):
    from llm_qa_system.embedding import Embedding
    collection_name = 'semantic_layout'
    
    # Delete collection if it exists
    qdrant = QdrantClient(os.getenv('qdrant_url'), port=os.getenv('qdrant_port'))
    collections = qdrant.get_collections().collections
    if any(c.name == collection_name for c in collections):
        logger.info(f"Deleting existing collection '{collection_name}'")
        qdrant.delete_collection(collection_name=collection_name)
    
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

                Embedder = Embedding(chunks=chunks, collection_name=collection_name)
                Embedder.store_chunks(chunks=chunks, metadata=metadata)
    all_results = evaluate_and_average_metrics(texts, collection_name, qa_path=qa_path,
                        label_path=label_path, agentic=True, output_dir=output_dir)
    return all_results
def experiment(chunking_method, dataset, length, output_dir):
    base_dirs = {
        'Literature_Paper': 'dataset/Literature_Paper',
        'Financial_Report': 'dataset/Financial_Report',
        'Wikiperson': 'dataset/Wikiperson_data'
    }

    schemas = {
        'Literature_Paper': ['author','abstract','introduction','methodology','related_work','experiment','background','dataset',
          'acknowledge','conclusion','result_discussion'],
        'Financial_Report': ['company_overview', 'manager_discussion', 'performance_summary', 'strategy_outlook', 'segment_business_review', 'risk_factors',
          'governance_and_leadership', 'sustainability', 'financial_statement', 'financial_statement_note', 'auditors_report', 'shareholder_information',
          'legal_compliance', 'miscellaneous'],
        'Wikiperson': ['introduction', 'early_fife', 'career', 'contribution', 'award', 'legal_trouble', 'personal_experience','miscellaneous']
    }

    qa_paths = {
        'Literature_Paper': 'dataset/Literature_Paper/qa.csv',
        'Financial_Report': 'dataset/Financial_Report/qa.csv',
        'Wikiperson': 'dataset/Wikiperson_data/qa.csv'
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
    texts = load_pdfs(base_dir)

    chunker=run_chunking_method(chunking_method)
    if chunking_method == 'semantic_layout_chunker':
        chunker(texts, schema, base_dir, qa_path, label_path, output_dir)
    elif chunking_method in ['token_chunker', 'sentence_chunker', 'paragraph_chunker']:
        chunker(texts, length,qa_path, output_dir)
    elif chunking_method == 'recursive_chunker':
        chunker(texts, length, int(length * 0.1), qa_path, output_dir)  # chunk_size and chunk_overlap
    elif chunking_method == 'semantic_chunker':
        chunker(texts, qa_path, output_dir)

def main():
    import argparse
    paser = argparse.ArgumentParser()
    paser.add_argument('--chunking_method', type=str, default= 'token_chunker', choices=['token_chunker',
        'sentence_chunker', 'paragraph_chunker', 'recursive_chunker', 'semantic_chunker', 'semantic_layout_chunker'],
        help= 'choose chunking methods from token_chunker, sentence_chunker, paragraph_chunker, recursive_chunker,'
        'semantic_chunker, semantic_layout_chunker')
    paser.add_argument('--dataset', type= str, default= 'Literature_Paper', choices=['Literature_Paper',
        'Financial_Report', 'Wikiperson'], help= 'choose dataset name to test')
    paser.add_argument('--length', type=int, default=500, help='The length used in boundary_aware and recursive chunking')
    paser.add_argument('--output_dir', type=str, default='./output/', help='The output path showing the results')
    args = paser.parse_args()
    experiment(args.chunking_method, args.dataset, args.length, args.output_dir)


if __name__ == '__main__':
    main()









