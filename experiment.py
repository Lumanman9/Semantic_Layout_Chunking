import llm_qa_system
from fix_length_chunking.fix_length import FixLengthChunker
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

'''
#Chunking, input a whole text and segment it to multiple chunks
#fix_length_chunking
token_chunker = FixLengthChunker(length=200, granularity='token')
sentence_chunker = FixLengthChunker(length=6, granularity='sentence')
paragraph_chunker = FixLengthChunker(length=2, granularity='paragraph')

for name,text in texts.items():
    logger.info(name)

    chunks_token = token_chunker.split_text(text)
    chunks_sentence = sentence_chunker.split_text(text)
    chunks_paragraph = paragraph_chunker.split_text(text)

    embedder = Embedding(chunks=chunks_token,collection_name='fix_length_token')
    embedder.store_chunks(metadata=[{'name':name}]*len(chunks_token))

    embedder = Embedding(chunks=chunks_sentence,collection_name='fix_length_sentence')
    embedder.store_chunks(metadata=[{'name':name}]*len(chunks_sentence))

    embedder = Embedding(chunks=chunks_paragraph,collection_name='fix_length_paragraph')
    embedder.store_chunks(metadata=[{'name':name}]*len(chunks_paragraph))


#recursive_chunking
# Load a spaCy model
nlp = spacy.load("en_core_web_sm")

def spacy_token_count(text):
    doc = nlp(text)
    return len(doc)

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=spacy_token_count,
    separators=["\n", "。",".", ""]
)
for name,text in texts.items():
    chunks_recursive = [chunk.page_content for chunk in recursive_splitter.create_documents([text])]
    logger.info(chunks_recursive)
    embedder = Embedding(chunks=chunks_recursive, collection_name='recursive')
    embedder.store_chunks(metadata=[{'name':name}]*len(chunks_recursive))


#semantic_chunking_embedding
for name,text in texts.items():
    chunks_embedding = get_chunks_as_list(text)
    embedder = Embedding(chunks=chunks_embedding, collection_name='semantic_embedding')
    embedder.store_chunks(metadata=[{'name':name}]*len(chunks_embedding))

#semantic_chunking_bert
bert_chunker = SemanticChunkerBert()
for name, text in texts.items():
    chunks_bert = bert_chunker.create_chunks(text)
    embedder = Embedding(chunks=chunks_bert, collection_name='semantic_bert')
    embedder.store_chunks(metadata=[{'name':name}]*len(chunks_bert))


#semantic_layout_chunking
schema = ['author','abstract','title','introduction','methodology','related_work','experiment','background','dataset',
          'acknowledge','conclusion','result_discussion']

# Base directory where folders 1-9 are located
base_dir = "data"  # Change this to your actual base directory if needed

# Generate paths for each directory
for dir_num in range(1, 10):
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
        ocr_engine= "tesseract"
    )
    layout_data = extractor.process()

    chunks=[doc['text'] for doc in layout_data]
    metadata=[{'name':pdf_filename,'id':doc['id'],'page':doc['page'],'bbox':doc['bbox'],'label':doc['label']} for doc in layout_data]

    Embedder = Embedding(chunks=chunks,collection_name='semantic_layout')
    Embedder.store_chunks(chunks=chunks,metadata=metadata)
'''

# Evaluation
def evaluate_and_average_metrics(texts,collection_name,qa_path,agentic=False,label_path=None):
    all_results = {}

    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    model = "gpt-4-turbo-preview"


    if label_path is None:
        for name, text in texts.items():
            questions, answers = extract_qa_by_document(qa_path, name)
            retrieve = Retrieve(collection_name=collection_name, n_chunks=3, meta=[name])
            rag_llm = RAG_llm(client=client, model=model, retrieve=retrieve)

            # Initialize evaluator. The retrieve method could be 'plain' or 'agentic'.
            if agentic:
                QAE = QAAccuracyEvaluator(questions=questions, answers=answers, llm_system=rag_llm,
                                      retrieve_method='agentic')
            else:
                QAE = QAAccuracyEvaluator(questions=questions, answers=answers, llm_system=rag_llm,
                                      retrieve_method='plain')

            results = QAE.evaluate()
            all_results[name] = results
            print(f"\n----- Results for {name} -----")
            QAE.print_evaluation_results(results)

    if label_path:
    # qa_with_label
        for name, text in texts.items():
            retrieve = Retrieve(collection_name=collection_name, n_chunks=3, meta=[name])
            rag_llm = RAG_llm(client=client, model=model, retrieve=retrieve)
            questions=[]
            answers=[]
            labels=[]
            csv_path = label_path
            # Iterate through rows with tuple unpacking
            for file_name, question, answer, label in read_csv_by_tuple(csv_path):
                if file_name + '.pdf' == name:
                    questions.append(question)
                    answers.append(answer)
                    labels.append(label)
            print(labels)
            # Initialize evaluator. The retrieve method could be 'plain' or 'agentic'.
            QAE = QAAccuracyEvaluator(questions=questions, answers=answers, llm_system=rag_llm,
                                      retrieve_method='agentic', labels=labels)
            results = QAE.evaluate()
            all_results[name] = results
            print(f"\n----- Results for {name} -----")
            QAE.print_evaluation_results(results)

    # Calculate averages across all documents
    avg_results = calculate_and_print_averages(all_results,collection_name)

    if label_path:
        outfile = f'all_evaluation_results_{collection_name}_with_label.json'
    else:
        outfile = f'all_evaluation_results_{collection_name}_without_label.json'
    # Save all results (individual and average) to a file
    with open(outfile, 'w') as f:
        import json
        # Create a complete results dictionary that includes individual results and averages
        complete_results = {
            'individual_document_results': all_results,
            'average_metrics': avg_results
        }
        json.dump(complete_results, f, indent=2)

    print(f"Complete evaluation results have been saved to all_evaluation_results_{collection_name}_with_label.json")

    return all_results


def calculate_and_print_averages(all_results,collection_name):
    """Calculate and print average metrics across all documents."""
    # Initialize counters for each metric
    metrics = {
        'exact_match': [],
        'bleu_score': [],
        'rouge-1': {'precision': [], 'recall': [], 'f1': []},
        'rouge-2': {'precision': [], 'recall': [], 'f1': []},
        'rouge-L': {'precision': [], 'recall': [], 'f1': []}
    }

    # Collect metrics from all documents
    for doc_name, results in all_results.items():
        # Collect exact match and BLEU scores
        metrics['exact_match'].extend([q['exact_match'] for q in results['per_question_metrics']])
        metrics['bleu_score'].extend([q['bleu_score'] for q in results['per_question_metrics']])

        # Collect ROUGE scores
        for q in results['per_question_metrics']:
            for rouge_type in ['rouge-1', 'rouge-2', 'rouge-L']:
                for metric in ['precision', 'recall', 'f1']:
                    metrics[rouge_type][metric].append(q['rouge_scores'][rouge_type][metric])

    # Calculate averages
    avg_results = {
        'collection': collection_name,
        'average_exact_match': np.mean(metrics['exact_match']),
        'average_bleu_score': np.mean(metrics['bleu_score']),
        'average_rouge_scores': {
            'rouge-1': {
                'precision': np.mean(metrics['rouge-1']['precision']),
                'recall': np.mean(metrics['rouge-1']['recall']),
                'f1': np.mean(metrics['rouge-1']['f1'])
            },
            'rouge-2': {
                'precision': np.mean(metrics['rouge-2']['precision']),
                'recall': np.mean(metrics['rouge-2']['recall']),
                'f1': np.mean(metrics['rouge-2']['f1'])
            },
            'rouge-L': {
                'precision': np.mean(metrics['rouge-L']['precision']),
                'recall': np.mean(metrics['rouge-L']['recall']),
                'f1': np.mean(metrics['rouge-L']['f1'])
            }
        }
    }

    # Print average results
    print("\n===== AVERAGE METRICS ACROSS ALL DOCUMENTS =====")
    print(f"Average Exact Match: {avg_results['average_exact_match']:.4f}")
    print(f"Average BLEU Score: {avg_results['average_bleu_score']:.4f}")

    print("\nAverage ROUGE Scores:")
    for rouge_type in ['rouge-1', 'rouge-2', 'rouge-L']:
        print(f"  {rouge_type}:")
        for metric in ['precision', 'recall', 'f1']:
            value = avg_results['average_rouge_scores'][rouge_type][metric]
            print(f"    {metric}: {value:.4f}")

    # Save average metrics to a file
    with open(f'average_metrics_summary_{collection_name}_with_label.json', 'w') as f:
        import json
        json.dump(avg_results, f, indent=2)

    print(f"\nAverage metrics have been saved to 'average_metrics_summary_{collection_name}_with_label.json'")

    return avg_results



'''
### Process literature paper ###
dir_path = 'data/'
texts=load_pdfs(dir_path)
'''


'''
dir_path = 'Wikiperson_data/'
texts=load_pdfs(dir_path)
token_chunks(texts=texts,num=200,collection_name='Wikiperson')
'''
'''
dir_path = 'Wikiperson_data/'
texts=load_pdfs(dir_path)
schema = ['introduction', 'early_fife', 'career', 'contribution', 'award', 'legal_trouble', 'personal_experience','miscellaneous']
semantic_chunks(texts,schema,num=10,base_dir='Wikiperson_data',collection_name = 'Wikiperson')
'''

'''
dir_path = 'Financial_Report/'
texts=load_pdfs(dir_path)
token_chunks(texts=texts,num=200,collection_name='Financial')
'''
'''
dir_path = 'Financial_Report/'
texts=load_pdfs(dir_path)
schema = ['company_overview', 'manager_discussion', 'performance_summary', 'strategy_outlook', 'segment_business_review', 'risk_factors',
          'governance_and_leadership', 'sustainability', 'financial_statement', 'financial_statement_note', 'auditors_report', 'shareholder_information',
          'legal_compliance', 'miscellaneous']
semantic_chunks(texts,schema,num=3,base_dir='Financial_Report',collection_name = 'Financial')
'''


'''
dir_path = 'Wikiperson_data/'
texts=load_pdfs(dir_path)
qa_path = 'Wikiperson_data/feta_qa.csv'
all_results = evaluate_and_average_metrics(texts,'fix_length_token_Wikiperson200',qa_path=qa_path)
'''


'''
#### Token ####
#number of token
num = 500


'''



'''
####  Paragraph ####
#number of paragraph in a chunk
num = 1
#paragraph_chunks(texts,num)
qa_path = 'data/paper_text_qa.csv'
all_results = evaluate_and_average_metrics(texts,'fix_length_paragraph_1',qa_path=qa_path)

#### Semantic Layout Without Label ####


#### Semantic Layout With label ####
label_path = "data/QA_with_label.csv"
qa_path = 'data/paper_text_qa.csv'
all_results = evaluate_and_average_metrics(texts,'semantic_layout',qa_path=qa_path,label_path=label_path,agentic=True)
'''

#dir_path = 'Financial_Report/'
#dir_path = 'Wikiperson_data/'
dir_path = 'data/'
texts=load_pdfs(dir_path)
#recursive_layout(texts,num=500,overlap=0.1,collection_name = 'Literature')
#sentence_chunks(texts,num=6,collection_name='Literature')
#paragraph_chunks(texts,1,collection_name='Financial')
#semantic_layout_chunks(texts,num =11,base_dir='Wikiperson_data',schema=['introduction-'],collection_name = 'Wikiperson')
label_path = 'data/QA_with_label 2.csv'
qa_path = 'data/paper_text_qa.csv'
all_results = evaluate_and_average_metrics(texts,'semantic_layout',qa_path=qa_path,label_path=label_path,agentic=True)
#all_results = evaluate_and_average_metrics(texts,'fix_length_token',qa_path=qa_path)










