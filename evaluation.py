import asyncio
import collections
import math
import os
from typing import List, Dict, Union, Optional, Any

import matplotlib.pyplot as plt
import nltk
import numpy as np
import openai
from dotenv import load_dotenv
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from fix_length_chunking.fix_length import FixLengthChunker
from llm_qa_system.embedding import Embedding
from llm_qa_system.llm_qa import RAG_llm
from llm_qa_system.retreival import Retrieve
from qa_preprocessing import extract_qa_by_document
from read_csv import read_csv_by_tuple


load_dotenv()


class RetrieveEfficiencyEvaluator:
    def __init__(self, file, chunker, collection_name):
        self.file = file
        self.chunker = chunker
        self.collection_name = collection_name

    def evaluate(self):
        with open(self.file, "r", encoding='utf-8') as f:
            text = f.read()

        chunks = self.chunker.split_text(text)
        chunk_lengths = [len(chunk) for chunk in chunks]
        counter = collections.Counter(chunk_lengths)

        results = {
            'collection_name': self.collection_name,
            'text_length': len(text),
            'num_chunks': len(chunks),
            'chunk_size_distribution': counter
        }

        plt.figure(figsize=(8, 6))
        plt.hist(chunk_lengths, bins=10, color='skyblue', edgecolor='black')
        plt.title("Distribution of Chunk Lengths")
        plt.xlabel("Chunk Length")
        plt.ylabel("Frequency")
        plt.show()

        return results


class RetrieveAccuracyEvaluator:
    def __init__(self, standard: str, chunks: Union[str, List[str]], collection_name: str):
        self.standard = standard
        self.collection_name = collection_name

        if isinstance(chunks, str):
            self.chunks = [chunks]
        else:
            self.chunks = chunks

        self.embedder = Embedding(chunks=self.chunks, collection_name=collection_name)

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def CosineSimilarity(self) -> float:
        try:
            chunks_text = ' '.join(self.chunks)
            chunks_embedding = self.embedder.generate_embedding(chunks_text)
            standard_embedding = self.embedder.generate_embedding(self.standard)

            dot_product = sum(c * s for c, s in zip(chunks_embedding, standard_embedding))
            chunks_magnitude = math.sqrt(sum(x * x for x in chunks_embedding))
            standard_magnitude = math.sqrt(sum(x * x for x in standard_embedding))

            if chunks_magnitude == 0 or standard_magnitude == 0:
                return 0.0
            return dot_product / (chunks_magnitude * standard_magnitude)
        except Exception as e:
            print(f"Error in CosineSimilarity calculation: {str(e)}")
            return 0.0

    def calculate_bleu_score(self, weights: tuple = (0.25, 0.25, 0.25, 0.25)) -> Dict[str, float]:
        try:
            standard_tokens = nltk.word_tokenize(self.standard.lower())
            chunks_tokens = [
                word
                for chunk in self.chunks
                for word in nltk.word_tokenize(chunk.lower())
            ]

            smoothing = SmoothingFunction().method1

            bleu_scores = {
                'bleu-1': sentence_bleu(
                    [standard_tokens],
                    chunks_tokens,
                    weights=(1, 0, 0, 0),
                    smoothing_function=smoothing
                ),
                'bleu-2': sentence_bleu(
                    [standard_tokens],
                    chunks_tokens,
                    weights=(0.5, 0.5, 0, 0),
                    smoothing_function=smoothing
                ),
                'bleu-3': sentence_bleu(
                    [standard_tokens],
                    chunks_tokens,
                    weights=(0.33, 0.33, 0.33, 0),
                    smoothing_function=smoothing
                ),
                'bleu-4': sentence_bleu(
                    [standard_tokens],
                    chunks_tokens,
                    weights=weights,
                    smoothing_function=smoothing
                )
            }

            return bleu_scores
        except Exception as e:
            print(f"Error in BLEU score calculation: {str(e)}")
            return {'bleu-1': 0.0, 'bleu-2': 0.0, 'bleu-3': 0.0, 'bleu-4': 0.0}

    def calculate_rouge_score(self) -> Dict[str, Dict[str, float]]:
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            combined_chunks = ' '.join(self.chunks)
            scores = scorer.score(self.standard, combined_chunks)

            return {
                'rouge-1': {
                    'precision': scores['rouge1'].precision,
                    'recall': scores['rouge1'].recall,
                    'f1': scores['rouge1'].fmeasure
                },
                'rouge-2': {
                    'precision': scores['rouge2'].precision,
                    'recall': scores['rouge2'].recall,
                    'f1': scores['rouge2'].fmeasure
                },
                'rouge-L': {
                    'precision': scores['rougeL'].precision,
                    'recall': scores['rougeL'].recall,
                    'f1': scores['rougeL'].fmeasure
                }
            }
        except Exception as e:
            print(f"Error in ROUGE score calculation: {str(e)}")
            return {}

    def evaluate_all_metrics(self) -> Dict[str, Union[float, Dict]]:
        try:
            results = {
                'cosine_similarity': self.CosineSimilarity(),
                'bleu_scores': self.calculate_bleu_score(),
                'rouge_scores': self.calculate_rouge_score()
            }

            print(f"Evaluation results for collection '{self.collection_name}':")
            print(f"Cosine Similarity: {results['cosine_similarity']}")
            print(f"BLEU Scores: {results['bleu_scores']}")
            print(f"ROUGE Scores: {results['rouge_scores']}\n")

            return results
        except Exception as e:
            print(f"Error in evaluation: {str(e)}")
            return {}


class QAAccuracyEvaluator:
    def __init__(self, questions: List[str], answers: List[str], llm_system,
                 retrieve_method: str = 'plain', labels: Optional[List[List[str]]] = None):
        if len(questions) != len(answers):
            raise ValueError("Number of questions must match number of answers")

        self.questions = questions
        self.reference_answers = answers
        self.generated_answers: List[Dict[str, str]] = []
        self.llm_system = llm_system
        self.retrieve_method = retrieve_method
        self.labels = labels

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )

    def generate_answers(self) -> List[Dict[str, str]]:
        self.generated_answers = []

        if self.labels is not None:
            for question, label in zip(self.questions, self.labels):
                try:
                    answer = self.llm_system.response(
                        question=question,
                        retrieval_method=self.retrieve_method,
                        labels=label
                    )
                    self.generated_answers.append(answer)
                except Exception as e:
                    print(f"Error generating answer for question '{question}': {str(e)}")
                    self.generated_answers.append({})
            return self.generated_answers

        for question in self.questions:
            try:
                answer = self.llm_system.response(
                    question=question,
                    retrieval_method=self.retrieve_method,
                    labels=None
                )
                self.generated_answers.append(answer)
            except Exception as e:
                print(f"Error generating answer for question '{question}': {str(e)}")
                self.generated_answers.append({})
        return self.generated_answers

    def calculate_rouge_scores(self, generated: str, reference: str) -> Dict[str, Dict[str, float]]:
        try:
            scores = self.rouge_scorer.score(reference, generated)
            return {
                'rouge-1': {
                    'precision': scores['rouge1'].precision,
                    'recall': scores['rouge1'].recall,
                    'f1': scores['rouge1'].fmeasure
                },
                'rouge-2': {
                    'precision': scores['rouge2'].precision,
                    'recall': scores['rouge2'].recall,
                    'f1': scores['rouge2'].fmeasure
                },
                'rouge-L': {
                    'precision': scores['rougeL'].precision,
                    'recall': scores['rougeL'].recall,
                    'f1': scores['rougeL'].fmeasure
                }
            }
        except Exception as e:
            print(f"Error calculating ROUGE scores: {str(e)}")
            return {
                'rouge-1': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                'rouge-2': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                'rouge-L': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            }

    @staticmethod
    def calculate_bleu_score(generated: str, reference: str) -> float:
        try:
            smoothing = SmoothingFunction().method1
            reference_tokens = nltk.word_tokenize(reference.lower())
            generated_tokens = nltk.word_tokenize(generated.lower())
            return sentence_bleu(
                [reference_tokens],
                generated_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothing
            )
        except Exception as e:
            print(f"Error calculating BLEU score: {str(e)}")
            return 0.0

    @staticmethod
    def calculate_exact_match(generated: str, reference: str) -> float:
        try:
            return float(generated.strip().lower() == reference.strip().lower())
        except Exception as e:
            print(f"Error calculating exact match: {str(e)}")
            return 0.0

    def evaluate(self) -> Dict[str, Union[float, Dict]]:
        if not self.generated_answers:
            self.generate_answers()

        total_rouge_scores = {
            'rouge-1': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
            'rouge-2': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
            'rouge-L': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        }
        total_bleu = 0.0
        total_exact_match = 0.0
        per_question_metrics = []
        valid_answers = 0

        for question, generated, reference in zip(
            self.questions, self.generated_answers, self.reference_answers
        ):
            if not generated or 'answer' not in generated:
                continue

            valid_answers += 1
            generated_answer = generated['answer']

            rouge_scores = self.calculate_rouge_scores(generated_answer, reference)
            bleu_score = self.calculate_bleu_score(generated_answer, reference)
            exact_match = self.calculate_exact_match(generated_answer, reference)

            for rouge_type in total_rouge_scores:
                for metric in total_rouge_scores[rouge_type]:
                    total_rouge_scores[rouge_type][metric] += rouge_scores[rouge_type][metric]

            total_bleu += bleu_score
            total_exact_match += exact_match

            per_question_metrics.append({
                'question': question,
                'generated_answer': generated_answer,
                'reference_answer': reference,
                'rouge_scores': rouge_scores,
                'bleu_score': bleu_score,
                'exact_match': exact_match
            })

        if valid_answers == 0:
            return {
                'error': 'No valid question-answer pairs to evaluate',
                'average_exact_match': 0.0,
                'average_bleu_score': 0.0,
                'average_rouge_scores': total_rouge_scores,
                'per_question_metrics': []
            }

        avg_rouge_scores = {
            rouge_type: {
                metric: score / valid_answers
                for metric, score in metrics.items()
            }
            for rouge_type, metrics in total_rouge_scores.items()
        }

        return {
            'average_exact_match': total_exact_match / valid_answers,
            'average_bleu_score': total_bleu / valid_answers,
            'average_rouge_scores': avg_rouge_scores,
            'per_question_metrics': per_question_metrics,
            'questions': self.questions,
            'generated_answers': self.generated_answers,
            'reference_answers': self.reference_answers
        }

    @staticmethod
    def print_evaluation_results(results: Dict[str, Union[float, Dict]]) -> None:
        if 'error' in results:
            print(f"\nError: {results['error']}")
            return

        print("\nQA System Evaluation Results:")
        print("=============================")
        print("\nAverage Metrics:")
        print("--------------")
        print(f"Exact Match Score: {results['average_exact_match']:.4f}")
        print(f"BLEU Score: {results['average_bleu_score']:.4f}")

        print("\nROUGE Scores:")
        for rouge_type, scores in results['average_rouge_scores'].items():
            print(f"\n{rouge_type.upper()}:")
            for metric, value in scores.items():
                print(f"  {metric}: {value:.4f}")

        print("\nPer-Question Results:")
        print("-------------------")
        for idx, metrics in enumerate(results['per_question_metrics']):
            print(f"\nQuestion {idx + 1}: {metrics['question']}")
            print(f"Generated Answer: {metrics['generated_answer']}")
            print(f"Reference Answer: {metrics['reference_answer']}")
            print(f"Exact Match: {metrics['exact_match']}")
            print(f"BLEU Score: {metrics['bleu_score']:.4f}")
            print("ROUGE Scores:")
            for rouge_type, scores in metrics['rouge_scores'].items():
                print(f"  {rouge_type}: F1={scores['f1']:.4f}, P={scores['precision']:.4f}, R={scores['recall']:.4f}")


def evaluate_and_average_metrics(texts, collection_name, qa_path,
                                 agentic: bool = False, label_path: Optional[str] = None,
                                 output_dir: str = 'output'):
    all_results: Dict[str, Any] = {}
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    model = "gpt-4-turbo-preview"

    if label_path is None:
        for name, text in texts.items():
            questions, answers = extract_qa_by_document(qa_path, name)
            if not questions:
                continue
            retrieve = Retrieve(collection_name=collection_name, n_chunks=3, meta=[name])
            rag_llm = RAG_llm(client=client, model=model, retrieve=retrieve)
            retrieve_method = 'agentic' if agentic else 'plain'
            evaluator = QAAccuracyEvaluator(
                questions=questions,
                answers=answers,
                llm_system=rag_llm,
                retrieve_method=retrieve_method
            )
            results = evaluator.evaluate()
            all_results[name] = results
            print(f"\n----- Results for {name} -----")
            QAAccuracyEvaluator.print_evaluation_results(results)

    if label_path:
        for name, text in texts.items():
            retrieve = Retrieve(collection_name=collection_name, n_chunks=3, meta=[name])
            rag_llm = RAG_llm(client=client, model=model, retrieve=retrieve)

            questions, answers, labels = [], [], []
            for file_name, question, answer, label in read_csv_by_tuple(label_path):
                if f"{file_name}.pdf" == name:
                    questions.append(question)
                    answers.append(answer)
                    labels.append(label)

            if not questions:
                continue

            evaluator = QAAccuracyEvaluator(
                questions=questions,
                answers=answers,
                llm_system=rag_llm,
                retrieve_method='agentic',
                labels=labels
            )
            results = evaluator.evaluate()
            all_results[name] = results
            print(f"\n----- Results for {name} -----")
            QAAccuracyEvaluator.print_evaluation_results(results)

    avg_results = calculate_and_print_averages(all_results, collection_name)

    os.makedirs(output_dir, exist_ok=True)
    suffix = 'with_label' if label_path else 'without_label'
    outfile = f'{output_dir}/all_evaluation_results_{collection_name}_{suffix}.json'
    with open(outfile, 'w') as f:
        import json
        json.dump(
            {
                'individual_document_results': all_results,
                'average_metrics': avg_results
            },
            f,
            indent=2
        )

    print(f"Complete evaluation results have been saved to {outfile}")
    return all_results


def calculate_and_print_averages(all_results, collection_name):
    metrics = {
        'exact_match': [],
        'bleu_score': [],
        'rouge-1': {'precision': [], 'recall': [], 'f1': []},
        'rouge-2': {'precision': [], 'recall': [], 'f1': []},
        'rouge-L': {'precision': [], 'recall': [], 'f1': []}
    }

    for doc_name, results in all_results.items():
        per_question = results.get('per_question_metrics', [])
        metrics['exact_match'].extend([q['exact_match'] for q in per_question])
        metrics['bleu_score'].extend([q['bleu_score'] for q in per_question])

        for q in per_question:
            for rouge_type in ['rouge-1', 'rouge-2', 'rouge-L']:
                for metric in ['precision', 'recall', 'f1']:
                    metrics[rouge_type][metric].append(q['rouge_scores'][rouge_type][metric])

    avg_results = {
        'collection': collection_name,
        'average_exact_match': float(np.mean(metrics['exact_match'])) if metrics['exact_match'] else 0.0,
        'average_bleu_score': float(np.mean(metrics['bleu_score'])) if metrics['bleu_score'] else 0.0,
        'average_rouge_scores': {
            'rouge-1': {
                'precision': float(np.mean(metrics['rouge-1']['precision'])) if metrics['rouge-1']['precision'] else 0.0,
                'recall': float(np.mean(metrics['rouge-1']['recall'])) if metrics['rouge-1']['recall'] else 0.0,
                'f1': float(np.mean(metrics['rouge-1']['f1'])) if metrics['rouge-1']['f1'] else 0.0
            },
            'rouge-2': {
                'precision': float(np.mean(metrics['rouge-2']['precision'])) if metrics['rouge-2']['precision'] else 0.0,
                'recall': float(np.mean(metrics['rouge-2']['recall'])) if metrics['rouge-2']['recall'] else 0.0,
                'f1': float(np.mean(metrics['rouge-2']['f1'])) if metrics['rouge-2']['f1'] else 0.0
            },
            'rouge-L': {
                'precision': float(np.mean(metrics['rouge-L']['precision'])) if metrics['rouge-L']['precision'] else 0.0,
                'recall': float(np.mean(metrics['rouge-L']['recall'])) if metrics['rouge-L']['recall'] else 0.0,
                'f1': float(np.mean(metrics['rouge-L']['f1'])) if metrics['rouge-L']['f1'] else 0.0
            }
        }
    }

    print("\n===== AVERAGE METRICS ACROSS ALL DOCUMENTS =====")
    print(f"Average Exact Match: {avg_results['average_exact_match']:.4f}")
    print(f"Average BLEU Score: {avg_results['average_bleu_score']:.4f}")
    print("\nAverage ROUGE Scores:")
    for rouge_type in ['rouge-1', 'rouge-2', 'rouge-L']:
        print(f"  {rouge_type}:")
        for metric in ['precision', 'recall', 'f1']:
            value = avg_results['average_rouge_scores'][rouge_type][metric]
            print(f"    {metric}: {value:.4f}")

    summary_file = f'average_metrics_summary_{collection_name}.json'
    with open(summary_file, 'w') as f:
        import json
        json.dump(avg_results, f, indent=2)

    print(f"\nAverage metrics have been saved to '{summary_file}'")
    return avg_results