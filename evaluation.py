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