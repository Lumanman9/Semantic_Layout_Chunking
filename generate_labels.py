"""
Script to generate section labels for questions using LLM.
Uses the same logic as agentic_query in retreival.py when labels=None.
"""
import csv
import os
import argparse
from dotenv import load_dotenv
import openai
from loguru import logger
from typing import List

load_dotenv()

# Schema definitions (same as in retreival.py)
SCHEMAS = {
    'Literature_Paper': ['author', 'abstract', 'introduction', 'methodology', 'related_work', 
                        'experiment', 'background', 'dataset', 'acknowledge', 'conclusion', 
                        'result_discussion'],
    'Financial_Report': ['company_overview', 'manager_discussion', 'performance_summary', 
                         'strategy_outlook', 'segment_business_review', 'risk_factors',
                         'governance_and_leadership', 'sustainability', 'financial_statement', 
                         'financial_statement_note', 'auditors_report', 'shareholder_information',
                         'legal_compliance', 'miscellaneous'],
    'Wikiperson': ['introduction', 'early_fife', 'career', 'contribution', 'award', 
                   'legal_trouble', 'personal_experience', 'miscellaneous']
}


def parse_sections_from_response(response_text: str) -> List[str]:
    """
    Parse section names from LLM response.
    Same logic as in retreival.py
    
    Args:
        response_text (str): Raw response from LLM
    
    Returns:
        list: List of section names
    """
    cleaned_text = response_text.replace('- ', '').replace('* ', '').replace('1. ', '')
    sections = [s.strip() for s in cleaned_text.replace('\n', ',').split(',') if s.strip()]
    return sections


def generate_label_for_question(question: str, schema: List[str], dataset_type: str, llm_client, model: str, 
                                temperature: float = 0.7, max_tokens: int = 500) -> List[str]:
    """
    Generate section labels for a question using LLM.
    Uses an improved prompt for better section identification.
    
    Args:
        question (str): The question to generate labels for
        schema (List[str]): List of available section names
        llm_client: OpenAI client instance
        model (str): Model name to use
        temperature (float): Temperature parameter
        max_tokens (int): Max tokens for response
    
    Returns:
        List[str]: List of section names that are most relevant
    """
    schema_str = ', '.join(schema)
    prompt = f'''You are analyzing a {dataset_type} document to identify which sections are most likely to contain the answer to the question.

Question: "{question}"

Available sections in the {dataset_type} document: {schema_str}

Task: Identify 1-3 sections (prioritize the most relevant ones) where the answer to this question is most likely to be found. Consider:
- Where would this type of information typically appear in a {dataset_type} document?
- Which sections would contain the specific details needed to answer this question?
- If multiple sections might be relevant, select the most important ones (up to 3).

Instructions:
- Return ONLY a comma-separated list of section names from the available sections
- Use the exact section names as listed above
- Do not include any explanations, prefixes, or additional text
- If unsure, select the most likely section(s)

Example format: ["introduction", "methodology", "conclusion"]'''
    
    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        relevant_sections = parse_sections_from_response(response.choices[0].message.content)
        return relevant_sections
    
    except Exception as e:
        logger.error(f"Error generating label for question '{question}': {str(e)}")
        return []


def process_csv(input_csv: str, output_csv: str, dataset_type: str = 'Literature_Paper', 
                model: str = "gpt-4-turbo-preview", temperature: float = 0.7, 
                max_tokens: int = 500, has_labels: bool = False):
    """
    Process a CSV file and generate labels for questions.
    
    Args:
        input_csv (str): Path to input CSV file
        output_csv (str): Path to output CSV file with labels
        dataset_type (str): Type of dataset (determines schema)
        model (str): LLM model to use
        temperature (float): Temperature for LLM
        max_tokens (int): Max tokens for LLM
        has_labels (bool): Whether input CSV already has labels (will regenerate them)
    """
    if dataset_type not in SCHEMAS:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {list(SCHEMAS.keys())}")
    
    schema = SCHEMAS[dataset_type]
    llm_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Read input CSV
    rows = []
    has_header = False
    with open(input_csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)

        # Peek first row and decide if it's a header or data
        first_row = next(reader, None)
        if first_row:
            lower_first = [c.strip().lower() for c in first_row]
            # Heuristic: treat as header if it looks like one
            if any(k in lower_first for k in ("file_name", "filename", "doc_name", "question")):
                header = first_row
                has_header = True

                # Determine CSV format from header
                if len(header) >= 4 and "label" in [h.lower() for h in header]:
                    # CSV with labels (qa_with_label format)
                    has_labels = True
                    logger.info("Detected CSV with existing labels - will regenerate them")
                elif len(header) >= 3:
                    # CSV without labels (qa format)
                    has_labels = False
                    logger.info("Detected CSV without labels - will generate new labels")
            else:
                # No real header -> treat first_row as data
                if len(first_row) >= 3:
                    rows.append(first_row)

        # Read remaining rows
        for row in reader:
            if len(row) >= 3:
                rows.append(row)
    
    logger.info(f"Processing {len(rows)} questions...")
    
    # Process each row and generate labels
    results = []
    for i, row in enumerate(rows):
        file_name = row[0]
        question = row[1]
        answer = row[2] if len(row) > 2 else ""
        
        logger.info(f"Processing question {i+1}/{len(rows)}: {question[:50]}...")
        
        # Generate labels using LLM
        labels = generate_label_for_question(
            question=question,
            schema=schema,
            dataset_type=dataset_type,
            llm_client=llm_client,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Format labels as comma-separated string
        labels_str = ','.join(labels) if labels else ""
        
        logger.info(f"Generated labels: {labels_str}")
        
        # Create output row
        if has_labels and len(row) >= 4:
            # Replace existing label
            output_row = [file_name, question, answer, labels_str]
        else:
            # Add new label column
            output_row = [file_name, question, answer, labels_str]
        
        results.append(output_row)
    
    # Write output CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header (normalized)
        writer.writerow(['file_name', 'question', 'answer', 'label'])
        # Write data rows
        writer.writerows(results)
    
    logger.info(f"Successfully generated labels and saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description='Generate section labels for questions using LLM')
    parser.add_argument('--input_csv', type=str, required=True,
                       help='Path to input CSV file (with or without labels)')
    parser.add_argument('--output_csv', type=str, required=True,
                       help='Path to output CSV file with generated labels')
    parser.add_argument('--dataset_type', type=str, default='Literature_Paper',
                       choices=['Literature_Paper', 'Financial_Report', 'Wikiperson'],
                       help='Type of dataset (determines schema)')
    parser.add_argument('--model', type=str, default='gpt-4-turbo-preview',
                       help='LLM model to use')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for LLM')
    parser.add_argument('--max_tokens', type=int, default=500,
                       help='Max tokens for LLM response')
    
    args = parser.parse_args()
    
    process_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        dataset_type=args.dataset_type,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )


if __name__ == '__main__':
    main()

