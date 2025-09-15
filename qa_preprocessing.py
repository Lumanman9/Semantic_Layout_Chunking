import pandas as pd


def extract_qa_by_document(csv_filename, document_name):
    """
    Extract questions and answers from a CSV file for a specific document name.
    Concatenates all answers for each question into a single string.

    Parameters:
    csv_filename (str): Path to the CSV file
    document_name (str): Name of the document to filter by (with or without extension)

    Returns:
    tuple: (questions_list, answers_list) where:
           - questions_list is a list of questions for the specified document
           - answers_list is a list of concatenated answers (one per question)
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_filename, sep='|', dtype=str)

        # Remove extension if present (e.g., .pdf) while preserving document identifiers that contain periods
        base_name = document_name
        if document_name.lower().endswith(('.pdf', '.txt', '.docx', '.doc')):
            # Find the last occurrence of common file extensions and remove it
            for ext in ['.pdf', '.txt', '.docx', '.doc']:
                if document_name.lower().endswith(ext):
                    base_name = document_name[:-len(ext)]
                    break

        # Filter rows by document name (checking against base name)
        filtered_df = df[df['doc_name'] == base_name]

        if filtered_df.empty:
            print(f"No entries found for document: {document_name}")
            return [], []

        # Extract questions as a list
        questions_list = filtered_df['question'].tolist()

        # Get answer columns (those starting with 'answer_')
        answer_columns = [col for col in df.columns if col.startswith('answer')]

        # Create a list to hold one concatenated answer per question
        answers_list = []

        # For each question, concatenate all its answers
        for _, row in filtered_df.iterrows():
            # Collect all non-empty answers for this question
            row_answers = []
            for col in answer_columns:
                if pd.notna(row[col]) and row[col].strip() != '':
                    row_answers.append(row[col])

            # Concatenate all answers for this question into a single string
            # Using space as separator to avoid any formatting issues
            concatenated_answer = " ".join(row_answers)
            answers_list.append(concatenated_answer)

        return questions_list, answers_list

    except Exception as e:
        print(f"Error processing the CSV file: {e}")
        return [], []


# Example usage
if __name__ == "__main__":
    # Replace with your actual CSV filename
    filename = 'Wikiperson_data/feta_qa.csv'

    doc_name_with_extension = 'Gary Holton.pdf'

    # Test with extension
    print("\nTesting with document name with extension:")
    questions, answers = extract_qa_by_document(filename, doc_name_with_extension)
    print(f"Questions for document {doc_name_with_extension}:")
    for i, question in enumerate(questions):
        print(f"{i + 1}. {question}")
        print(f"   Answers: {answers[i]}")