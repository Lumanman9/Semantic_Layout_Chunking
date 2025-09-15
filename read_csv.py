import csv


def read_csv_by_tuple(file_path):
    with open(file_path, 'r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)

        # Skip header row if your CSV has one
        next(csv_reader, None)  # Remove this line if you don't have a header

        # Process each row
        for row in csv_reader:
            if len(row) >= 4:
                file_name, question, answer, label = row[0], row[1], row[2], row[3]

                # Process labels into a list
                label_list = [l.strip() for l in label.split(',')]

                # You can return the processed data or work with it directly here
                yield (file_name, question, answer, label_list)
            else:
                print(f"Warning: Skipping row with insufficient columns: {row}")


# Example usage
if __name__ == "__main__":
    csv_path = "data/QA_with_label.csv"

    # Iterate through rows with tuple unpacking
    for file_name, question, answer, labels in read_csv_by_tuple(csv_path):
        print(f"File: {file_name}")
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Labels: {labels}")
        print("---")