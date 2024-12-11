from datasets import load_from_disk
import os

def read_datasets():
    # Path to your data
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ragData'))
    qa_path = os.path.join(base_dir, 'question_answer')
    text_path = os.path.join(base_dir, 'text_corpus')

    # Load datasets
    qa_dataset = load_from_disk(qa_path)
    text_dataset = load_from_disk(text_path)

    # Inspect question-answer dataset (SQuAD)
    print("\n=== Question-Answer Dataset (SQuAD) ===")
    print(f"Number of examples: {len(qa_dataset)}")
    print("\nFirst QA example:")
    first_qa = qa_dataset[0]
    print(f"Question: {first_qa['question']}")
    print(f"Context: {first_qa['context'][:200]}...")
    print(f"Answer: {first_qa['answers']}")

    # Inspect text corpus dataset (tiny_shakespeare)
    print("\n=== Text Corpus Dataset (Shakespeare) ===")
    print(f"Number of examples: {len(text_dataset)}")
    print("\nFirst text example:")
    first_text = text_dataset[0]
    print(f"Text: {first_text['text'][:200]}...")

    # Show available fields
    print("\nQA Dataset fields:", qa_dataset.column_names)
    print("Text Dataset fields:", text_dataset.column_names)

    # Show a few more QA examples
    print("\n=== More QA Examples ===")
    for i in range(3):  # Show first 3 examples
        qa_pair = qa_dataset[i]
        print(f"\nExample {i+1}:")
        print(f"Question: {qa_pair['question']}")
        print(f"Answer: {qa_pair['answers']['text'][0]}")
        print(f"Context preview: {qa_pair['context'][:100]}...")

if __name__ == "__main__":
    read_datasets()