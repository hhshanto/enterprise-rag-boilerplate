import os
from datasets import load_dataset

def download_datasets():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ragData'))
    os.makedirs(data_dir, exist_ok=True)
    
    # Load the datasets
    question_answer_dataset = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
    text_corpus_dataset = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
    
    # Save the datasets directly to 'ragData' folder
    question_answer_dataset.save_to_disk(os.path.join(data_dir, 'question_answer'))
    text_corpus_dataset.save_to_disk(os.path.join(data_dir, 'text_corpus'))

if __name__ == "__main__":
    download_datasets()