from datasets import load_from_disk
import os
import json
from typing import Dict, Any
import pyarrow as pa
import sys
from collections import Counter

def explore_dataset_structure(dataset_path: str) -> Dict[str, Any]:
    """Explore the internal structure of a dataset."""
    
    # Load dataset info
    with open(os.path.join(dataset_path, 'dataset_info.json'), 'r') as f:
        dataset_info = json.load(f)
        
    # Load the actual dataset
    dataset = load_from_disk(dataset_path)
    
    # Get Arrow table
    arrow_table = dataset.data.table
    
    return {
        "dataset_info": dataset_info,
        "features": dataset.features,
        "num_rows": len(dataset),
        "column_names": dataset.column_names,
        "size_in_bytes": dataset.data.nbytes,
        "arrow_schema": arrow_table.schema
    }

def print_dataset_analysis(dataset_path: str, dataset_name: str, output_file):
    """Print detailed analysis of the dataset."""
    print(f"\n=== Analyzing {dataset_name} ===", file=output_file)
    
    structure = explore_dataset_structure(dataset_path)
    
    print("\n1. Basic Information:", file=output_file)
    print(f"Number of rows: {structure['num_rows']}", file=output_file)
    print(f"Size in memory: {structure['size_in_bytes'] / 1024 / 1024:.2f} MB", file=output_file)
    
    print("\n2. Features (Column Types):", file=output_file)
    for feature_name, feature_type in structure['features'].items():
        print(f"{feature_name}: {feature_type}", file=output_file)
    
    print("\n3. Dataset Info:", file=output_file)
    print(f"Description: {structure['dataset_info'].get('description', 'N/A')}", file=output_file)
    print(f"Citation: {structure['dataset_info'].get('citation', 'N/A')[:200]}...", file=output_file)
    
    print("\n4. Arrow Schema:", file=output_file)
    print(structure['arrow_schema'], file=output_file)
    
    # Load dataset to show sample
    dataset = load_from_disk(dataset_path)
    print("\n5. Sample Record:", file=output_file)
    sample = dataset[0]
    if 'text' in sample and isinstance(sample['text'], str) and len(sample['text']) > 500:
        sample['text'] = sample['text'][:500] + "... [truncated]"
    print(json.dumps(sample, indent=2, default=str), file=output_file)

    # Add word count for text corpus
    if 'text' in sample:
        word_count = len(sample['text'].split())
        print(f"\nTotal word count in sample: {word_count}", file=output_file)

def analyze_text_corpus(dataset, output_file):
    """Provide additional analysis for text corpus dataset."""
    print("\n6. Text Corpus Analysis:", file=output_file)
    
    # Get the full text
    full_text = dataset[0]['text']
    
    # Word count
    words = full_text.split()
    word_count = len(words)
    print(f"Total word count: {word_count}", file=output_file)
    
    # Character count
    char_count = len(full_text)
    print(f"Total character count: {char_count}", file=output_file)
    
    # Line count
    line_count = full_text.count('\n') + 1
    print(f"Total line count: {line_count}", file=output_file)
    
    # Unique words
    unique_words = len(set(words))
    print(f"Unique word count: {unique_words}", file=output_file)
    
    # Most common words (top 10)
    word_freq = Counter(words)
    print("\nTop 10 most common words:", file=output_file)
    for word, count in word_freq.most_common(10):
        print(f"{word}: {count}", file=output_file)

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ragData'))
    doc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs'))
    
    # Create doc directory if it doesn't exist
    os.makedirs(doc_dir, exist_ok=True)
    
    output_file_path = os.path.join(doc_dir, 'dataset_analysis.txt')
    
    with open(output_file_path, 'w') as output_file:
        # Redirect stdout to the file
        original_stdout = sys.stdout
        sys.stdout = output_file
        
        try:
            # Analyze QA dataset
            qa_path = os.path.join(base_dir, 'question_answer')
            print_dataset_analysis(qa_path, "Question-Answer Dataset (SQuAD)", output_file)
            
            # Analyze text corpus dataset
            text_path = os.path.join(base_dir, 'text_corpus')
            print_dataset_analysis(text_path, "Text Corpus Dataset (Shakespeare)", output_file)
            dataset = load_from_disk(text_path)
            analyze_text_corpus(dataset, output_file)
        
        finally:
            # Restore stdout
            sys.stdout = original_stdout
    
    print(f"Dataset analysis has been written to: {output_file_path}")

if __name__ == "__main__":
    main()