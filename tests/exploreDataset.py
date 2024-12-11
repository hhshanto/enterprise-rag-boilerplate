from datasets import load_from_disk
import os
import json
from typing import Dict, Any
import pyarrow as pa

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

def print_dataset_analysis(dataset_path: str, dataset_name: str):
    """Print detailed analysis of the dataset."""
    print(f"\n=== Analyzing {dataset_name} ===")
    
    structure = explore_dataset_structure(dataset_path)
    
    print("\n1. Basic Information:")
    print(f"Number of rows: {structure['num_rows']}")
    print(f"Size in memory: {structure['size_in_bytes'] / 1024 / 1024:.2f} MB")
    
    print("\n2. Features (Column Types):")
    for feature_name, feature_type in structure['features'].items():
        print(f"{feature_name}: {feature_type}")
    
    print("\n3. Dataset Info:")
    print(f"Description: {structure['dataset_info'].get('description', 'N/A')}")
    print(f"Citation: {structure['dataset_info'].get('citation', 'N/A')[:200]}...")
    
    print("\n4. Arrow Schema:")
    print(structure['arrow_schema'])
    
    # Load dataset to show sample
    dataset = load_from_disk(dataset_path)
    print("\n5. Sample Record:")
    print(json.dumps(dataset[0], indent=2, default=str))

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ragData'))
    
    # Analyze QA dataset
    qa_path = os.path.join(base_dir, 'question_answer')
    print_dataset_analysis(qa_path, "Question-Answer Dataset (SQuAD)")
    
    # Analyze text corpus dataset
    text_path = os.path.join(base_dir, 'text_corpus')
    print_dataset_analysis(text_path, "Text Corpus Dataset (Shakespeare)")

if __name__ == "__main__":
    main()