from datasets import load_from_disk
import os
import json
from typing import List, Dict
import pandas as pd

class QAExtractor:
    def __init__(self):
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ragData'))
        self.qa_path = os.path.join(self.base_dir, 'question_answer')
        
    def extract_qa_pairs(self) -> List[Dict]:
        """Extract question-answer pairs from the dataset"""
        dataset = load_from_disk(self.qa_path)
        
        qa_pairs = []
        # Directly iterate over the dataset (no 'train' split needed)
        for i in range(len(dataset)):
            example = dataset[i]
            qa_pairs.append({
                'question': example['question'],
                'answer': example['answers']['text'][0],  # Taking first answer
                'context': example['context'],  # Keep context for verification
                'title': example['title']  # Adding title field
            })
        return qa_pairs
    
    def save_qa_pairs(self, format: str = 'json'):
        """Save QA pairs in specified format"""
        qa_pairs = self.extract_qa_pairs()
        
        # Create evaluation directory if it doesn't exist
        eval_dir = os.path.join(self.base_dir, 'evaluation')
        os.makedirs(eval_dir, exist_ok=True)
        
        if format == 'json':
            # Save as JSON
            output_path = os.path.join(eval_dir, 'qa_pairs.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(qa_pairs)} QA pairs to {output_path}")
            
        elif format == 'csv':
            # Save as CSV
            output_path = os.path.join(eval_dir, 'qa_pairs.csv')
            df = pd.DataFrame(qa_pairs)
            df.to_csv(output_path, index=False)
            print(f"Saved {len(qa_pairs)} QA pairs to {output_path}")
            
        # Save questions only (for testing)
        questions_path = os.path.join(eval_dir, 'questions.txt')
        with open(questions_path, 'w', encoding='utf-8') as f:
            for pair in qa_pairs:
                f.write(pair['question'] + '\n')
        print(f"Saved {len(qa_pairs)} questions to {questions_path}")
        
        return qa_pairs
    
    def print_sample_pairs(self, n: int = 5):
        """Print sample QA pairs"""
        qa_pairs = self.extract_qa_pairs()
        print(f"\n=== Sample of {n} QA Pairs ===")
        for i, pair in enumerate(qa_pairs[:n], 1):
            print(f"\nPair {i}:")
            print(f"Title: {pair['title']}")
            print(f"Q: {pair['question']}")
            print(f"A: {pair['answer']}")
            print(f"Context: {pair['context'][:200]}...")

def main():
    extractor = QAExtractor()
    
    # Print dataset structure first
    dataset = load_from_disk(extractor.qa_path)
    print("Dataset structure:")
    print(f"Number of examples: {len(dataset)}")
    print(f"Available fields: {dataset.column_names}")
    print("\nFirst example structure:")
    print(dataset[0])
    
    # Save in both formats
    extractor.save_qa_pairs(format='json')
    extractor.save_qa_pairs(format='csv')
    
    # Print sample pairs
    extractor.print_sample_pairs(5)
    
    print("\nYou can find the extracted QA pairs in the 'data/ragData/evaluation' directory")
    print("Files created:")
    print("1. qa_pairs.json - Complete QA pairs with context")
    print("2. qa_pairs.csv  - Same data in CSV format")
    print("3. questions.txt - Just the questions (for testing)")

if __name__ == "__main__":
    main()