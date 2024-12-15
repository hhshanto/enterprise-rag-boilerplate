import os
import shutil

def delete_embedded_data():
    # Define the path to the embedding directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    embedding_dir = os.path.join(root_dir, 'data', 'embedding')

    # Check if the directory exists
    if os.path.exists(embedding_dir):
        try:
            # Remove the entire directory and its contents
            shutil.rmtree(embedding_dir)
            print(f"Successfully deleted embedded data from: {embedding_dir}")
        except Exception as e:
            print(f"Error deleting embedded data: {str(e)}")
    else:
        print(f"Embedding directory not found: {embedding_dir}")

if __name__ == "__main__":
    # Prompt for confirmation
    confirm = input("Are you sure you want to delete all embedded data? (yes/no): ")
    if confirm.lower() == 'yes':
        delete_embedded_data()
    else:
        print("Operation cancelled.")