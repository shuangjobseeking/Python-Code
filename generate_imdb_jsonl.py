import os
import random
import json

LABEL_POS = "positive"
LABEL_NEG = "negative"

def load_reviews_from_folder(folder_path, sample_size=None):
    # List all .txt files in the folder
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    # Sample files if sample_size is specified, otherwise use all files
    if sample_size is not None:
        sample_size = min(sample_size, len(all_files))
        sampled_files = random.sample(all_files, sample_size)
    else:
        sampled_files = all_files  # Use all files
    reviews = []
    # Read and clean text from each sampled file
    for filename in sampled_files:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip().replace('\n', ' ')
            reviews.append(text)
    return reviews

def create_jsonl_chat_format(pos_folder, neg_folder, output_path, samples_per_class=None):
    # Load positive and negative reviews
    pos_reviews = load_reviews_from_folder(pos_folder, samples_per_class)
    neg_reviews = load_reviews_from_folder(neg_folder, samples_per_class)

    # Write the output JSONL file in chat message format
    with open(output_path, 'w', encoding='utf-8') as fout:
        for review in pos_reviews:
            item = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Review: {review}\nWhat is the sentiment of this review? (positive/negative)"
                    },
                    {
                        "role": "assistant",
                        "content": LABEL_POS
                    }
                ]
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

        for review in neg_reviews:
            item = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Review: {review}\nWhat is the sentiment of this review? (positive/negative)"
                    },
                    {
                        "role": "assistant",
                        "content": LABEL_NEG
                    }
                ]
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    pos_folder = r'.\datasets\aclImdb_v1\train\pos'
    neg_folder = r'.\datasets\aclImdb_v1\train\neg'
    output_file = r'.\data\imdb_finetune_chat.jsonl'

    # Set None to use all available files, or specify an integer for sample size per class
    samples_per_class = 800  

    create_jsonl_chat_format(pos_folder, neg_folder, output_file, samples_per_class)
    count = samples_per_class if samples_per_class is not None else 12500
    print(f"Created {output_file} with {count*2} samples in chat format.")
