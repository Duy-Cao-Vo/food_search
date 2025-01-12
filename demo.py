import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


# Average pooling function
def average_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# Function to get embeddings using saved tokenizer and model
def get_embeddings2(texts, model_dir):  # `texts` is a list of sentences
    # Load the tokenizer and model from the local directory
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)

    # Tokenize the input texts
    batch_dict = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**batch_dict)

    # Average pooling
    last_hidden_states = outputs.last_hidden_state
    attention_mask = batch_dict['attention_mask']
    embeddings = average_pool(last_hidden_states, attention_mask)

    # Normalize embeddings
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.numpy()

# Function to precompute and save embeddings for a list of food names
def precompute_embeddings(food_list, model_dir, save_path):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize and compute embeddings in batches
    batch_size = 64  # Adjust batch size for memory usage
    all_embeddings = []
    for i in range(0, len(food_list), batch_size):
        batch_texts = food_list[i:i + batch_size]
        batch_dict = tokenizer(batch_texts, max_length=64, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
        
        with torch.no_grad():
            outputs = model(**batch_dict)
        
        # Average pooling
        last_hidden_states = outputs.last_hidden_state
        attention_mask = batch_dict['attention_mask']
        embeddings = average_pool(last_hidden_states, attention_mask)
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu().numpy())
    
    # Concatenate all embeddings and save to file
    all_embeddings = np.vstack(all_embeddings)
    np.save(save_path, all_embeddings)  # Save embeddings as .npy file
    return all_embeddings

df = pd.read_csv("./dataset.txt", sep='|', on_bad_lines='skip', header=None)
df = df[[0,2,4]]
df.columns = ['description','name','labels']
df['labels'] = df['labels'].astype(int)

# Example: Precompute and save embeddings
model_dir = './phobert-finetuned-vietnamese'
food_list = df['name'].unique().tolist()  # Extract unique food names
save_path = './food_embeddings.npy'
precompute_embeddings(food_list, model_dir, save_path)


from sklearn.metrics.pairwise import cosine_similarity


# Function to find similar food names using precomputed embeddings
def get_similar_vietnamese_food_fast(food_name, food_list, embeddings_path, model_dir, threshold=0.6, limit=10):
    # Load precomputed embeddings
    food_embeddings = np.load(embeddings_path)
    
    # Compute embedding for the input food name
    input_embedding = get_embeddings2([food_name], model_dir)
    
    # Compute cosine similarity
    similarities = cosine_similarity(input_embedding, food_embeddings)[0]
    
    # Pair similarities with corresponding food names
    indexed_scores = [(i, score) for i, score in enumerate(similarities) if score > threshold]
    
    # Sort scores in descending order
    sorted_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top `limit` most similar items
    top_results = sorted_scores[:limit]
    
    # Retrieve the corresponding food names
    results = [(food_list[i], score) for i, score in top_results]
    
    return results

# Test function
def test_local_model_fast():
    # Define the saved model directory and embeddings path
    model_dir = './phobert-finetuned-vietnamese'
    embeddings_path = './food_embeddings.npy'
    
    # Example food descriptions
    food_names = ["nước ép cam tươi nguyên chất",
                  "nuoc ep cam nguyen chat",
                  "nuoc ep",
                  "nước cam",
                  "nuoc cam",
                  "cơm đùi gà nướng",
                  "cơm dui ga",
                  "tra sữa",
                  "trà sữa phúc long"]

    # Similarity threshold
    threshold = 0.6

    for food_name in food_names:
        # Get similar food names
        similar_foods = get_similar_vietnamese_food_fast(food_name, df['name'].unique().tolist(), embeddings_path, model_dir, threshold, limit=10)

        # Print results
        print(f"Input Food Description: {food_name}")
        print("Top Similar Foods:")
        for food, score in similar_foods:
            print(f"+ {food} (Similarity: {score:.4f})")
        print()
test_local_model_fast()