import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re

# Step 1: Extract content from a product URL

def extract_text_from_url(url, div_class):
    response = requests.get(url)
    if response.status_code != 200:
        return "Error fetching page"

    soup = BeautifulSoup(response.text, "html.parser")
    content = soup.find("div", {"class": div_class})
    
    if content:
        return content.get_text(strip=True)
    return "No content found"

# Step 2: Generate BERT embeddings for extracted text

def get_bert_embedding(text):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**tokens)
    
    return output.last_hidden_state.mean(dim=1)  # Sentence embedding

# Step 3: Find relevant keywords using extracted text and product name
def find_relevant_keywords(product_name, product_text, general_keywords):
    combined_text = product_name + " " + product_text  # Merge product name and description
    product_embedding = get_bert_embedding(combined_text)
    
    keyword_embeddings = {word: get_bert_embedding(word) for word in general_keywords}
    
    similarities = {
        word: cosine_similarity(product_embedding, emb).item() for word, emb in keyword_embeddings.items()
    }
    
    sorted_keywords = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_keywords[:10]]  # Return top 10 relevant keywords

# Step 4: Extract common words from the product text
def extract_common_words(text, top_n=10):
    words = re.findall(r'\b\w+\b', text.lower())  # Extract words and convert to lowercase
    common_words = Counter(words).most_common(top_n)
    return [word for word, _ in common_words]  # Return only words, not frequencies

# Step 5: Run the script on a sample product URL

product_url = "https://www.example.com/sample-product"
div_class = "tabs__content rte overflow-hidden"  # Update with actual class if needed
product_name = "Wireless Noise Cancelling Headphones"

product_text = extract_text_from_url(product_url, div_class)

general_search_terms = ["noise cancellation", "wireless earbuds", "best sound quality", "Bluetooth 5.0", 
                        "long battery life", "gaming headset", "comfortable fit", "premium audio", 
                        "fast charging", "IPX water resistance"]

optimized_keywords = find_relevant_keywords(product_name, product_text, general_search_terms)
common_words = extract_common_words(product_text)

print("Optimized Keywords:", optimized_keywords)
print("Commonly Used Words:", common_words)
