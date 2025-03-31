import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Step 1: Extract Content and Product Name from a Specific <div>
def extract_text_and_product_name(url, div_class):
    response = requests.get(url)
    if response.status_code != 200:
        return "Error fetching page", ""

    soup = BeautifulSoup(response.text, "html.parser")
    content_div = soup.find("div", {"class": div_class})
    
    if not content_div:
        return "No content found", ""
    
    # Extract product name from <h2> inside the div
    product_name_tag = content_div.find("h2")
    product_name = product_name_tag.get_text(strip=True) if product_name_tag else "Unknown Product"
    
    return content_div.get_text(strip=True), product_name

# Step 2: Generate BERT Embeddings
def get_bert_embedding(text):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**tokens)
    
    return output.last_hidden_state.mean(dim=1)  # Sentence embedding

# Step 3: Extract Commonly Used Words
def extract_common_words(text, top_n=10):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())  # Extract words of at least 3 letters
    filtered_words = [word for word in words if word not in stop_words]  # Remove stop words
    word_counts = Counter(filtered_words)
    return [word for word, _ in word_counts.most_common(top_n)]

# Step 4: Generate Dynamic Search Terms from Product Name
def generate_dynamic_search_terms(product_name):
    words = product_name.lower().split()
    
    # Generate possible user search queries based on product category
    dynamic_terms = []
    for word in words:
        dynamic_terms.append(f" {word}")
        # dynamic_terms.append(f"{word} ingredients")
        # dynamic_terms.append(f"{word} review")
        # dynamic_terms.append(f"{word} 2025")
    
    return list(set(dynamic_terms))  # Remove duplicates

# Step 5: Find Relevant Keywords
def find_relevant_keywords(product_name, product_text):
    full_text = product_name + " " + product_text  # Combine product name and extracted text
    product_embedding = get_bert_embedding(full_text)

    # Dynamically generate search terms
    general_keywords = generate_dynamic_search_terms(product_name)

    # Generate embeddings for the search terms
    keyword_embeddings = {word: get_bert_embedding(word) for word in general_keywords}

    # Compute similarity scores
    similarities = {
        word: cosine_similarity(product_embedding, emb).item() for word, emb in keyword_embeddings.items()
    }

    # Return top 10 relevant search keywords
    sorted_keywords = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_keywords[:10]]

# Step 6: Run the Script
product_url = input("enter the product Url")
div_class = "tabs__content rte overflow-hidden"  # Update with actual class if needed

# Extract product description text and product name
product_text, product_name = extract_text_and_product_name(product_url, div_class)
print("________product_text________________",product_text )
print("________product_name________________",product_name )
# Extract common words from the product description
common_words = extract_common_words(product_text, top_n=10)

# Generate optimized keywords
optimized_keywords = find_relevant_keywords(product_name, product_text)

# Print Results
print("\nExtracted Product Name:", product_name)
print("\nExtracted Common Words:", common_words)
print("\nOptimized Keywords for Search Ranking:", optimized_keywords)
