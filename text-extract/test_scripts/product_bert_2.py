import requests
from bs4 import BeautifulSoup
import re
import nltk
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

# Load pre-trained BERT model & tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Step 1: Web Scraping - Extract Product Description
def extract_product_description(url, div_class):
    """
    Fetches the product description from a given webpage.
    
    Args:
        url (str): Product URL
        div_class (str): CSS class of the product description div
    
    Returns:
        str: Extracted product description text
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print("Failed to fetch the webpage.")
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    product_desc = soup.find('div', class_=div_class)  # Modify class name as needed

    if product_desc:
        return product_desc.get_text(strip=True)
    return None

# Step 2: Text Preprocessing
def preprocess_text(text):
    """
    Cleans and tokenizes text for analysis.
    
    Args:
        text (str): Raw product description text
    
    Returns:
        list: Tokenized words after cleaning
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    words = text.split()  # Tokenization
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return words

# Step 3: Generate BERT Embeddings
def get_bert_embedding(text):
    """
    Converts text into a BERT embedding.
    
    Args:
        text (str): Input text
    
    Returns:
        numpy.array: BERT vector representation of text
    """
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Extract [CLS] token representation

# Step 4: Find Similar Words
def get_most_similar_words(query, words, word_embeddings):
    """
    Finds the most relevant words based on BERT similarity.
    
    Args:
        query (str): Search keyword
        words (list): List of product description words
        word_embeddings (dict): Dictionary mapping words to BERT embeddings
    
    Returns:
        list: Top 5 similar words
    """
    query_embedding = get_bert_embedding(query)  # Convert query to BERT embedding
    
    similarities = []
    for word in words:
        if word in word_embeddings:
            word_embedding = word_embeddings[word].reshape(1, -1)
            similarity = cosine_similarity(query_embedding, word_embedding)[0][0]
            similarities.append((word, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)  # Sort by highest similarity
    return similarities[:10]  # Return top 5 similar words

# Step 5: Implement the Process
url = input("https://example.com/product-page")  # Replace with actual product URL
div_class = input("product-description")  # Replace with actual class of product description div

# Extract product content
product_text = extract_product_description(url, div_class)

if product_text:
    print("\nExtracted Product Description:\n", product_text)
    
    # Preprocess the text
    words = preprocess_text(product_text)
    print("\nTokenized Words:\n", words)

    # Generate BERT embeddings for each word
    word_embeddings = {word: get_bert_embedding(word) for word in words}

    # Get related words for a given keyword
    search_keyword = "noise cancellation"  # Example search query
    similar_words = get_most_similar_words(search_keyword, words, word_embeddings)
    
    print(f"\nSimilar words for '{search_keyword}':\n", similar_words)
