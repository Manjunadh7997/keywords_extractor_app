import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from itertools import combinations
from difflib import get_close_matches

# Ensure stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.add("the")

def extract_text_and_product_name(url, div_class):
    response = requests.get(url)
    if response.status_code != 200:
        return "Error fetching page", ""
    
    soup = BeautifulSoup(response.text, "html.parser")
    content_div = soup.find("div", {"class": div_class})
    
    if not content_div:
        return "No content found", ""
    
    product_name_tag = content_div.find("h2")
    product_name = product_name_tag.get_text(strip=True) if product_name_tag else "Unknown Product"
    
    return content_div.get_text(strip=True), product_name

def get_bert_embedding(text):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**tokens)
    
    return output.last_hidden_state.mean(dim=1)

def extract_common_words(text, top_n=10):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    filtered_words = [word for word in words if word not in stop_words]
    word_counts = Counter(filtered_words)
    return [word for word, _ in word_counts.most_common(top_n)]

def generate_dynamic_search_terms(product_name):
    words = product_name.lower().split()
    return list(set(words))

def find_relevant_keywords(product_name, product_text):
    full_text = product_name + " " + product_text
    product_embedding = get_bert_embedding(full_text)
    general_keywords = generate_dynamic_search_terms(product_name)
    keyword_embeddings = {word: get_bert_embedding(word) for word in general_keywords}
    similarities = {word: cosine_similarity(product_embedding, emb).item() for word, emb in keyword_embeddings.items()}
    sorted_keywords = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    sorted_words_without_stopwords=[word for word in sorted_keywords if word not in stop_words]
    return [word for word, _ in sorted_words_without_stopwords[:10]]
    # return [word for word, _ in sorted_keywords[:10]]
# print(stop_words)


def generate_meaningful_combinations(product_name, keywords):
    meaningful_combinations = []
    common_word_pairs = list(combinations(keywords, 2))
    
    for word1, word2 in common_word_pairs:
        combined_word = word1 + word2 if word1 in product_name or word2 in product_name else word1 + word2.capitalize()
        
        if get_close_matches(combined_word, keywords, n=1, cutoff=0.8):
            meaningful_combinations.append(combined_word)
    
    return meaningful_combinations
