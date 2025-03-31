import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
import numpy as np

# Download necessary NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Step 1: Web Scraping - Extract Product Description
def extract_product_description(url, div_class):
    """
    Fetch the product description from a given webpage.
    
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

# Step 3: Generate Bag of Words
def generate_bag_of_words(texts):
    """
    Converts text into a bag of words model.
    
    Args:
        texts (list): List of cleaned text data
    
    Returns:
        list: List of unique words
    """
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    return vectorizer.get_feature_names_out()  # Returns unique words

# Step 4: Train Word2Vec Model
def train_word2vec(texts):
    """
    Trains a Word2Vec model on tokenized words.
    
    Args:
        texts (list): Tokenized words in list format
    
    Returns:
        Word2Vec: Trained Word2Vec model
    """
    model = Word2Vec(sentences=texts, vector_size=100, window=5, min_count=1, workers=4)
    return model

# Step 5: Find Similar Words
def get_similar_words(word, model):
    """
    Finds words similar to the given word using Word2Vec.
    
    Args:
        word (str): Input word for similarity search
        model (Word2Vec): Trained Word2Vec model
    
    Returns:
        list: List of similar words with similarity scores
    """
    try:
        return model.wv.most_similar(word, topn=5)
    except KeyError:
        return []

# Step 6: Implement the Process
url = input("https://example.com/product-page")  # Replace with actual product URL
div_class =input( "product-description" ) # Replace with actual class of product description div

# Extract product content
product_text = extract_product_description(url, div_class)

if product_text:
    print("\nExtracted Product Description:\n", product_text)
    
    # Preprocess the text
    words = preprocess_text(product_text)
    print("\nTokenized Words:\n", words)

    # Create Bag of Words
    bow = generate_bag_of_words([" ".join(words)])
    print("\nBag of Words:\n", bow)

    # Train Word2Vec model
    model = train_word2vec([words])

    # Get related words for a given keyword
    search_keyword = "noise"  # Example search keyword
    similar_words = get_similar_words(search_keyword, model)
    print(f"\nSimilar words for '{search_keyword}':\n", similar_words)
