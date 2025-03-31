import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertModel
import torch
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from urllib.parse import quote_plus

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

def search_product_on_amazon(product_name):
    search_query = quote_plus(product_name)
    # amazon_search_url = f"https://www.amazon.com/s?k={search_query}"
    amazon_search_url = f"https://www.google.com={search_query}"
    
    headers = {"User-Agent": "Mozilla/5.0"}  # To prevent request blocking
    response = requests.get(amazon_search_url, headers=headers)
    
    if response.status_code != 200:
        return "Error fetching search results"
    
    soup = BeautifulSoup(response.text, "html.parser")
    product_links = []
    
    for link in soup.select("h2 a.a-link-normal"):  # Amazon product links are usually in <h2> tags
        href = link.get("href")
        if href.startswith("/gp") or href.startswith("/dp"):
            product_links.append("https://www.amazon.com" + href)
    
    return product_links[:5]  # Return top 5 product links

# Run the script
product_url = input("Enter the product URL: ")
div_class = "tabs__content rte overflow-hidden"

product_text, product_name = extract_text_and_product_name(product_url, div_class)
print("Extracted Product Name:", product_name)

if product_name and product_name != "Unknown Product":
    amazon_links = search_product_on_amazon(product_name)
    print("\nAmazon Product Links:")
    for link in amazon_links:
        print(link)
else:
    print("Could not extract a valid product name.")
