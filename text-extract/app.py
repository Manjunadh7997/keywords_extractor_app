from flask import Flask, render_template, request, jsonify
from utils.extractor import extract_text_and_product_name, extract_common_words, find_relevant_keywords, generate_meaningful_combinations

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract():
    data = request.json
    url = data.get('url', '')
    
    div_class = "tabs__content rte overflow-hidden"
    product_text, product_name = extract_text_and_product_name(url, div_class)
    
    if product_text.startswith("Error") or product_text.startswith("No content"):
        return jsonify({"error": product_text}), 400

    common_words = extract_common_words(product_text, top_n=7)
    optimized_keywords = find_relevant_keywords(product_name, product_text)
    meaningful_keywords = generate_meaningful_combinations(product_name, common_words)

    return jsonify({
        "product_name": product_name,
        "common_words": common_words,
        "optimized_keywords": optimized_keywords,
        "meaningful_keywords": meaningful_keywords
    })

if __name__ == '__main__':
    app.run(debug=True)
