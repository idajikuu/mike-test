from test import *
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/api', methods=['GET', 'POST'])
def add_message():
    content = request.json
    sentence = content["url"]
    return jsonify({"results":predict(scrape_website(sentence))})

if __name__ == '__main__':
    app.run(host= '0.0.0.0',debug=True)