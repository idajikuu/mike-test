from test import *
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/api', methods=['GET', 'POST'])
def add_message():
    content = request.json
    sentence = content["url"]
    print(sentence)
    ai_score, real_score = predict(scrape_website(sentence))
    
    return jsonify({"AI SCORE":str(ai_score),"REAL SCORE":str(real_score)})

if __name__ == '__main__':
    app.run(host= '0.0.0.0',debug=True)