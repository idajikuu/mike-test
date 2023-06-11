from transformers import AutoTokenizer, AutoModelForSequenceClassification,RobertaTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import numpy as np
import pandas as pd
from urllib.request import urlopen, Request
from urllib.error import URLError
from bs4 import BeautifulSoup
import re
max_length = 627

tokenizer = RobertaTokenizer.from_pretrained('roberta-base-openai-detector')
model = TFAutoModelForSequenceClassification.from_pretrained("hananeChab/mike-model")
def predict(text):
  encoded=tokenizer(text, truncation=True, padding=True, max_length=max_length, return_tensors='tf')

  pred=model.predict(dict(encoded))

  prob = tf.nn.softmax(pred.logits)[0]
  predictions=np.argmax(prob)
  return prob


def scrape_website(url):
    try:
        # Set the user agent header
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
        }

        # Create a request object with the URL and headers
        request = Request(url, headers=headers)

        # Send the GET request to the specified URL
        response = urlopen(request)

        # Check if the request was successful
        if response.getcode() == 200:
            # Read the response content
            content = response.read()

            # Create a BeautifulSoup object with the HTML content
            soup = BeautifulSoup(content, 'html.parser')

            # Find and extract the desired text from the webpage
            text_elements = soup.find_all('p')  # Extract all <p> elements, you can customize this based on your needs

            # Append the extracted text to a string
            result = ""
            for element in text_elements:
                text = re.sub('<.*?>', '', str(element))  # Remove HTML tags using regular expressions
                result += text + " "

            return result.strip()  # Remove leading/trailing spaces and return the result string
        else:
            print("Failed to retrieve the webpage.")
    except URLError as e:
        print("An error occurred while making the request:", e)
        
def segmentation(text):
    max_length = 627  # Maximum length for each chunk
    total_predictions = 0
    human_written_count = 0
    ai_generated_count = 0

    # Split the text into chunks
    chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]

    # Loop over the chunks and count predictions
    for chunk in chunks:
        prob = predict(chunk)  # Call the predict function
        total_predictions += 1
        if prob[0] > 0.5:
            human_written_count += 1
        else:
            ai_generated_count += 1
    if total_predictions == 0:
         total_predictions = 1
    # Calculate probabilities
    human_written_prob = human_written_count / total_predictions
    ai_generated_prob = ai_generated_count / total_predictions
    if ai_generated_prob > 0.5:
        predicted_label = 'AI-Generated'
    else:
        predicted_label = 'Human-Written'

    return human_written_prob, ai_generated_prob, predicted_label

def main():
    scrape_website(input("give url"))

if __name__ == "__main__":
    main()
