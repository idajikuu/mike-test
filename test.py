import pandas as pd
import numpy as np
from urllib.request import urlopen, Request
from urllib.error import URLError
from bs4 import BeautifulSoup
import re
import urllib
from transformers import AutoTokenizer, AutoModelForSequenceClassification,RobertaTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
from transformers import AutoTokenizer,TFRobertaForSequenceClassification, AutoModelForSequenceClassification,RobertaTokenizer, TFAutoModelForSequenceClassification

tokenizer = RobertaTokenizer.from_pretrained("idajikuu/AI-HUMAN-detector")
model = TFRobertaForSequenceClassification.from_pretrained("idajikuu/AI-HUMAN-detector")
def segment_text(text, max_length):
    segments = []
    current_segment = ""
    
    for char in text:
        if len(current_segment) < max_length:
            current_segment += char
        else:
            segments.append(current_segment)
            current_segment = char
    
    if current_segment:
        segments.append(current_segment)
    
    return segments
def predict(text):
    max_length = 627  # Maximum length of each segment
    if text == "":
        avg_prob = [0, 0]  # Empty probabilities
        most_common_label = "Error retrieving text"  # Label for empty text
    elif len(text) > max_length:
        # Perform segmentation on the text
        segments = segment_text(text, max_length)
        probs = []
        predicted_labels = []
        for segment in segments:
            # Encode each segment using the tokenizer
            print(len(segment))
            encoded = tokenizer(segment, truncation=True, padding=True, max_length=max_length, return_tensors='tf')
            # Make predictions on the segment
            pred = model.predict(dict(encoded), verbose=0)
            prob = tf.nn.softmax(pred.logits)[0]
            predictions = np.argmax(prob)
            probs.append(prob)
            if len(probs) > 0:
                # Calculate the average probabilities across segments
                avg_prob = sum(probs) / len(probs)
            else:
                avg_prob = [0, 0]
    else:
        # Process the text without segmentation
        encoded = tokenizer(text, truncation=True, padding=True, max_length=max_length, return_tensors='tf')
        pred = model.predict(dict(encoded), verbose=0)
        prob = tf.nn.softmax(pred.logits)[0]
        predictions = np.argmax(prob)

        if len(probs) > 0:
                # Calculate the average probabilities across segments
                avg_prob = sum(probs) / len(probs)
        else:
                avg_prob = [0, 0]

    return avg_prob
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen
import urllib.parse
from bs4 import BeautifulSoup
import re

def scrape_website(url):
    try:
        # Set the user agent header
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
        }

        # Encode the URL using the appropriate encoding method
        encoded_url = urllib.parse.quote(url, safe=':/')

        # Create a request object with the encoded URL and headers
        request = Request(encoded_url, headers=headers)

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
                text = text.replace('\n', ' ')  # Remove line breaks
                result += text + " "

            return result.strip()  # Remove leading/trailing spaces and return the result string
        else:
            print("Failed to retrieve the webpage.")
    except HTTPError as e:
        print("HTTP Error:", e)
    except URLError as e:
        print("An error occurred while making the request:", e)

    return ""  # Return an empty string if any exception occurs
def main():
    url = input("give url")
    predict = scrape_website(url)
    print(predict)

if __name__ == "__main__":
    main()