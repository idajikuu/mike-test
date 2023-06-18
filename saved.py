import pandas as pd
from urllib.request import urlopen, Request
from urllib.error import URLError
from bs4 import BeautifulSoup
import re
import urllib
from transformers import AutoTokenizer, AutoModelForSequenceClassification,RobertaTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

max_length = 627

tokenizer = RobertaTokenizer.from_pretrained("hananeChab/ai-detector2")
model = TFRobertaForSequenceClassification.from_pretrained("hananeChab/ai-detector2")
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
                result += text + " "

            return result.strip()  # Remove leading/trailing spaces and return the result string
        else:
            print("Failed to retrieve the webpage.")
    except URLError as e:
        print("An error occurred while making the request:", e)

    return ""  # Return an empty string if any exception occurs

def process_row(row):
    try:
        index = row.name
        domain = row['domain']
        url = row['url']

        # Perform scraping on the URL
        website_text = scrape_website(url)

        # Perform AI calculation and update scores
        real_score, ai_score, predicted_label = segmentation(website_text)

        return index, domain, url, real_score, ai_score, predicted_label
def process_excel_file(file_path, batch_size=100):
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Add the new column names to the DataFrame
    df['real_score'] = ''
    df['ai_score'] = ''
    df['predicted_label'] = ''

    total_rows = len(df)
    processed_rows = 0
    batch_count = 1

    # Process rows in batches
    for batch_start in range(0, total_rows, batch_size):
        batch_end = min(batch_start + batch_size, total_rows)
        batch_df = df.iloc[batch_start:batch_end]

        # Process each row in the batch
        for index, row in batch_df.iterrows():
            result = process_row(row)
            index, domain, url, real_score, ai_score, predicted_label = result

            # Handle missing values for each argument
            if 'domain' in result:
                df.at[index, 'domain'] = domain
            else:
                df.at[index, 'domain'] = 'N/A'

            if 'url' in result:
                df.at[index, 'url'] = url
            else:
                df.at[index, 'url'] = 'N/A'

            if 'real_score' in result:
                df.at[index, 'real_score'] = real_score
            else:
                df.at[index, 'real_score'] = 'N/A'

            if 'ai_score' in result:
                df.at[index, 'ai_score'] = ai_score
            else:
                df.at[index, 'ai_score'] = 'N/A'

            if 'predicted_label' in result:
                df.at[index, 'predicted_label'] = predicted_label
            else:
                df.at[index, 'predicted_label'] = 'N/A'

            processed_rows += 1
            print(f"Processed row {processed_rows} of {total_rows}")
            print(f"Updated values for row {index}:")
            print(f"Domain: {df.at[index, 'domain']}")
            print(f"URL: {df.at[index, 'url']}")
            print(f"Real Score: {df.at[index, 'real_score']}")
            print(f"AI Score: {df.at[index, 'ai_score']}")
            print(f"Predicted Label: {df.at[index, 'predicted_label']}")
            print()

        # Save the batch DataFrame to a separate Excel file
        batch_output_file_path = file_path.replace('.xlsx', f'_batch{batch_count}_scraped.xlsx')
        batch_df.to_excel(batch_output_file_path, index=False)
        print(f"Batch {batch_count} completed. Result saved to {batch_output_file_path}")
        batch_count += 1

    # Save the final DataFrame to a new Excel file
    output_file_path = file_path.replace('.xlsx', '_scraped.xlsx')
    df.to_excel(output_file_path, index=False)
    print(f"Scraping completed for {file_path}. Result saved to {output_file_path}")

def process_row(row):
    try:
        index = row.name
        domain = row['domain']
        url = row['url']

        # Perform scraping on the URL
        website_text = scrape_website(url)

        # Perform AI calculation and update scores
        real_score, ai_score, predicted_label = segmentation(website_text)

        return index, domain, url, real_score, ai_score, predicted_label
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

    # Calculate probabilities
    human_written_prob = human_written_count / total_predictions
    ai_generated_prob = ai_generated_count / total_predictions
    if ai_generated_prob > 0.5:
        predicted_label = 'AI-Generated'
    else:
        predicted_label = 'Human-Written'

    return human_written_prob, ai_generated_prob, predicted_label
