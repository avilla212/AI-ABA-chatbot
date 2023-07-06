import PyPDF2
import openai
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources (run this once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess the data
def preprocess_data(data):
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(data)
    words = [word.lower() for sentence in sentences for word in word_tokenize(sentence)]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return words

# Used to read the whole PDF file and use num_pages to read only the first few pages
def read_pdf(file, num_pages):
    # Read the PDF file
    pdf_file = open(file, 'rb')
    read_pdf = PyPDF2.PdfReader(pdf_file)

    # Get the number of pages
    total_pages = len(read_pdf.pages)

    # Determine the actual number of pages to read
    num_pages_to_read = min(num_pages, total_pages)

    # Read the selected pages
    data = ''
    for i in range(num_pages_to_read):
        page = read_pdf.pages[i]
        data += page.extract_text()

    return data

# Used to store data to a .txt file
def store_data(data):
    with open('stored_info.txt', 'w', encoding='utf-8') as f:
        f.write(data)

# Function to extract information from the preprocessed data


def parse_data_to_answer_question(data, question):
    # Preprocess the data
    words = preprocess_data(data)

    # Placeholder example: Perform simple keyword matching
    answer = ""
    for word in words:
        if question.lower() in word:
            answer = "Found a relevant keyword: " + word
            break

    # Return the answer
    return answer
openai.api_key = 'YOUR_API_KEY'

def generate_response(question):
    # Make a request to the OpenAI API
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=question,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7
    )
    
    # Extract the generated response
    generated_response = response.choices[0].text.strip()
    
    return generated_response


# Example usage
pdf_file = 'ABA_ai_book1.pdf'
question = input('Ask a question: ')

# Ask the user for the number of pages to read
num_pages = int(input('How many pages to read? '))

# Read the PDF file
data = read_pdf(pdf_file, num_pages)

# Store the data in a text file
store_data(data)

# Parse the data to answer the question
answer = parse_data_to_answer_question(data, question)

if not answer:
    answer = generate_response(question)

# Print the answer
print(answer)
