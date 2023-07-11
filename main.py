import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from PyPDF2 import PdfReader



class SimpleChatbot:
    def __init__(self, filepath):
        # Loading the text from the file
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        # Splitting the text into sentences
        self.text = sent_tokenize(raw_text)

        # Initialize TfidfVectorizer and transform the text
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.text_vectors = self.vectorizer.fit_transform(self.text)

    def find_relevant_section(self, question):
        # Transform the question into tf-idf vector
        question_vector = self.vectorizer.transform([question])

        # Compute the cosine similarity between the question vector and the text vectors
        similarities = cosine_similarity(question_vector, self.text_vectors)

        # Find the section with the highest cosine similarity
        most_similar_index = similarities.argmax()

        # If none of the sections are similar to the question, return None
        if similarities[0][most_similar_index] == 0:
            return None
        else:
            return self.text[most_similar_index]

    def generate_response(self, section, question):
        prompt = section + "\n" + question + "\n"
        
        api_key = 'INSERT API KEY HERE'
        
        openai.api_key = api_key
        
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.5,
            max_tokens=100,
        )

        generated_text = response.choices[0].text.strip()

        return generated_text

    def ask_question(self, question):
        section = self.find_relevant_section(question)
        if section is None:
            return "Sorry, I don't know the answer to that question."
        else:
            return self.generate_response(section, question)


def pdf_to_text(pdf_path, txt_path):
     # Open the PDF file
    pdf = PdfReader(pdf_path)

    # Initialize a string to store the text
    text = ""

    # Loop through each page in the PDF and add its text to the string
    for page in pdf.pages:
        text += page.extract_text()

    # Write the text to the output text file
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)


# Convert the PDF to a text file
pdf_to_text('ABA_ai_book1.pdf', 'stored_info.txt')

# Create a chatbot instance with the path to the text file
chatbot = SimpleChatbot('stored_info.txt')

# Ask the user for their question
question = input("Please enter your question: ")

# Get the response
response = chatbot.ask_question(question)

# Print the response
print("Response: ", response)
