import re
from bs4 import BeautifulSoup

# Function for text cleaning. It takes text form the post column and perform cleaning
# as lowercasing all words, replacing url addresses and so on 
def clean_text(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    return text
