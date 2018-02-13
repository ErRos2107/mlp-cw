import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def read_from_file():
    os.chdir(raw_data_path)

    stp = set(stopwords.words('english'))


if __name__ == '__main__':
    read_from_file()
