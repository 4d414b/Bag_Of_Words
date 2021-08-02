import pandas as pd

# If you do not already have the data locally you may download (and save) by

file_location = 'https://gitlab.com/michaelallen1966/00_python_snippets' +\
    '_and_recipes/raw/master/machine_learning/data/IMDb.csv'
imdb = pd.read_csv(file_location)
# save to current directory
imdb.to_csv('imdb.csv', index=False)

# If you already have the data locally then you may run the following

# Load data example
imdb = pd.read_csv('imdb.csv')

# Truncate data for example if you want to speed up the example
# imdb = imdb.head(5000)

import nltk
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# If not previously performed:
# nltk.download('stopwords')

stemming = PorterStemmer()
stops = set(stopwords.words("english"))

def apply_cleaning_function_to_list(X):
    cleaned_X = []
    for element in X:
        cleaned_X.append(clean_text(element))
    return cleaned_X


def clean_text(raw_text):
    """This function works on a raw text string, and:
        1) changes to lower case
        2) tokenizes (breaks down into words
        3) removes punctuation and non-word text
        4) finds word stems
        5) removes stop words
        6) rejoins meaningful stem words"""
    
    # Convert to lower case
    text = raw_text.lower()
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Keep only words (removes punctuation + numbers)
    # use .isalnum to keep also numbers
    token_words = [w for w in tokens if w.isalpha()]
    
    # Stemming
    stemmed_words = [stemming.stem(w) for w in token_words]
    
    # Remove stop words
    meaningful_words = [w for w in stemmed_words if not w in stops]
    
    # Rejoin meaningful stemmed words
    joined_words = ( " ".join(meaningful_words))
    
    # Return cleaned data
    return joined_words

# Get text to clean
text_to_clean = list(imdb['review'])

# Clean text
cleaned_text = apply_cleaning_function_to_list(text_to_clean)

# Add cleaned data back into DataFrame
imdb['cleaned_review'] = cleaned_text

# Remove temporary cleaned_text list (after transfer to DataFrame)
del cleaned_text

from sklearn.model_selection import train_test_split
X = list(imdb['cleaned_review'])
y = list(imdb['sentiment'])
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size = 0.25)

def create_bag_of_words(X):
    from sklearn.feature_extraction.text import CountVectorizer
    
    print ('Creating bag of words...')
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.  
    
    # In this example features may be single words or two consecutive words
    # (as shown by ngram_range = 1,2)
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 ngram_range = (1,2), \
                                 max_features = 10000
                                ) 

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of 
    # strings. The output is a sparse array
    train_data_features = vectorizer.fit_transform(X)
    
    # Convert to a NumPy array for easy of handling
    train_data_features = train_data_features.toarray()
    
    # tfidf transform
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf = TfidfTransformer()
    tfidf_features = tfidf.fit_transform(train_data_features).toarray()

    # Get words in the vocabulary
    vocab = vectorizer.get_feature_names()
   
    return vectorizer, vocab, train_data_features, tfidf_features, tfidf

vectorizer, vocab, train_data_features, tfidf_features, tfidf  = \
    create_bag_of_words(X_train)

bag_dictionary = pd.DataFrame()
bag_dictionary['ngram'] = vocab
bag_dictionary['count'] = train_data_features[0]
bag_dictionary['tfidf_features'] = tfidf_features[0]

# Sort by raw count
bag_dictionary.sort_values(by=['count'], ascending=False, inplace=True)
# Show top 10
print(bag_dictionary.head(10))

def train_logistic_regression(features, label):
    print ("Training the logistic regression model...")
    from sklearn.linear_model import LogisticRegression
    ml_model = LogisticRegression(C = 100,random_state = 0)
    ml_model.fit(features, label)
    print ('Finished')
    return ml_model

ml_model = train_logistic_regression(tfidf_features, y_train)

test_data_features = vectorizer.transform(X_test)
# Convert to numpy array
test_data_features = test_data_features.toarray()

test_data_tfidf_features = tfidf.fit_transform(test_data_features)
# Convert to numpy array
test_data_tfidf_features = test_data_tfidf_features.toarray()

predicted_y = ml_model.predict(test_data_tfidf_features)
correctly_identified_y = predicted_y == y_test
accuracy = np.mean(correctly_identified_y) * 100
print ('Accuracy = %.0f%%' %accuracy)