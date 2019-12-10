import sys

# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle
import re
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])

def load_data(database_filepath):
    """Load dataset from database and assign X and Y 
    Args:
        database_filepath (str): string filepath of the sqlite database
    Returns:
        X (pandas dataframe): Feature data
        Y (pandas dataframe): Classification labels
        category_names (list): List of the category names for evaluate the model

    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM disaster_ETL", con=engine)
   
    X = df.iloc[:,:-36]
    Y = df.iloc[:,4:40]
    category_names = Y.columns.tolist()

    return X , Y , category_names

def tokenize(text):
    """Normalize, tokenize , remve stepwords and stem text string
    Args:
    text: string. String containing message for processing
       
    Returns:
    tokens: list of strings. List containing cleaned tokens in the message
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        clean_tokens = [w for w in clean_tokens if w not in stopwords.words("english")]

    return clean_tokens


def build_model():
    """Returns the GridSearchCV object to be used as the model
    Args:
        None
    Returns:
        cv (scikit-learn GridSearchCV): Grid search model object
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # specify parameters for grid search
    parameters = {
         'clf__estimator__n_estimators': [20],
         'clf__estimator__min_samples_split': [2]
}

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv
    
def evaluate_model(model, X_test, Y_test, category_names):
    """Prints multi-output classification results
    Args:
        model (pandas dataframe): the model
        X_text (pandas dataframe): The X test set
        Y_test (pandas dataframe): the Y test labels
        category_names (list): the category names
    Returns:
        None
    """
    y_pred = model.predict(X_test['message'])

    print(classification_report(Y_test.iloc[:,1:].values, np.array([x[1:] for x in y_pred]), target_names=category_names))



def save_model(model, model_filepath):
    """save model
    
    Args:
    model: model object. the Fitted model.
    model_filepath: string. the filepath to save the model
    
    Returns:
    None
    """
    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train['message'], Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()