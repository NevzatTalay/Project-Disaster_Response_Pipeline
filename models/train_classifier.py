import sys
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sqlalchemy import create_engine
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
nltk.download(['punkt','stopwords', 'wordnet', 'averaged_perceptron_tagger'])
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
    """Loads data from a SQLite database file.
    Args:
        database_filepath: The path to the SQLite database file.
    Returns:
        X: A Pandas DataFrame containing the message column from the database.
        Y: A Pandas DataFrame containing all other columns from the database,
        except for id, message, original, and genre.
        category_names: A list of the category names in the Y DataFrame.
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('MessageDatabase2', engine)  
    X = df['message']
    Y = df.drop(['id','message','original','genre'],axis=1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Tokenizes and lemmatizes text.
    Args:
        text: A string containing the text to be tokenized.
    Returns:
        A list of clean tokens.
    """
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """Extracts a boolean feature indicating whether the first word in a sentence is a verb.

    This feature can be useful for tasks such as sentiment analysis, where the starting verb can often provide clues about the overall tone of a sentence.

    Attributes:
        None

    Methods:
        fit: Does nothing, as this transformer is stateless.
        transform: Takes a list of strings as input and returns a Pandas DataFrame with a single column, `starting_verb`, which contains a boolean value indicating whether the first word in each sentence is a verb.
    """

    def starting_verb(self, text):
        """
        Determine if a given text begins with a verb or a specified adverb.
        
        Args:
            text (str): The input text to be analyzed.
    
        Returns:
            bool: True if the first sentence in the text starts with a verb (VB or VBP) 
                  or if the first word is 'RT' (a specified adverb), False otherwise.
        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform a given dataset using the 'starting_verb' method.
    
        This method applies the 'starting_verb' function to each element in the input dataset 'X' and
        creates a DataFrame with the results.
    
        Args:
            X (iterable): A collection of text data, such as a list, Series, or DataFrame, where
                         each element represents a piece of text to be analyzed.
    
        Returns:
            pandas.DataFrame: A DataFrame where each row corresponds to an element in the input dataset 'X',
                             and each cell contains a Boolean value indicating whether the corresponding text
                             starts with a verb (VB or VBP) or a specified adverb ('RT'). True indicates
                             the text starts with a verb or 'RT', and False indicates otherwise.
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    """
    Create and configure a machine learning model for multi-output classification.

    This function assembles a machine learning pipeline that includes text processing and
    a multi-output random forest classifier. It also performs hyperparameter tuning using
    GridSearchCV.

    Returns:
        sklearn.model_selection.GridSearchCV: A GridSearchCV object that can be used to
        fine-tune the model's hyperparameters.
    """
    rfc_classifier = RandomForestClassifier()
    multi_rfc_classifier = MultiOutputClassifier(rfc_classifier)
    pipeline_2 = Pipeline([
    ('features', FeatureUnion([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        ('starting_verb', StartingVerbExtractor())
        ])),
            ('clf2', multi_rfc_classifier)
        ])
    param_grid2 = {
        'clf2__estimator__min_samples_leaf': (1,2),
    }
    clf2 = GridSearchCV(pipeline_2, param_grid=param_grid2, verbose=2, n_jobs=-1)
    return clf2

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of a multi-output classification model and print a classification report.

    This function takes a trained multi-output classification model, the test dataset, and a list of
    category names and calculates the classification report based on the model's predictions on the
    test data. The classification report includes precision, recall, f1-score, and support for each
    category.

    Args:
        model: A trained multi-output classification model, such as a scikit-learn estimator.
        X_test: The feature data of the test dataset.
        Y_test: The true labels of the test dataset, where each label corresponds to one or more categories.
        category_names (list of str): A list of category names for labeling in the classification report.

    Returns:
        None: The function prints the classification report to the console.
    """
    y_pred = model.predict(X_test)
    class_report = classification_report(Y_test, y_pred, target_names=category_names)
    print(class_report)
    


def save_model(model, model_filepath):
    """
    Save a machine learning model to a binary file.

    This function allows you to save a trained machine learning model to a binary file
    at the specified file path. The saved model can later be loaded and used for predictions
    or further analysis.

    Args:
        model: A trained machine learning model, such as a scikit-learn estimator.
        model_filepath (str): The file path where the model will be saved.

    Returns:
        None: The function does not return any values.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
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
