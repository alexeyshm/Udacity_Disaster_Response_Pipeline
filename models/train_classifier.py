import sys

import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
import re
from nltk.stem import WordNetLemmatizer
import nltk
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
nltk.download('punkt')
nltk.download('wordnet')


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM messages_categories""", engine)
    X = df['message']
    Y = df[[c for c in df.columns if c not in ('id', 'message', 'original', 'genre')]]
    categories = Y.columns

    return X, Y, categories


def tokenize(text):

    text = re.sub(r"[^a-zA-Z]", " ", text).lower()
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w).strip() for w in tokens]

    return tokens


def build_model():
    classifier = RandomForestClassifier()
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('moc', MultiOutputClassifier(estimator=classifier))
    ])


    parameters = {
        'tfidf__use_idf': (True, False),
        'vect__max_df': (0.5, 1.0),
        'moc__estimator__n_estimators': [50, 100],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns=category_names)

    for cat in category_names:
        report = classification_report(Y_test[cat], Y_pred[cat], output_dict=True)
        print(f"{cat}: F1-score = {report['weighted avg']['f1-score']}; \
        Precision = {report['weighted avg']['precision']}; \
        Recall: = {report['weighted avg']['recall']}")


def save_model(model, model_filepath):
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