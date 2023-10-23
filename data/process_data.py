import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge message and category data from CSV files to create a structured DataFrame.

    This function reads two CSV files containing message and category data, processes the data,
    and merges them to create a structured DataFrame with categories as columns.

    Args:
        messages_filepath (str): File path to the CSV file containing message data.
        categories_filepath (str): File path to the CSV file containing category data.

    Returns:
        pandas.DataFrame: A DataFrame containing the merged data with each message's associated categories
                         represented as individual columns.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")
    categories = df['categories'].str.split(";",expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split("-")[0])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
    categories['related'] = categories['related'].map(lambda x: 1 if x==2 else x)
    df = df.drop(['categories'],axis=1)
    df = pd.concat([df, categories],axis=1)
    return df

def clean_data(df):
    """
    Clean and remove duplicate messages from a DataFrame.

    This function identifies and removes duplicate messages in a given DataFrame, keeping only the
    first occurrence of each unique message. It helps to ensure that the dataset contains only
    unique messages, which is often necessary in data processing.

    Args:
        df (pandas.DataFrame): The DataFrame containing message data to be cleaned.

    Returns:
        pandas.DataFrame: A cleaned DataFrame with duplicate messages removed.
    """
    df['message'].duplicated().sum()
    df = df.drop_duplicates(subset=['message'], keep='first')
    return df


def save_data(df, database_filename):
    """
    Save a DataFrame to an SQLite database.

    This function takes a pandas DataFrame and saves it to an SQLite database at the specified
    database file path. The DataFrame is stored as a table named 'MessageDatabase2' within the
    SQLite database.

    Args:
        df (pandas.DataFrame): The DataFrame to be saved to the database.
        database_filename (str): The file path for the SQLite database where the DataFrame will be stored.

    Returns:
        None: The function does not return any values.
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('MessageDatabase2', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
