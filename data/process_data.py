import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

    """
    Load and preprocess data from two CSV files, merging them into a single DataFrame.

    Parameters:
        messages_filepath (str): Filepath to the CSV file containing message data.
        categories_filepath (str): Filepath to the CSV file containing category data.

    Returns:
        pandas.DataFrame: A DataFrame that combines message and category data with proper data preprocessing.

    This function loads data from two separate CSV files, one containing messages and another containing categories.
    It merges the data on the 'id' column and then performs several preprocessing steps to make the data suitable for analysis:

    1. Split the 'categories' column into separate category columns.
    2. Extract the category names from the first row of the categories DataFrame.
    3. Clean and convert the category values to integers.
    4. Map the 'related' category to ensure it contains only 0 and 1 values.
    5. Remove the original 'categories' column.
    6. Concatenate the cleaned category columns with the original message data.

    The resulting DataFrame is a cleaned and structured dataset for further analysis and modeling.
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
    Remove duplicate messages from a DataFrame.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing message data.

    Returns:
        pandas.DataFrame: A DataFrame with duplicate messages removed.

    This function takes a DataFrame as input and performs the following steps to clean the data:

    1. Identifies and counts the number of duplicated messages in the 'message' column.
    2. Drops duplicate messages while keeping only the first occurrence, ensuring unique messages.

    The resulting DataFrame contains a clean and de-duplicated dataset with unique messages.

    """
    
    df['message'].duplicated().sum()
    df = df.drop_duplicates(subset=['message'], keep='first')
    return df


def save_data(df, database_filename):

    """
    Save a DataFrame to an SQLite database.

    Parameters:
        df (pandas.DataFrame): The DataFrame to be saved to the database.
        database_filename (str): The filepath for the SQLite database where the DataFrame will be stored.

    This function saves the provided DataFrame into an SQLite database with the specified filename. The steps include:

    1. Creating an SQLite database engine using the provided filepath.
    2. Deriving a table name from the database filename (lowercase, without the extension).
    3. Saving the DataFrame to the database under the derived table name.
    4. Printing the name of the database table that was created.
    """

    engine = create_engine('sqlite:///' + database_filename)
    table_name = database_filename.split(".")[0].lower()
    df.to_sql(table_name, engine, index=False)  
    print("Database table name is: " + table_name)


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
