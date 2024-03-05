## This script is used to load, clean and save the data to a sqlite database

# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load data from messages and categories files and 
    merge them into a single dataframe

    Args:
        messages_filepath (str): path to messages file
        categories_filepath (str): path to categories file

    Returns:
        pd.DataFrame: merged dataframe
    """    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """Clean the dataframe by splitting the categories column into
    separate columns and converting the values to binary
    Args:
        df (pd.DataFrame): dataframe to be cleaned
        

    Raises:
        ValueError: When input dataframe is None 
        ValueError: When input dataframe does not contain 'categories' column

    Returns:
        df (pd.DataFrame): cleaned dataframe
    """    
    if df is None:
        raise ValueError("Input dataframe is None")
    if 'categories' not in df.columns:
        raise ValueError("Input dataframe does not contain 'categories' column")
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[1]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], join='inner', axis=1)
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """Save the dataframe to a sqlite database

    Args:
       df (pd.DataFrame): dataframe to be saved
        database_filename (str): path to the database file
    Outputs:
        None
    """    
    database_filename = 'sqlite:///' + database_filename
    engine = create_engine(database_filename)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


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