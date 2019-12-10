import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load and merge messages and categories datasets
    
    Args:
    messages_filepath: string. Filepath for csv file containing messages dataset.
    categories_filepath: string. Filepath for csv file containing categories dataset.
       
    Returns:
    df: dataframe. Dataframe containing merged content of messages and categories datasets.
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)    
    #replace the NA on the original columns by the values on the message columns
    messages.loc[messages['original'].isnull(),'original'] = messages['message']
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets and the common columns is id
    df = messages.merge(categories, how='outer',on=['id'])
    return df

def clean_data(df):
    """Clean dataframe by Split categories into separate category columns and converting categories from strings 
    to 0 and 1 afrer that removing duplicates .
    
    Args:
    df: dataframe. Dataframe containing merged content of messages and categories datasets.
       
    Returns:
    df: dataframe. containing clean dataframe.
    """
    # create a dataframe of the 36 individual category columns
    categories_split = df['categories'].str.split(";", n=36, expand=True)
    # select the first row of the categories_split dataframe
    row = categories_split.iloc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.transform(lambda x: x[:-2]).tolist()
    # rename the columns of `categories_split`
    categories_split.columns = category_colnames
    #Convert category values to just numbers 0 or 1
    for column in categories_split:
    # set each value to be the last character of the string
        categories_split[column] = pd.Series(categories_split[column]).str.extract(r'[-](\d)', expand=True)
    
    # convert column from string to numeric
        categories_split[column] = categories_split[column].astype(int)
    
   # drop the original categories column from `df`
    df.drop('categories',axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories_split` dataframe
    df = pd.concat([df,categories_split], join='inner', axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    """Save cleaned data into an SQLite database.
    
    Args:
    df: dataframe. containing clean dataframe of merged message and 
    categories data.
    database_filename: string. Filename for database.
       
    Returns:
    None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_ETL', engine, index=False, if_exists='replace')  


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