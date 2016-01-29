#!/usr/bin/python

import pandas as pd
import myfuncs
import numpy as np

def read_data(filename, filetype='csv'):

    if filetype == 'csv':
        df = pd.read_csv(filename,
                         sep=',',
                         header=0,
                         )
    else:
        df = pd.read_pickle(filename,
                            )

    return df

def create_naive_prediction(data):

    pass

def write_data(filename, df):

    df.to_pickle(filename)

    # what?

def main():

    # Filenames
    DIR_DATA = '../data/'
    DIR_DATA_PROD = '../data_products/'
    FILENAME_TRAIN_CSV = DIR_DATA + 'train_users.csv'
    FILENAME_TEST_CSV = DIR_DATA + 'test_users.csv'
    FILENAME_COUNTRIES = DIR_DATA + 'countries.csv'
    FILENAME_TRAIN = DIR_DATA_PROD + 'train'
    FILENAME_TEST = DIR_DATA_PROD + 'test'
    FILENAME_VALID = DIR_DATA_PROD + 'valid'
    FILENAME_PREDICT = DIR_DATA_PROD + 'predict'

    # Options
    RECREATE_TRAIN_DATA = 0

    # Read in data
    if RECREATE_TRAIN_DATA:
        df_train = read_data(FILENAME_TRAIN_CSV)
        df_train, df_valid = myfuncs.recreate_train_data(df_train)

        write_data(FILENAME_TRAIN + '.pickle', df_train)
        write_data(FILENAME_VALID + '.pickle', df_valid)
    else:
        if 0:
            df_train = read_data(FILENAME_TRAIN + '.pickle',
                                 filetype='df')
            df_valid = read_data(FILENAME_VALID + '.pickle',
                                 filetype='df')
        df_train = read_data(FILENAME_TRAIN_CSV)



    df_test = read_data(FILENAME_TEST_CSV,)
    df_countries = read_data(FILENAME_COUNTRIES,)

    print np.unique(df_train.language.values)
    print df_countries.columns.values
    print np.unique(df_countries.lng_destination.values)

    # Begin analysis
    df_predict = myfuncs.create_naive_prediction(df_test, df_countries)

    df_predict = myfuncs.predict_neural_network(df_train, df_test)

    # write prediction
    myfuncs.write_submission(df_predict, FILENAME_PREDICT + '.csv')

if __name__ == '__main__':
    main()

