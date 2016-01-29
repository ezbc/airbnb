#!/usr/bin/python

import pandas as pd

def read_data(filename):

    df = pd.read_csv(filename,
                     sep=',',
                     header=0,
                     )

    print len(df.columns.values)

def main():

    DIR_DATA = '../data/'
    FILENAME_TRAIN = DIR_DATA + 'train_users.csv'
    FILENAME_TEST = DIR_DATA + 'test_users.csv'
    FILENAME_SESSIONS = DIR_DATA + 'sessions.csv'

    df_train = read_data(FILENAME_TRAIN)
    df_test = read_data(FILENAME_TEST)
    #df_sessions = read_data(FILENAME_SESSIONS)

if __name__ == '__main__':

    main()
