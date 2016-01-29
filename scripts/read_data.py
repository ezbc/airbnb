#!/usr/bin/python

import pandas as pd

def read_data(filename):

    df = pd.read_csv(filename,
                     sep=',',
                     header=0,
                     )

<<<<<<< HEAD
    print len(df.columns.values)
=======
    data = df.date_account_created[:10]
    print data
    #print pd.to_datetime(data, format='%f')
>>>>>>> fe9627f1099c3220adf245dd718988cbc588fba0

def main():

    DIR_DATA = '../data/'
<<<<<<< HEAD
    FILENAME_TRAIN = DIR_DATA + 'train_users.csv'
    FILENAME_TEST = DIR_DATA + 'test_users.csv'
    FILENAME_SESSIONS = DIR_DATA + 'sessions.csv'

    df_train = read_data(FILENAME_TRAIN)
    df_test = read_data(FILENAME_TEST)
    #df_sessions = read_data(FILENAME_SESSIONS)
=======
    FILENAME_TRAIN = DIR_DATA + 'train_users_2.csv'

    df_train = read_data(FILENAME_TRAIN)
>>>>>>> fe9627f1099c3220adf245dd718988cbc588fba0

if __name__ == '__main__':

    main()
