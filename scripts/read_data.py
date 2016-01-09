#!/usr/bin/python

import pandas as pd

def read_data(filename):

    df = pd.read_csv(filename,
                     sep=',',
                     header=0,
                     )

    data = df.date_account_created[:10]
    print data
    #print pd.to_datetime(data, format='%f')

def main():

    DIR_DATA = '../data/'
    FILENAME_TRAIN = DIR_DATA + 'train_users_2.csv'

    df_train = read_data(FILENAME_TRAIN)

if __name__ == '__main__':

    main()
