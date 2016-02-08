#!/usr/bin/python

import pandas as pd
import numpy as np
import pickle

def recreate_train_data(df, train_fraction=0.66):

    ''' Splits train data into validation and train dataset

    Parameters
    ----------
    df : pandas.DataFrame
        Train Data

    train_fraction : float
        Fraction of original data to become new train data set.

    '''

    # Get sizes of train and validation datasets
    rows = df.index.values
    size = len(rows)
    size_train = np.floor(size * train_fraction)
    size_valid = np.floor(size * (1 - train_fraction))
    if size_train + size_valid < size:
        size_train += 1

    # Select the rows
    rows_train = np.random.choice(rows,
                                  size=size_train,
                                  replace=False)
    rows_valid = np.random.choice(rows,
                                  size=size_valid,
                                  replace=False)

    df_train = df.iloc[rows_train]
    df_valid = df.iloc[rows_valid]

    return df_train, df_valid

def create_naive_prediction(df_test, df_countries):

    countries = list(df_countries.country_destination.values)
    countries.append('NDF')
    users = df_test.id.values
    users = list(users)

    # add multiple countries for some users
    rand_users = np.random.randint(0, len(users), size=len(users))
    for rand_user in rand_users:
        users.append(users[rand_user])

    countries_predict = np.random.choice(countries,
                                         size=len(users),
                                         replace=True)
    data_predict = np.array((users, countries_predict)).T

    # create empty dataset
    df_predict = pd.DataFrame(data_predict, columns=['id','country'])

    return df_predict

def write_submission(df, filename):

    df.to_csv(filename,
              sep=',',
              index=False,
              )

def convert_categorical_data(df, cols=[]):

    # copy new dataframe to be joined with dummy variables, then old variables
    # removed
    df_new = df.copy()

    # cycle through each categorical variable
    for col in cols:
        # get the number of dummy variables needed
        unique_labels = np.unique(df[col].values) #[nan, A, B, C, D, F]

        # formatting issue with '-unknown-'
        df[col] = df[col].str.strip('-')

        # create a new dummy variable column for each unique label
        for label in unique_labels:
            data = np.zeros(df[col].values.size)
            data[np.where(df[col].values == label)[0]] = 1
            df_new = df_new.join(pd.Series(data=data,
                                           name=col + "_is_" + str(label),
                                           dtype=int),
                                           )

        # remove the original categorical column
        del df_new[col]

    return df_new

def convert_nans(df):

    # convert nans to mean value of each column.
    # using inplace=True to actually change the contents of df
    for col in df.columns.values:
        df[col].fillna(np.nanmean(df[col].values), inplace=True)

    return df

def convert_labels(df, init_type=str, labels=None):
    # if initial type is a string, convert to integers
    if init_type is str:
        labels =  np.unique(df)
        df_data = df.copy()
        for i in xrange(len(labels)):
            label = labels[i]
            df_data[df == label] = i
    # if initial type is an integer, convert to strings
    elif init_type is int:
        # countries have 3 character code
        df_data = np.chararray(df.shape, itemsize=3)
        df_unique =  np.unique(df)

        # cycle through each country
        for i in xrange(len(labels)):
            label = labels[i]
            df_data[np.where(df == i)[0]] = label

    return df_data

def prep_data(df_train, df_test):

    ''' Makes dummy variables and removes nans
    '''

    columns_to_keep = ['age',
                       'gender',
                       'signup_method',
                       'language',
                       'first_device_type',
                       'first_browser',
                       ]

    df_train_data = df_train[columns_to_keep]
    df_test_data = df_train[columns_to_keep]
    df_labels = pd.DataFrame(df_train[df_train.columns.values[-1]])

    # get a list of all the counties
    country_labels = np.unique(df_labels)

    # get the user ids from the test data
    print df_test.columns.values
    ids = df_test['id']

    # Convert column data containing strings to dummy variables
    # e.g. column 'gender' with 'male' and 'female' will be two columns:
    # column 'gender_is_male' and column 'gender_is_female', each with values of
    # either 0 or 1
    columns_categorical = ['gender',
                           'signup_method',
                           'language',
                           'first_device_type',
                           'first_browser',
                           ]

    df_train_data = convert_categorical_data(df_train_data,
                                             cols=columns_categorical,)
    df_test_data = convert_categorical_data(df_test_data,
                                             cols=columns_categorical,)
    df_labels_data = convert_categorical_data(df_labels,
                                              cols=['country_destination',],)

    # convert nans to mean value
    df_train_data = convert_nans(df_train_data)
    df_test_data = convert_nans(df_test_data)

    return df_train_data, df_test_data, df_labels_data, country_labels, ids

def merge_predictions(ids, df_predict):

    df_merge = pd.concat([ids, df_predict], axis=1)

    return df_merge

def predict_labels(df_train, df_test, crop=1):

    # see http://yandex.github.io/rep/estimators.html#module-rep.estimators.sklearn

    import pandas as pd

    # crop to smaller datasets for testing:
    if crop:
        df_train = df_train.drop(df_train.index[1000:], inplace=0)
        df_test = df_test.drop(df_test.index[1000:], inplace=0)

    print('\nPrepping data...')
    df_train, df_test, df_labels, countries, ids = prep_data(df_train, df_test)

    print 'df_labels', df_labels.columns
    print 'countries', countries

    # convert the labels to integers
    #df_labels_data = convert_labels(df_labels, init_type=str)


    # fit the data

    filename = '../data_products/prediction.pickle'
    if 1:
        print('\nFitting regressor...')
        df_predict = fit_categorical_labels(df_train, df_test, df_labels,
                               labels_list=countries)
        df_predict.to_pickle('../data_products/prediction.pickle')
    else:
        df_predict = pd.read_pickle(filename).squeeze()

    # merge test user ids with predictions
    df_predict = merge_predictions(ids, df_predict)

    return df_predict

def fit_categorical_labels(df_train, df_test, df_labels, fit_type='regressor',
        labels_list=None):

    from rep.estimators import SklearnClassifier, SklearnRegressor
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import GradientBoostingRegressor

    # Using gradient boosting with default settings
    if fit_type == 'classifier':
        sk = SklearnClassifier(GradientBoostingClassifier(),
                               features=df_train.columns.values)
    elif fit_type == 'regressor':
        sk = SklearnRegressor(GradientBoostingRegressor(),
                              features=df_train.columns.values)


    prediction_array = np.empty(df_labels.shape)
    for i, column in enumerate(df_labels.columns.values):
        # get a single column to predict
        labels = df_labels[column]

        # fit the data with the training set
        sk.fit(df_train, labels)

        # predict new countries
        prediction = sk.predict(df_test)
        prediction_array[:, i] = prediction

        #prediction = pd.read_pickle(filename).squeeze()

    df_predict = pd.DataFrame(prediction_array, columns=df_labels.columns.values)
    df_predict = gather_dummy_predictions(df_predict, labels_list)

    return df_predict

def gather_dummy_predictions(df_predict, labels):

    # construct empty file
    orig_col = np.chararray(len(df_predict), itemsize=3)

    print labels

    for i in xrange(len(df_predict.axes[0])):

        row = df_predict.iloc[i]

        # use the label with the highest probability
        idx_max = np.where(row == np.max(row))[0][0]
        orig_col[i] = labels[idx_max]

    orig_col = pd.DataFrame(orig_col, columns=['country'])

    return orig_col


