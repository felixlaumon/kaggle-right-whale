import argparse
import os
import pandas as pd
import cPickle as pickle
from sklearn.preprocessing import LabelEncoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    df = pd.read_csv('data/train.csv')
    df['whaleID'] = df['whaleID'].apply(lambda x: x.split('_')[1])

    encoder = LabelEncoder()
    y = df['whaleID'].values.astype(int)
    encoder.fit(y)
    pickle.dump(encoder, open('models/encoder.pkl', 'wb'))
    print 'Wrote encoder to models/encoder.pkl'
    print
    print encoder.classes_
