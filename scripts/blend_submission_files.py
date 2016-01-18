import argparse
import os
import sys
import pandas as pd
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('submissions', type=str, nargs='+', help='submission csv files')

    args = parser.parse_args()

    print args.submissions

    dfs = [pd.read_csv(fname) for fname in args.submissions]
    all_probas = [df[df.columns[1:]].values for df in dfs]
    all_probas = np.dstack(all_probas)
    probas = all_probas.mean(axis=2)
    probas = probas / probas.sum(axis=1)[:, np.newaxis]

    assert np.allclose(probas.sum(axis=1), 1)

    df_final = dfs[0].copy()
    df_final[df_final.columns[1:]] = probas
    df_final.head()

    basenames = [os.path.splitext(os.path.basename(fname))[0] for fname in args.submissions]
    final_fname = 'submissions/' + '+'.join(basenames) + '.csv'
    print final_fname

    if os.path.exists(final_fname):
        print 'destination exists. aborting.'
        sys.exit(1)

    df_final.to_csv(final_fname, index=False)
