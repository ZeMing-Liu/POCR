import pandas as pd
import sklearn.metrics as metrics
import numpy as np
import sys
import getopt

def evaluate(true_labels_file,predict_labels_file):
    true_labels = pd.read_csv(true_labels_file, sep="\s+")
    predict_labels = pd.read_csv(predict_labels_file, index_col=0, header=None,sep=" ")

    true_labels = np.ravel(true_labels.T)
    predict_labels = np.ravel(predict_labels.T.values)
    ari = metrics.adjusted_rand_score(true_labels, predict_labels)
    nmi = metrics.normalized_mutual_info_score(true_labels,predict_labels)
    print("ARI:"+str(ari)+" NMI:"+str(nmi))

if __name__ == '__main__':
    argv = sys.argv[1:]
    truelabels_file = ''
    predictlabels_file = ''
    try:
        opts, args = getopt.getopt(argv, "ht:p:", ["help"])
    except getopt.GetoptError:
        print('wrong input')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '--help':
            print('evaluate.py -t <true_labels_file> -p <predict_labels_file>')
            sys.exit()
        elif opt == "-t":
            truelabels_file = arg
        elif opt == "-p":
            predictlabels_file = arg
    evaluate(truelabels_file,predictlabels_file)
    print("finished.")