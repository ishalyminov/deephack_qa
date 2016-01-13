import argparse
import random

import pandas as pd
import subprocess
from sklearn.ensemble import RandomForestClassifier


GLOVE_TRAIN_FEATURES_FILE = 'glove_result_train.csv'
WIKI_TRAIN_FEATURES_FILE = 'wiki_result_train.csv'
GLOVE_TEST_FEATURES_FILE = 'glove_result_test.csv'
WIKI_TEST_FEATURES_FILE = 'wiki_result_test.csv'


def call_glove(in_data_file, in_result_file):
    subprocess.call([
            'python',
            'glove_predict.py',
            '--fname',
            in_data_file,
            '--result_file',
            in_result_file
    ])


def call_wiki(in_data_file, in_result_file):
    subprocess.call([
            'python',
            'ck12_wiki_predict.py',
            '--fname',
            in_data_file,
            '--result_file',
            in_result_file
        ])


def make_training_data(in_glove_file, in_wiki_file, in_train_file):
    train_data = pd.read_csv(in_train_file, sep='\t')
    answers = train_data['correctAnswer']
    glove_data = pd.read_csv(in_glove_file, sep=',')
    wiki_data = pd.read_csv(in_wiki_file, sep=',')

    regression_training_data = pd.DataFrame({
        'gloveWeightA': glove_data['weightA'],
        'gloveWeightB': glove_data['weightB'],
        'gloveWeightC': glove_data['weightC'],
        'gloveWeightD': glove_data['weightD'],
        'wikiWeightA': wiki_data['weightA'],
        'wikiWeightB': wiki_data['weightB'],
        'wikiWeightC': wiki_data['weightC'],
        'wikiWeightD': wiki_data['weightD'],
        'correctAnswer': answers
    })

    return regression_training_data


def train_log_regression(in_train_file):
    training_data = make_training_data(GLOVE_TRAIN_FEATURES_FILE, WIKI_TRAIN_FEATURES_FILE, in_train_file)
    model = RandomForestClassifier(n_estimators=128)
    model.fit(
            training_data[['gloveWeightA', 'gloveWeightB', 'gloveWeightC', 'gloveWeightD', 'wikiWeightA', 'wikiWeightB', 'wikiWeightC', 'wikiWeightD']],
            training_data['correctAnswer']
    )
    test_res = predict(model, GLOVE_TRAIN_FEATURES_FILE, WIKI_TRAIN_FEATURES_FILE)
    train_dataframe = pd.read_csv(in_train_file, sep='\t')
    print 'quality on train: ', sum(map(lambda x: x[0] == x[1], zip(test_res, train_dataframe['correctAnswer']))) / float(len(test_res))
    return model


def predict(
        in_classifier,
        in_glove_features=GLOVE_TEST_FEATURES_FILE,
        in_wiki_features=WIKI_TEST_FEATURES_FILE,
):
    glove_features = pd.read_csv(GLOVE_TEST_FEATURES_FILE, sep=',')
    wiki_features = pd.read_csv(WIKI_TEST_FEATURES_FILE, sep=',')
    all_features = pd.DataFrame({
        'gloveWeightA': glove_features['weightA'],
        'gloveWeightB': glove_features['weightB'],
        'gloveWeightC': glove_features['weightC'],
        'gloveWeightD': glove_features['weightD'],
        'wikiWeightA': wiki_features['weightA'],
        'wikiWeightB': wiki_features['weightB'],
        'wikiWeightC': wiki_features['weightC'],
        'wikiWeightD': wiki_features['weightD']
    })
    result = []
    for i in range(glove_features.shape[0]):
        cls = in_classifier.predict(all_features[['gloveWeightA', 'gloveWeightB', 'gloveWeightC', 'gloveWeightD', 'wikiWeightA', 'wikiWeightB', 'wikiWeightC', 'wikiWeightD']][i: i+1])
        result.append(cls[0])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='training_set.tsv', help='file name with data')
    parser.add_argument('--test_file', type=str, default='validation_set.tsv', help='file name with data')
    parser.add_argument('--N', type=int, default=300, help='embeding size (50, 100, 200, 300 only)')
    parser.add_argument('--result_file', type=str, default='regression_result.csv')
    parser.add_argument('--recalc_features', type=bool, default=False)
    args = parser.parse_args()

    if args.recalc_features:
        call_glove(args.train_file, GLOVE_TRAIN_FEATURES_FILE)
        call_wiki(args.train_file, WIKI_TRAIN_FEATURES_FILE)
        call_glove(args.test_file, GLOVE_TEST_FEATURES_FILE)
        call_wiki(args.test_file, WIKI_TEST_FEATURES_FILE)

    classifier = train_log_regression('data/' + args.train_file)
    #predict
    res = predict(classifier, 'data/' + args.test_file)
    data = pd.read_csv('data/' + args.test_file, sep='\t')
    #save result
    pd.DataFrame({
        'id': list(data['id']),
        'correctAnswer': res
    })[['id', 'correctAnswer']].to_csv("regression_result.csv", index=False)
