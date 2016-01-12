import argparse

import operator
import random

import pandas as pd
import subprocess
from sklearn.linear_model import LogisticRegression


def make_training_data(in_glove_file, in_wiki_file, in_train_file):
    train_data = pd.read_csv(in_train_file, sep='\t')
    answers = train_data['correctAnswer']
    glove_data = pd.read_csv(in_glove_file, sep=',')
    wiki_data = pd.read_csv(in_wiki_file, sep=',')

    regression_training_data = pd.DataFrame({
        'glove_weight': pd.concat([glove_data['weightA'], glove_data['weightB'], glove_data['weightC'], glove_data['weightD']),
        'wiki_weight': pd.concat([wiki_data['weightA'], wiki_data['weightB'], wiki_data['weightC'], wiki_data['weightD']]),
        'correct': map(lambda ans: ans == 'A', answers) + map(lambda ans: ans == 'B', answers) + map(lambda ans: ans == 'C', answers) + map(lambda ans: ans == 'D', answers)
    })

    return regression_training_data


def train_log_regression(in_features, in_answers):
    model = LogisticRegression()
    model.fit(in_features, in_answers)
    return model


def predict(in_classifier, in_test_file):
    subprocess.call([
        'python',
        'glove_predict.py',
        '--fname',
        in_test_file,
        '--result_file',
        'glove_test_result.csv'
    ])
    subprocess.call([
        'python',
        'ck12_wiki_predict.py',
        '--fname',
        in_test_file,
        '--result_file',
        'wiki_test_result.csv'
    ])

    glove_features = pd.read_csv('glove_test_result.csv', sep=',')
    wiki_features = pd.read_csv('glove_test_result.csv', sep=',')
    result = []
    for i in range(glove_features.shape[0]):
        answer_classes = [
            (in_classifier.predict([glove_features[answer][i], wiki_features[answer][i]]))
            for answer in ['answerA', 'answerB', 'answerC', 'answerD']
        ]
        answer_weights = [
            (in_classifier.predict_proba([glove_features[answer][i], wiki_features[answer][i]]))
            for answer in ['answerA', 'answerB', 'answerC', 'answerD']
        ]
        if True in answer_classes:
            candidate_answers = zip(answer_classes, answer_weights, ['A', 'B', 'C', 'D'])
            candidate_answers = filter(lambda ans: ans[0] == True, candidate_answers)
            best_answer = sorted(candidate_answers, key=operator.itemgetter(1), reverse=True)[0]
            result.append(best_answer[2])
        else:
            result.append(random.choice(['A', 'B', 'C', 'D']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='training_set.tsv', help='file name with data')
    parser.add_argument('--test_file', type=str, default='validation_set.tsv', help='file name with data')
    parser.add_argument('--N', type=int, default= 300, help='embeding size (50, 100, 200, 300 only)')
    parser.add_argument('--result_file', type=str, default='regression_result.csv')
    args = parser.parse_args()

    subprocess.call([
        'python',
        'glove_predict.py',
        '--fname',
        args.train_file
    ])
    subprocess.call([
        'python',
        'ck12_wiki_predict.py',
        '--fname',
        args.train_file
    ])

    training_data = make_training_data('glove_result.csv', 'wiki_result.csv', 'data/' + args.train_file)
    classifier = train_log_regression(
        training_data[['glove_weight', 'wiki_weight']],
        training_data[['correct']]
    )
    #predict
    res = predict(classifier, 'data/' + args.test_file)
    data = pd.read_csv('data/' + args.test_file, sep='\t')
    #save result
    pd.DataFrame({
        'id': list(data['id']),
        'correctAnswer': res
    })[['id', 'correctAnswer']].to_csv("regression_result.csv", index=False)
