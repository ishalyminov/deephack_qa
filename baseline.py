import pandas as pd
import random
import sys


def get_answer_baseline(in_question, in_answers):
    return random.choice(in_answers)[0]


def process(in_data):
    result = pd.DataFrame({
        'id': in_data['id'],
        'correctAnswer': map(
            lambda q_a: get_answer_baseline(q_a[1][0], zip('ABCD', q_a[1][1:])),
            in_data[['question', 'answerA', 'answerB', 'answerC', 'answerD']].iterrows()
        )
    })
    return result


def do(in_stream, out_stream):
    data = pd.read_csv(in_stream, sep='\t')
    answers_data_frame = process(data)
    out_stream.write(answers_data_frame[['id', 'correctAnswer']].to_csv(
        sep=',',
        index=False
    ))


if __name__ == '__main__':
    do(sys.stdin, sys.stdout)
