'''
Correlate the model's predictions with human judgments.
'''
import argparse
import json
import numpy as np
import scipy.stats
import pprint
import os
import pickle
import collections
import sklearn.metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('instances')
    parser.add_argument('predictions')

    args = parser.parse_args()

    if not os.path.exists('im2txt_annotations_all_v2.json'):
        print('please run download.py')
        quit()
    args.annotation_json = 'im2txt_annotations_all_v2.json'

    return args


def pairwise_acc(model_pred, label):
    #noise tiebreak vector
    np.random.seed(1)
    tiebreak_preds = np.random.random(size=10)/1E9
    tiebroken_preds = model_pred + tiebreak_preds[:len(model_pred)]
    correct, total = 0, 0
    for idx1 in range(len(label)):
        for idx2 in range(idx1 + 1, len(label)):
            if label[idx1] == label[idx2]: continue
            total += 1
            correct += int((label[idx1] < label[idx2]) == (tiebroken_preds[idx1] < tiebroken_preds[idx2]))

    if total > 0 :
        return (correct / total - .5) * 2
    else:
        return 0.0


def main():
    args = parse_args()

    with open(args.instances) as f:
        original_instances = json.load(f)

    if '.json' in args.annotation_json:
        with open(args.annotation_json) as f:
            annotations = json.load(f)
    else:
        with open(args.annotation_json, 'rb') as f:
            annotations = pickle.load(f)
        source2cand2uniter_score = {}
        for a in annotations:
            for c in a['candidates']:
                source2cand2uniter_score[(a['Input_iid'], c['source_iid'])] = c['uniter_score']

    if args.predictions:
        #cosine sims
        preds = -np.load(args.predictions) + 1
    else:
        preds = None

    eval_fn = pairwise_acc
    instance_id2idx = {o['instance_id']: idx for idx, o in enumerate(original_instances)}
    instance_id2instance = {o['instance_id']: o for o in original_instances}

    spearmans = {'human_corr': [], 'model_corr': [], 'random_corr': [], 'idxs_in_file': []}
    for a in annotations:
        if a['Input_iid'] not in instance_id2idx: continue

        source_idx = instance_id2idx[a['Input_iid']]
        cand_idxs = np.array([instance_id2idx[c['source_iid']] for c in a['candidates']])
        if preds is not None:
            model_preds = [preds[source_idx][c_idx] for c_idx in cand_idxs]
        else:
            model_preds = [source2cand2uniter_score[(a['Input_iid'], c['source_iid'])] for c in a['candidates']]

        ann1_preds = np.array([float(c['annot1']) for c in a['candidates']])
        ann2_preds = np.array([float(c['annot2']) for c in a['candidates']])

        spearman_human, spearman_model, spearman_random = [], [], []


        human_and_label = [(ann1_preds, ann2_preds),
                           (ann2_preds, ann1_preds)]

        for human_pred, label in human_and_label:
            spearman_human.append(eval_fn(human_pred, label))
            spearman_model.append(eval_fn(model_preds, label))
            spearman_random.append(eval_fn(np.random.random(label.shape), label))

        spearmans['human_corr'].append(np.mean(spearman_human))
        spearmans['model_corr'].append(np.mean(spearman_model))
        spearmans['random_corr'].append(np.mean(spearman_random))
        spearmans['idxs_in_file'].append(a['line_id'])

    print('Human Accuracy over Likert Judgments:')
    print('~~~')
    print('Human Agreement (upper bound): {:.2f}'.format(np.mean(spearmans['human_corr'])*100))
    print('Our Model corr: {:.2f}'.format(np.mean(spearmans['model_corr'])*100, len(spearmans['human_corr'])))
    print('UNITER-Large Baseline: {:.1f}'.format(18.7))
    print('Random corr (lower bound): {:.2f}'.format(np.mean(spearmans['random_corr'])*100, len(spearmans['random_corr'])))




if __name__ == '__main__':
    main()
