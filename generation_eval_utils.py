'''
Automatic generation evaluation metrics.

The most useful function here is

get_all_metrics(refs, cands)

'''
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
import argparse
import os
import torch
import numpy as np
import tqdm
import pprint
import json
from bert_score import score as run_bert_score


def bert_score(refs, cands):
    refs, cands = tokenize(refs, cands)

    ordered_ks = list(refs.keys())

    refs_as_list = []
    cands_as_list = []

    for k in ordered_ks:
        cands_as_list.append(cands[k][0])
        refs_as_list.append(refs[k])

    (P, R, F), bs_hash = run_bert_score(cands_as_list, refs_as_list, lang='en',
                                        verbose=True, batch_size=128, return_hash=True,
                                        rescale_with_baseline=True)
    P, R, F = P.cpu().numpy(), R.cpu().numpy(), F.cpu().numpy()
    return P, R, F, bs_hash


def get_bleu_and_bertscore_metrics(refs, cands):
    metrics = []
    names = []

    pycoco_eval_cap_scorers = [(Bleu(4), 'bleu'),
                               (Cider(), 'cider'),
                               (Meteor(), 'meteor')]
    
    for scorer, name in pycoco_eval_cap_scorers:
        overall, per_cap = pycoco_eval(scorer, refs, cands)
        if name == 'bleu':
            metrics.append(overall[-1])
        else:
            metrics.append(overall)
        names.append(name)

    P, R, F, bs_hash = bert_score(refs, cands)
    names.append('robertascoref')
    metrics.append(np.mean(F))
    
    metrics = dict(zip(names, metrics))
    return metrics, bs_hash


def get_all_metrics(refs, cands, lowercase=False):
    metrics = []
    names = []

    if lowercase:
        refs = [[r.lower() for r in ref] for ref in refs]
        cands = [c.lower() for c in cands]

    pycoco_eval_cap_scorers = [(Bleu(4), 'bleu'),
                               (Meteor(), 'meteor'),
                               (Rouge(), 'rouge'),
                               (Cider(), 'cider'),
                               (Spice(), 'spice')]

    for scorer, name in pycoco_eval_cap_scorers:
        overall, per_cap = pycoco_eval(scorer, refs, cands)
        metrics.append(overall)
        names.append(name)

    metrics = dict(zip(names, metrics))
    return metrics


def tokenize(refs, cands, no_op=False):
    # no_op is a debug option to see how significantly not using the PTB tokenizer
    # affects things
    tokenizer = PTBTokenizer()

    if no_op:
        refs = {idx: [r for r in c_refs] for idx, c_refs in enumerate(refs)}
        cands = {idx: [c] for idx, c in enumerate(cands)}

    else:
        refs = {idx: [{'caption':r} for r in c_refs] for idx, c_refs in enumerate(refs)}
        cands = {idx: [{'caption':c}] for idx, c in enumerate(cands)}

        refs = tokenizer.tokenize(refs)
        cands = tokenizer.tokenize(cands)

    return refs, cands


def pycoco_eval(scorer, refs, cands):
    '''
    scorer is assumed to have a compute_score function.
    refs is a list of lists of strings
    cands is a list of predictions
    '''
    refs, cands = tokenize(refs, cands)
    average_score, scores = scorer.compute_score(refs, cands)
    return average_score, scores
