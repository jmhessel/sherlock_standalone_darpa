'''
Computes the metrics for the table in the paper

see run_compute_metrics.sh

'''
import argparse
import json
import numpy as np
import pprint
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata
import generation_eval_utils
import os
import tqdm
import subprocess
sns.set()
import statsmodels.stats.api as sms
import collections
import hashlib


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('insts_in')
    parser.add_argument('dists_in')
    
    parser.add_argument('output_json')
    
    parser.add_argument('--force',
                        help='if 1, force computation of metrics even if json exists',
                        default=0,
                        type=int)
    
    args = parser.parse_args()

    if not args.force and os.path.exists(args.output_json):
        print('{} already done.'.format(args.output_json))
        quit()

    return args


def main():
    args = parse_args()
    np.random.seed(1)

    with open(args.insts_in) as f:
        insts = json.load(f)

    im2txt = np.load(args.dists_in)

    # we will just run with split 0 to avoid downloading 20GB+ of images
    all_splits = [0]
    
    run_hash = hashlib.sha3_512()
    im2txts, txt2ims, p_at_1_im2txt, diversity_at_1, diversity_at_5, generation_metrics = [], [], [], [], [], collections.defaultdict(list)
    for split_idx in tqdm.tqdm(all_splits):
        valid_idxs = np.array([idx for idx, inst in enumerate(insts) if inst['split_idx'] == split_idx])
        valid_insts = [insts[idx] for idx in valid_idxs]

        all_ids_sorted = list(sorted([v['instance_id'] for v in valid_insts]))
        run_hash.update(json.dumps(all_ids_sorted).encode('utf-8'))
        assert len(all_ids_sorted) == len(set(all_ids_sorted))
        
        im2txt_split = im2txt[valid_idxs][:, valid_idxs]

        full_im2txt_ranks = rankdata(im2txt_split, axis=1, method='ordinal')
        ever_pred_in_top = set(full_im2txt_ranks[:, :1].flatten())
        ever_pred_in_top_5 = set(full_im2txt_ranks[:, :5].flatten())
        possible = set(full_im2txt_ranks[0])
        diversity_at_1.append(100 * len(ever_pred_in_top) / len(possible))
        diversity_at_5.append(100 * len(ever_pred_in_top_5) / len(possible))
        im2text_ranks = np.diagonal(rankdata(im2txt_split, axis=1))
        text2im_ranks = np.diagonal(rankdata(im2txt_split, axis=0))
        im2txts.append(float(np.mean(im2text_ranks)))
        txt2ims.append(float(np.mean(text2im_ranks)))
        p_at_1_im2txt.append(float(100*np.mean(im2text_ranks == 1.0)))
        
        # dont let it pick itself for this eval
        np.fill_diagonal(im2txt_split, np.inf)
        preds = [valid_insts[idx]['targets']['inference'] for idx in np.argmin(im2txt_split, axis=1)]
        refs = [[valid_insts[idx]['targets']['inference']] for idx in range(len(im2txt_split))]

        metrics, bs_hash = generation_eval_utils.get_bleu_and_bertscore_metrics(refs, preds)
        for k, v in metrics.items():
            generation_metrics[k+ '_hardmode'].append(float(v))

    print('im2txt: {:.2f} (lower better, UNITER-Large baseline=72.6)'.format(np.mean(im2txts)))
    print('txt2im: {:.2f} (lower better, UNITER-Large baseline=64.7)'.format(np.mean(txt2ims)))
    print('im2txt p@1: {:.2f} (higher better, UNITER-Large baseline=11.7)'.format(np.mean(p_at_1_im2txt)))
    k = 'robertascoref_hardmode'
    print(k + ': {:.2f} (higher better, UNITER-Large baseline=18.7)'.format(100*np.mean(generation_metrics[k])))
    print('INSTANCES IDENTIFIER:')
    digest = run_hash.hexdigest()
    print(digest)
    print('digest should be:')
    print('00ef77c5ffc358a17ebde21fe0400e3d9454e4ab19fc3a1747fdf460f9b9050ecb2cc9ea65f93ed5f2792dad64ac5a9ca56f060a1f60a94697737b6c96415a49')
    with open(args.output_json, 'w') as f:
        f.write(json.dumps(
            {'im2txt': im2txts,
             'txt2im': txt2ims,
             'diversity_at_1': diversity_at_1,
             'p_at_1': p_at_1_im2txt,
             'generation_metrics': generation_metrics,
             'bs_hash': bs_hash,
             'args': vars(args),
             'digest': digest}))


if __name__ == '__main__':
    main()
