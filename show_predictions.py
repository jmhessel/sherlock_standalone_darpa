'''
Show/save some predictions of the model in both tsv and html format.

'''
import argparse
import train_retrieval_clip
import json
import numpy as np
import pprint
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata
import os
import tqdm
import subprocess

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('insts_in')
    parser.add_argument('dists_in')
    parser.add_argument('tsv_out_top_predictions')

    args = parser.parse_args()
    return args


def save_html(insts, im2txt, n=100, seed=1):
    fout = open('examples.html', 'w')
    for idx in tqdm.tqdm(np.random.choice(len(insts), size=n)):
        fout.write('<p><img src="{}" width=500></p>\n'.format('https://storage.googleapis.com/ai2-jack-public/sherlock_mturk/images_with_bboxes/{}.jpg'.format(insts[idx]['instance_id'])))
        fout.write('<p>true: {} (rank={}/{})</p>\n'.format(insts[idx]['targets']['inference'], scipy.stats.rankdata(im2txt[idx])[idx], im2txt.shape[1]))
        fout.write('<p>predicted</p>\n')
        fout.write('<ul>\n')
        for top_pred_idx in np.argsort(im2txt[idx])[:10]:
            fout.write('<li>{}</li>'.format(insts[top_pred_idx]['targets']['inference']))
        fout.write('</ul>\n')
        fout.write('<hr>\n')
    fout.close()


def main():
    args = parse_args()
    np.random.seed(1)
    
    with open(args.insts_in) as f:
        insts = json.load(f)

    im2txt = np.load(args.dists_in)

    # write tsv of top 10 predictions
    lines = []
    header = ['image_url'] + ['base_image_url'] + ['top_{}_pred'.format(idx) for idx in range(1, 11)] + ['top_{}_pred_id'.format(idx) for idx in range(1, 11)] + ['true_inference', 'true_inference_rank']
    lines.append('\t'.join(header))

    true_ranks = []
    for idx, inst in tqdm.tqdm(enumerate(insts)):
        image_url = 'https://storage.googleapis.com/ai2-jack-public/sherlock_mturk/images_with_bboxes/{}.jpg'.format(inst['instance_id'])
        base_image_url = inst['inputs']['image']['url']
        im2txt_top_idxs = np.argsort(im2txt[idx])
        ranks = rankdata(im2txt[idx])
        predicted_infs = [insts[top_idx]['targets']['inference'] for top_idx in im2txt_top_idxs[:10]]
        predicted_infs_ids = [insts[top_idx]['instance_id'] for top_idx in im2txt_top_idxs[:10]]
        true_inference = inst['targets']['inference']
        true_inference_rank = ranks[idx]
        lines.append('\t'.join([image_url, base_image_url] + predicted_infs + predicted_infs_ids + [true_inference] + [str(true_inference_rank)]))

    save_html(insts, im2txt)
        
    np.random.shuffle(lines)
    with open(args.tsv_out_top_predictions, 'w') as f:
        f.write('\n'.join(lines))
        

if __name__ == '__main__':
    main()
