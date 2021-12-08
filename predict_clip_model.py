'''
This script computes predictions for the sherlock dataset. A GPU is quite helpful.

'''
import argparse
import numpy as np
import torch
import json
import pprint
from PIL import Image, ImageDraw
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandomGrayscale, ColorJitter
import tempfile
import tqdm
import os
import collections
import clip
import sklearn.metrics
from scipy.stats import rankdata
import train_retrieval_clip


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('val')

    parser.add_argument('load_model_from',
                        default=str)

    parser.add_argument('--clip_model',
                        default='RN50x16',
                        choices=['ViT-B/32', 'RN50', 'RN101', 'RN50x4', 'ViT-B/16', 'RN50x16'])

    parser.add_argument('--batch_size',
                        default=64,
                        type=int)

    parser.add_argument(
        '--vcr_dir',
        default='self_eval_images',
        help='directory with all of the VCR image data, contains, e.g., movieclips_Lethal_Weapon')

    parser.add_argument(
        '--vg_dir',
        default='self_eval_images',
        help='directory with visual genome data, contains VG_100K and VG_100K_2')

    parser.add_argument('--hide_true_bbox',
                        type=int,
                        default=2)

    parser.add_argument('--workers_dataloader',
                        type=int,
                        default=0)

    parser.add_argument('--prompt_for_zero_shot',
                        type=str,
                        default='A photo of ') # for no highlight mode

    parser.add_argument('--widescreen_processing',
                        type=int,
                        help='if 1, then we will run CLIP twice over each image twice to get a bigger field of view',
                        default=1)

    parser.add_argument('--dont_compute_ranking',
                        type=int,
                        help='if 1, we will skip computing the rank metrics.',
                        default=1)

    args = parser.parse_args()
    args.output_predictions_path = '.'.join(args.load_model_from.split('.')[:-1]) + '~predictions'
    if not os.path.exists(args.output_predictions_path):
        os.makedirs(args.output_predictions_path)
    args.output_predictions_path  += '/{}_preds'.format(args.val.split('/')[-1].split('.')[0])

    if os.path.exists(args.output_predictions_path + "_image2text_cosine_dists.npy"):
        print(args.output_predictions_path + '_image2text_cosine_dists.npy already exists! skipping.')
        quit()

    if args.vcr_dir[-1] != '/':
        args.vcr_dir += '/'
    if args.vg_dir[-1] != '/':
        args.vg_dir += '/'
    return args


def main():
    args = parse_args()
    np.random.seed(1)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load(args.clip_model, jit=False)

    if 'zero_shot' not in args.load_model_from:
        print('Getting model weights from {}'.format(args.load_model_from))
        state = torch.load(args.load_model_from)
        state['model_state_dict'] = {k.replace('module.clip_model.', '') : v for k, v in state['model_state_dict'].items()}
        model.load_state_dict(state['model_state_dict'])

    try:
        args.input_resolution = model.visual.input_resolution
    except:
        args.input_resolution = model.input_resolution

    if args.hide_true_bbox != 4:
        model = train_retrieval_clip.CLIPExtractor(model, args)
    else:
        model = train_retrieval_clip.CLIPTextOnlyExtractor(model, args)

    model.to(args.device)
    model.eval()

    with open(args.val) as f:
        val = json.load(f)
        if 'zero_shot' in args.load_model_from:
            for v in val:
                v['targets']['inference'] = args.prompt_for_zero_shot + v['targets']['inference']
        val = torch.utils.data.DataLoader(
            train_retrieval_clip.CLIPDataset(val, args, training=False),
            batch_size=args.batch_size, num_workers=args.workers_dataloader, shuffle=False, worker_init_fn=worker_init_fn)

    loss_img = torch.nn.CrossEntropyLoss()
    loss_txt = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        bar = tqdm.tqdm(enumerate(val), total=len(val))
        all_val_im_embs, all_val_txt_embs, all_val_ids = [], [], []
        n, running_sum_loss = 0, 0
        ground_truth = torch.arange(args.batch_size, dtype=torch.long, device=args.device)
        insts = []
        for i, batch in bar:
            insts.extend([val.dataset.get(cid) for cid in list(batch['id'])])

            if args.hide_true_bbox != 4:
                images, captions = batch['image'].to(args.device), batch['caption'].to(args.device)
                image_features, text_features = model(images, captions)
            else:
                clues, inferences = batch['clue'].to(args.device), batch['caption'].to(args.device)
                image_features, text_features = model(clues, inferences)

            logit_scale = model.clip_model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            c_batch_size = logits_per_image.shape[0]
            total_loss = (loss_img(logits_per_image, ground_truth[:c_batch_size]) +
                          loss_txt(logits_per_text, ground_truth[:c_batch_size]))/2

            all_val_im_embs.append(image_features)
            all_val_txt_embs.append(text_features)
            all_val_ids.extend(list(batch['id']))

            n += 1
            running_sum_loss += total_loss.cpu().detach().numpy()
            bar.set_description('loss = {:.6f}'.format(running_sum_loss / n))

        all_val_im_embs = torch.cat(all_val_im_embs).cpu()
        all_val_txt_embs = torch.cat(all_val_txt_embs).cpu()

        im2text_dist = sklearn.metrics.pairwise_distances(all_val_im_embs,
                                                          all_val_txt_embs,
                                                          metric='cosine',
                                                          n_jobs=args.workers_dataloader)
        np.save(args.output_predictions_path + "_image2text_cosine_dists.npy", im2text_dist)

        if not args.dont_compute_ranking:
            im2text_ranks = np.diagonal(rankdata(im2text_dist, axis=0))
            text2im_ranks = np.diagonal(rankdata(im2text_dist, axis=1))
            print('im2text rank: {:.1f}, text2im rank: {:.1f}, size: {}'.format(
                np.mean(im2text_ranks),
                np.mean(text2im_ranks),
                len(text2im_ranks)))
        print('Final val loss: {:.5f}'.format(running_sum_loss / n))

if __name__ == '__main__':
    main()
