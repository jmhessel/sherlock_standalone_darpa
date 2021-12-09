'''
This script downloads the best model and the data needed to run the analytic.

python download_model_and_data.py

to clean temporary data files you can run:

rm *.zip ; rm im2txt_annotations_all_v2.json; rm -rf self_eval_images/; rm val_instances_darpa_self_eval.json; rm *.pt; rm -rf *~predictions; rm auto_metrics.json; rm examples.html; rm top_10_predictions.tsv; rm -rf __pycache__/;
'''
import argparse
import subprocess
import os

def parse_args():
    parser = argparse.ArgumentParser()

    return parser.parse_args()


def call(x):
    subprocess.call(x, shell=True)
    

def main():
    args = parse_args()

    for url in [
            'https://storage.googleapis.com/ai2-jack-public/sherlock_darpa_selfeval/RN50x16_best_checkpoint_inference_only.pt',
            'https://storage.googleapis.com/ai2-jack-public/sherlock_darpa_selfeval/im2txt_annotations_all_v2.zip',
            'https://storage.googleapis.com/ai2-jack-public/sherlock_darpa_selfeval/self_eval_images.zip',
            'https://storage.googleapis.com/ai2-jack-public/sherlock_darpa_selfeval/val_instances_darpa_self_eval.json.zip']:
        if not os.path.exists(url.split('/')[-1]):
            call('wget {}'.format(url))

    if not os.path.exists('im2txt_annotations_all_v2.json'):
        call('unzip im2txt_annotations_all_v2.zip')
    if not os.path.exists('self_eval_images/VG_100K_2/2412277.jpg'):
        call('unzip self_eval_images.zip')
    if not os.path.exists('val_instances_darpa_self_eval.json'):
        call('unzip val_instances_darpa_self_eval.json.zip')
    
    
if __name__ == '__main__':
    main()
