#!/bin/bash

# Download the model/data.
python download_model_and_data.py

# Run the prediction script which runs the model on the data (a GPU really helps speed!)
python predict_clip_model.py val_instances_darpa_self_eval.json RN50x16_best_checkpoint_inference_only.pt --batch_size 64

# Run automatic evaluations on the predictions:
python run_auto_eval.py val_instances_darpa_self_eval.json RN50x16_best_checkpoint_inference_only~predictions/val_instances_darpa_self_eval_preds_image2text_cosine_dists.npy auto_metrics.json

# Run human correlation evaluations on the predictions:
python run_human_eval.py val_instances_darpa_self_eval.json RN50x16_best_checkpoint_inference_only~predictions/val_instances_darpa_self_eval_preds_image2text_cosine_dists.npy

# Run exploration script that outputs human/machine readable predictions:
python show_predictions.py val_instances_darpa_self_eval.json RN50x16_best_checkpoint_inference_only~predictions/val_instances_darpa_self_eval_preds_image2text_cosine_dists.npy top_10_predictions.tsv'
