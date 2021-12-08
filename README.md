# What's in here?

This repo contains a standalone version of our sherlock data/models for self-evaluation.


## How to run the code?

1. Download the model/data. `python download_model_and_data.py`
2. Run the prediction script which runs the model on the data (a GPU really helps speed!) `python predict_clip_model.py val_instances_darpa_self_eval.json RN50x16_best_checkpoint_inference_only.pt --batch_size 64`
3. Run automatic evaluations on the predictions: `python run_auto_eval.py val_instances_darpa_self_eval.json RN50x16_best_checkpoint_inference_only~predictions/val_instances_darpa_self_eval_preds_image2text_cosine_dists.npy auto_metrics.json`
4. Run human correlation results `python run_human_eval.py val_instances_darpa_self_eval.json RN50x16_best_checkpoint_inference_only~predictions/val_instances_darpa_self_eval_preds_image2text_cosine_dists.npy`
5. Run exploration script that outputs human readable predictions.