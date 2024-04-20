# llava-medical-finetuning

Finetuning and evaluation of LLaVA-Med on the VQA-RAD dataset

## Training

* Setup

Note: all of the scripts above should set the appropriate values for the following variables:
```sh
weights_path_og # path to base llava model weights
weights_path_finetuned # path to finetuned model weights
```

```sh
pip install -e .
#create the llava model from pretrained models+llava weight delta
./scripts/apply_llava_delta.sh
```

* Train
```sh
#create train+test datasets
./scripts/vqa_rad_data_preprocessing.sh
#run training scripts
./scripts/train_vqa_rad.sh
```

## Evaluation
```sh
#run evals (assuming test dataset is created)
./scripts/eval_vqa_rad.sh
```