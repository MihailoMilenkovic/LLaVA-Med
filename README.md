# llava-medical-finetuning

Finetuning and evaluation of LLaVA-Med on the VQA-RAD dataset

## Training

* Setup
```sh
pip install -e .
```

* Train
```sh
./scripts/vqa_rad_data_preprocessing.sh
./scripts/train_vq_rad.sh
```

## Evaluation
```sh
./scripts/eval_vq_rad.sh
```