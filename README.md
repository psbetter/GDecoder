# GDecoder

Code for paper titled 'GDecoder: Generative Decoder for Zero-Shot Object Counting', Feature space-based and latent space-based versions of GDecoder will be released gradually.

## Preparation

### Environment
```
pip install requirement.txt
```
### Data & pretrained model

The three dataset could be downloaded at: [FSC-147](https://github.com/cvlab-stonybrook/LearningToCountEverything) | [REC-8K](https://github.com/sydai/referring-expression-counting) | [CARPK](https://lafi.github.io/LPN/).

Pretrained model: [MAE](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base_full.pth) | [Text](https://huggingface.co/google-bert/bert-base-uncased/tree/main)

## Train & Eval

GDecoder with feature space
```
sh run.sh
```

## Others

coming soon
