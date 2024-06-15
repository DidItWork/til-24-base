# Backstreet Boys TIL Repo

## Automatic Speech Recognition

[Whisper-tiny](https://huggingface.co/openai/whisper-tiny) was finetuned to translate the audio sequence into text. The finetuning code can be found [here](finetune_asr.py).

## Natural Language Processing

An Encoder-Decoder Model made up of two BERTs (one for encoder and one for decoder) was finetuned to extract keywords from a text. Finetuned weights from a [Bert2Bert summary task](https://huggingface.co/patrickvonplaten/bert2bert-cnn_dailymail-fp16) were used to speed up the finetune process and fix performance issues related to the lack of examples for training.

## Vision Language Model

OWLv2 was finetuned with guidance from sentence encoder [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) to perform open-set vocabulary object detection. The finetuning code can be found [here](owl_vit_fine_tuning.ipynb) and is edited based on resources taken from [here](https://colab.research.google.com/drive/1y8q0sYR7ZX2puXM2Btg6YDoqcf415kCi?usp=sharing).