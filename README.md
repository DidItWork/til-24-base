# Backstreet Boys TIL Repo

## Automatic Speech Recognition

[Whisper-tiny](https://huggingface.co/openai/whisper-tiny) was finetuned to translate the audio sequence into text. The finetuning code can be found [here](finetune_asr.py).

## Natural Language Processing

An Encoder-Decoder Model made up of two [BERTs](https://huggingface.co/docs/transformers/model_doc/bert) (one for encoder and one for decoder) was finetuned to extract keywords from a text. Finetuned weights from a [Bert2Bert summary task](https://huggingface.co/patrickvonplaten/bert2bert-cnn_dailymail-fp16) were used to speed up the finetune process and fix performance issues related to the lack of examples for training. The finetuning code can be found [here](finetune_nlp.py)

## Vision Language Model

An end-to-end Vision Language Model - [OWLv2](https://huggingface.co/google/owlv2-base-patch16) - was finetuned with guidance from a sentence encoder [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) to perform open-set vocabulary object detection. The finetuning code can be found [here](owl_vit_fine_tuning.ipynb) and is edited based on resources taken from [here](https://colab.research.google.com/drive/1y8q0sYR7ZX2puXM2Btg6YDoqcf415kCi?usp=sharing).

Formated training and testing data in a 0.75-0.25 split are also provided in [train_caption_vlm.jsonl](train_caption_vlm.jsonl) and [test_caption_vlm.jsonl](test_caption_vlm.jsonl) respectively. Data can also be formated from the original `vlm.jsonl` with [dataset_converter.py](dataset_converter.py)