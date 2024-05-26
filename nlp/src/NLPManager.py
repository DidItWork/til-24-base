from typing import Dict
import torch
from transformers import AutoTokenizer, BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel

device = "cuda" if torch.cuda.is_available() else "cpu"

class NLPManager:
    def __init__(self):
        # initialize the model here
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")

        # use BERT's cls token as BOS token and sep token as EOS token
        # encoder = BertGenerationEncoder.from_pretrained("google-bert/bert-base-cased", bos_token_id=101, eos_token_id=102)
        # # add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
        # decoder = BertGenerationDecoder.from_pretrained(
        #     "google-bert/bert-base-cased", add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102
        # )
        # self.bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)
        self.bert2bert = EncoderDecoderModel.from_pretrained("/home/benluo/til-24-base/nlp/weights/nlp-ft/checkpoint-1000").to(device)

        # bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("google-bert/bert-base-cased", "google-bert/bert-base-cased")

        # bert2bert = BertModel.from_pretrained("bert-base-cased")
        self.bert2bert.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.bert2bert.config.pad_token_id = self.tokenizer.pad_token_id
        
        self.bert2bert.eval()

    def qa(self, context: str) -> Dict[str, str]:
        # perform NLP question-answering
        with torch.no_grad():
            input_tokenized = self.tokenizer(context, padding="max_length", truncation=True, max_length=64, return_tensors="pt").to(device)
            outputs = self.bert2bert.generate(input_tokenized.input_ids.to(device), attention_mask=input_tokenized.attention_mask.to(device))
            output_str = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        print(context)
        print(output_str)

        try:
            tool, heading, target = output_str.split("|")
        except:
            heading = ""
            tool = ""
            target = ""

        return {"heading": heading.strip(), "tool": tool.strip(), "target": target.strip()}
