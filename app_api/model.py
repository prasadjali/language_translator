

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
# recommended to run this on a gpu with flash_attn installed
# don't set attn_implemetation if you don't have flash_attn

#src_lang, tgt_lang = "eng_Latn", "hin_Deva"
#src_lang, tgt_lang = "eng_Latn", "kan_Knda"
#src_lang, tgt_lang = "eng_Latn", "tam_Taml"
#model_name = "ai4bharat/indictrans2-en-indic-1B"
#model_name = "ai4bharat/indictrans2-en-indic-dist-200M"

global model, tokenizer, ip

class MyModel:

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_ip(self):
        return self.ip

    def __init__(self, model_name):
        global model, tokenizer, ip

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16, # performance might slightly vary for bfloat16
            attn_implementation="flash_attention_2"
        ).to(DEVICE)

        self.ip = IndicProcessor(inference=True)

        return



    def translate(self, input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        batch = ip.preprocess_batch(
            input_sentences,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )

        # Tokenize the sentences and generate input encodings
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        generated_tokens = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Postprocess the translations, including entity replacement
        translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        #for input_sentence, translation in zip(input_sentences, translations):
        #    print(f"{src_lang}: {input_sentence}")
        #    print(f"{tgt_lang}: {translation}")
        
        return translations

model_name = "ai4bharat/indictrans2-en-indic-dist-200M"

mymodel = MyModel(model_name)