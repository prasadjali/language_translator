import os
import sys
import glob
from tqdm import tqdm
from google.cloud import translate
import requests

# Expects a json file containing the API credentials.
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join( os.path.dirname(__file__), r"api_key.json")

flores_to_iso = {
    "asm_Beng": "as",
    "ben_Beng": "bn",
    "doi_Deva": "doi",
    "eng_Latn": "en",
    "gom_Deva": "gom",
    "guj_Gujr": "gu",
    "hin_Deva": "hi",
    "kan_Knda": "kn",
    "mai_Deva": "mai",
    "mal_Mlym": "ml",
    "mar_Deva": "mr",
    "mni_Mtei": "mni_Mtei",
    "npi_Deva": "ne",
    "ory_Orya": "or",
    "pan_Guru": "pa",
    "san_Deva": "sa",
    "sat_Olck": "sat",
    "snd_Arab": "sd",
    "tam_Taml": "ta",
    "tel_Telu": "te",
    "urd_Arab": "ur",
}


# Copy the project id from the json file containing API credentials
def translate_text(text, src_lang, tgt_lang):
    # Translate using REST API
    translated_texts = []
    api_url = "http://localhost:8001/api/v1/predict"  # replace with your actual API endpoint
    # Construct the payload according to your schema
    payload = {
        "inputs": [
            {
                "FarmerId": 123,               # Replace with actual ID or None
                "TgtLang": "hin_Deva",               # Replace with target language code or None
                "Text": ""
            }
        ]
    }

    payload['inputs'][0]['TgtLang'] = tgt_lang  # Use target language from list
    payload['inputs'][0]['Text'] = text
    print(f"Translating: to {tgt_lang}")

    response = requests.post(api_url, json=payload)
    if response.status_code != 200:
        print(f"❌ Failed to translate: {response.status_code} - {response.text}")
        return ""

    translated_text = response.json().get('Text', '')
    print(f"Translated: {translated_text}")
    if translated_text:
        translated_texts.append(translated_text)
        
    return translated_text


if __name__ == "__main__":
    root_dir = sys.argv[1]

    pairs = sorted(glob.glob(os.path.join(root_dir, "*")))

    for pair in pairs:

        print(pair)

        basename = os.path.basename(pair)


        src_lang, tgt_lang = basename.split("-")
        print(basename)
        print(f"source lang {src_lang}")
        print(f"tgt_lang {tgt_lang}")
        if src_lang not in flores_to_iso.keys() or tgt_lang not in flores_to_iso.keys():
            continue


        if src_lang == "eng_Latn":
            lang = tgt_lang
        else:
            lang = src_lang

        lang = flores_to_iso[lang]

        if lang not in "as bn doi gom gu hi kn mai ml mni_Mtei mr ne or pa sa sd ta te ur":
            continue

        print(f"{src_lang} - {tgt_lang}")

        # source to target translations

        src_infname = os.path.join(pair, f"test.{src_lang}")
        tgt_outfname = os.path.join(pair, f"test.{tgt_lang}.pred.indictrans2")
        if os.path.exists(src_infname) and not os.path.exists(tgt_outfname):
            src_sents = [
                sent.replace("\n", "").strip()
                for sent in open(src_infname, "r").read().split("\n")
                if sent
            ]
            translations = [
                translate_text(text, src_lang, tgt_lang).strip() for text in tqdm(src_sents)
            ]
            with open(tgt_outfname, "w") as f:
                f.write("\n".join(translations))

 
        print(f"hello {src_infname}")
