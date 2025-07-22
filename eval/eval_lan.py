import pandas as pd
import requests
import sacrebleu
import time
import psutil
import mlflow
import GPUtil

# Load Excel
df = pd.read_excel("../textsamples.xlsx")

print(f"Loaded {len(df)} rows from Excel file.")
print("Columns:", df.columns.tolist())

# 🚀 Start MLflow Experiment
mlflow.set_experiment("Language_Translation_IndicTrans2")
mlflow.start_run(run_name="translation_eval_run")
mlflow.register_model(model_uri="ai4bharat/indictrans2-en-indic-dist-200M", name="IndicTrans2")

# Assuming columns: 'English', 'Kannada'
src_texts = df['Human_English'].tolist()
ref_texts = df['Human_Kannada'].tolist()
tgt_lang = df["Tgt_lang"].tolist()

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



for sentence, tgt in zip(src_texts, tgt_lang):
    payload['inputs'][0]['TgtLang'] = tgt  # Use target language from list
    payload['inputs'][0]['Text'] = sentence
    print(f"Translating: {sentence} to {tgt}")

    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().used / (1024 ** 2)  # in MB
    gpu = GPUtil.getGPUs()[0].load * 100 if GPUtil.getGPUs() else 0
    start_time = time.time()

    response = requests.post(api_url, json=payload)
    if response.status_code != 200:
        print(f"❌ Failed to translate: {response.status_code} - {response.text}")
        continue

    elapsed_time = time.time() - start_time
    mlflow.log_metric("Total_Time_Sec", elapsed_time)
    mlflow.log_param("Model", "IndicTrans2")
    #mlflow.log_param("Language_Pair", f"en-{tgt}")

    mlflow.log_metric("CPU_Usage_Percent", cpu)
    mlflow.log_metric("GPU_Usage_Percent", gpu)
    mlflow.log_metric("Memory_Used_MB", mem)

    translated = response.json().get('Text', '')
    print(f"Translated: {translated}")
    if translated:
        translated_texts.append(translated)
    
    

# Compute BLEU for each pair
scores = []
for pred, ref in zip(translated_texts, ref_texts):
    bleu = sacrebleu.corpus_bleu([pred], [[ref]])  # sacrebleu expects list of predictions and list of reference lists
    scores.append(bleu.score)

# Save or inspect results
df['Translated'] = translated_texts
df['BLEU_Score'] = scores

print(f"Average BLEU Score: {sum(scores)/len(scores):.2f}")

# Compute chrF++ score
scores = []
chrf = sacrebleu.metrics.CHRF(
    char_order=6,     # Character n-gram order
    word_order=2,     # Word n-gram order (chrF++ uses >0)
    beta=2,           # Recall weight
    lowercase=True   # Case-sensitive comparison
)
print("hello")
print (chrf.corpus_score(translated_texts, ref_texts))

# # Compute score
# for pred, ref in zip(translated_texts, ref_texts):
#     # Compute chrF++ score
#     score = chrf.score([pred], [[ref]])
#     scores.append(score)
#     print(f"chrF++ Score: {score.score:.2f}")

# # Save or inspect results
# df['chrf_Score'] = scores

# print(f"Average chrf Score: {sum(scores)/len(scores):.2f}")

# df.to_excel("evaluated_translations.xlsx", index=False)

from bert_score import score

# Compute BERTScore
P, R, F1 = score(translated_texts, ref_texts, lang="kn", rescale_with_baseline=True)

# Print F1 score (semantic similarity)
print(f"BERTScore F1: {F1.mean():.4f}")

# Compute BLEU for each pair
scores = []
lang_pairs = {"kan_Knda": "Kannada", "hin_Deva": "Hindi", "tam_Taml": "Tamil", "tel_Telu": "Telugu", "ben_Beng": "Bengali"}

i = 0
for pred, ref, tgt in zip(translated_texts, ref_texts, tgt_lang):
    P, R, F1 = score([pred], [[ref]], lang=lang_pairs[tgt], rescale_with_baseline=True)
    # Print F1 score (semantic similarity)
    print(f"BERTScore F1: {F1.mean():.4f}")
    mlflow.log_metric(f"BERTScore_F1_Lang {lang_pairs[tgt]}", F1.mean().item(), step=i)
    i += 1

    
mlflow.end_run()

