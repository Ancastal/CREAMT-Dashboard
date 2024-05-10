import os
from pathlib import Path

import evaluate
import pandas as pd
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

cache_dir = "mistral_cache"
output_directory = "clean_creamt_mistral_lora"

cache_dir = "mistral_cache"

# Specify the model name.
model_name = "mistralai/Mistral-7B-v0.1"

peft_model_path = Path(output_directory) / "checkpoint-211"

peftconfig = PeftConfig.from_pretrained(peft_model_path)

model_base = AutoModelForCausalLM.from_pretrained(
    peftconfig.base_model_name_or_path, device_map="auto", cache_dir=cache_dir
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    add_bos_token=True,
    add_eos_token=False,  # always False for inference
)

new_model = PeftModel.from_pretrained(model_base, peft_model_path)

print("Peft model loaded")


def generate_response(prompt, model):
    encoded_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    model_inputs = encoded_input.to("cuda")

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        min_new_tokens=1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded_output = tokenizer.batch_decode(generated_ids)

    return decoded_output[0].replace(prompt, "")


def generate_prompts(df, tgt_lang, prompt_template):
    prompts = []
    for _, row in df.iterrows():
        if prompt_template == "ZHANG23":
            prompt = f'English: {row["en"]}' + "\n" + f"{tgt_lang}:"
        prompts.append(prompt)
    return prompts


# Zero-shot prompting
test_fr = pd.read_csv("test_en-fr.csv")[:100]
test_fr = test_fr.iloc[:, :2]
test_fr.columns = ["en", "fr"]

prompts = generate_prompts(test_fr, "French", "ZHANG23")


results = []
for prompt in prompts:
    results.append(generate_response(prompt, new_model))

bleu = evaluate.load("bleu")
chrf = evaluate.load("chrf")
comet = evaluate.load("comet")

predictions = results
references = test_fr["fr"].tolist()

bleu_score = bleu.compute(predictions=predictions, references=references)["bleu"]
chrf_score = chrf.compute(predictions=predictions, references=references)["score"]
comet_score = comet.compute(
    predictions=predictions, references=references, sources=test_fr["en"].tolist()
)["mean_score"]

print("Results for EN-FR:")
print(f"BLEU: {bleu_score}")
print(f"CHRF: {chrf_score}")
print(f"COMET: {comet_score}")
