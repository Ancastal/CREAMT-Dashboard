import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import login
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import torch
from trl import SFTTrainer


login(token="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")  # same as before
print("Logged in. Let's go!")
en_de = pd.read_csv('en-de.csv')
en_de = en_de.iloc[:, :2]
en_de.columns = ["en", "de"]
en_fr = pd.read_csv('en-fr.csv')
en_fr = en_fr.iloc[:, :2]
en_fr.columns = ["en", "fr"]
en_it = pd.read_csv('en-it.csv')
en_it = en_it.iloc[:, :2]
en_it.columns = ["en", "it"]
print("Data loaded.")


def choose_prompt_with_target(
    template, src, target, src_lang="English", tgt_lang="Italian"
):
    if template == "ZHANG":
        return f"{src_lang}: {src}\n{tgt_lang}: {target}"
    elif template == "GAO":
        return (f"This is a {src_lang} to {tgt_lang} translation, please provide the {tgt_lang} translation for this sentence: {src}\n{tgt_lang}: {target}")
    elif template == "JIAO":
        return f"Please provide the {tgt_lang} translation for this sentence: {src}\n{tgt_lang}: {target}"
    elif template == "NEW":
        return f"Translate the following {src_lang} sentence into {tgt_lang}:\n {src_lang}{src}\n{tgt_lang}: {target}"
    raise Exception("Choose a valid prompt template. [ZHANG, GAO, JIAO]")


prompts = []

for source, target in en_de[["en", "de"]].values:
    prompts.append(
        choose_prompt_with_target(
            "ZHANG", source, target, src_lang="English", tgt_lang="German"
        )
    )
for source, target in en_fr[["en", "fr"]].values:
    prompts.append(
        choose_prompt_with_target(
            "ZHANG", source, target, src_lang="English", tgt_lang="French"
        )
    )
for source, target in en_it[["en", "it"]].values:
    prompts.append(
        choose_prompt_with_target(
            "ZHANG", source, target, src_lang="English", tgt_lang="Italian"
        )
    )

print("Prompts created.")

random.shuffle(prompts)

dataset = DatasetDict(
    {
        "train": Dataset.from_dict({"text": prompts[: int(len(prompts) * 0.80)]}),
        "validation": Dataset.from_dict({"text": prompts[int(len(prompts) * 0.80):]}),
    }
)
print("Dataset created.")

# Define the cache directory for storing model/tokenizer data.
cache_dir = "mistral_cache"

# Specify the model name.
model_name = "mistralai/Mistral-7B-v0.1"

# Configure the model for 4-bit quantization using the BitsAndBytesConfig.
# This is particularly useful for reducing model size and potentially increasing inference speed.
nf4_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load the pre-trained causal language model with the specified quantization configuration.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=nf4_quant_config,
    use_cache=False,
    cache_dir=cache_dir,
)

# Load the tokenizer associated with the model, configuring it for specific use cases.
tokenizer = AutoTokenizer.from_pretrained(
    model_name, cache_dir=cache_dir, add_bos_token=True, add_eos_token=False
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


peft_config = LoraConfig(
    lora_alpha=16, lora_dropout=0.1, r=64, bias="none", task_type="CAUSAL_LM"
)


model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
print("Model prepared for training.")

output_directory = "clean_creamt_mistral_lora"

training_args = TrainingArguments(
    output_dir=output_directory,
    num_train_epochs=1,
    # max_steps=594,  # comment out to train in epochs
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=0.03,
    logging_steps=20,
    save_strategy="epoch",
    # evaluation_strategy="epoch",
    evaluation_strategy="steps",
    eval_steps=50,  # comment out this line if you want to evaluate at the end of each epoch
    # eval_accumulation_steps=4,
    learning_rate=2e-3,  # 2e-4 # lower LE for smaller batch sizes
    bf16=True,
    lr_scheduler_type="constant",
)


max_seq_length = 1024  # increase if needed

trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    dataset_text_field="text",
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    # compute_metrics=compute_metrics,
)
print("Trainer created.")
trainer.train()
