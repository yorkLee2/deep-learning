from transformers import AutoTokenizer, AutoModelForCausalLM
from data_prepare import samples
import json
import os
import torch

model_name = "/home/york/Documents/1.5b"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
#model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("cuda")

print("✅ Model loaded successfully.")

#prepare for dataset
with open("datasets.jsonl", "w", encoding="utf-8") as f:
    for s in samples:
        json_line = json.dumps(s, ensure_ascii=False)
        f.write(json_line + "\n")
    else:
        print("prepare data finished")

from datasets import load_dataset

os.environ["HF_DATASETS_CACHE"] = "/Users/yorkheyongchao/Documents/tmp/new_dataset_cache"

#Prepare training and testing sets
dataset = load_dataset(
    path="json",
    data_files={"train": "datasets.jsonl"},
    split="train",
    keep_in_memory=True        #not load old memory 
)
print("data size", len(dataset))



#Split the dataset into training and evaluation sets
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

print(f"train dataset len: {len(train_dataset)}")
print(f"test dataset len: {len(eval_dataset)}")
print("------ Finished preparing training data ------")


#Define a function to tokenize a batch of samples
def tokenizer_function(many_samples):
    texts = [f"{prompt}\n{completion}" for prompt, completion in zip(many_samples["prompt"], many_samples["completion"])]
    tokens = tokenizer(texts,max_length=512, padding="max_length", truncation=True, )
    tokens["labels"] = tokens["input_ids"].copy()

    return tokens

tokenized_train_dataset = train_dataset.map(tokenizer_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenizer_function, batched=True)

print("tokenized finished ")
print(tokenized_train_dataset[0])


#Quantization  only cuda
# from transformers import BitsAndBytesConfig
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)
# # Load the model with quantization and automatic device placement
# model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=quantization_config,device_map="auto")
# print("------ Finished loading the quantized model ------")


# Step 6: Configure LoRA fine-tuning
from peft import get_peft_model, LoraConfig, TaskType

# Define the LoRA configuration
lora_config = LoraConfig(
    r=8,                        # Rank of LoRA matrices
    lora_alpha=16,              # Scaling factor
    lora_dropout=0.05,          # Dropout rate for LoRA layers
    task_type=TaskType.CAUSAL_LM  # Specify the task type as causal language modeling
)

# Apply the LoRA configuration to the base model
model = get_peft_model(model, lora_config)

# Print which parameters are trainable (i.e., affected by LoRA)
model.print_trainable_parameters()

print("------ LoRA fine-tuning setup completed ------")
