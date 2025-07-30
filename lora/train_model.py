from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from torch.optim import AdamW
from lora import model
from datasets import load_dataset
from lora import model, tokenizer, tokenized_train_dataset, tokenized_eval_dataset

#parameters
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
#    eval_steps=20,
#    save_steps=20,
    save_total_limit=2,
    fp16=False,  # Must be False on Mac (no CUDA)
    report_to="none",  
    no_cuda=True,                   
)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    tokenizer=tokenizer,
    eval_dataset=tokenized_eval_dataset,
    optimizers=(optimizer, None),  # (optimizer, scheduler)    #label_names=["labels"]

)
trainer.train()
print("——--------train finished")