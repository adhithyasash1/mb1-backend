import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# Load the Llama model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)

# Load the JSON data
with open("resume_qa_pairs.json", "r") as f:
    qa_pairs = json.load(f)

# Convert Q&A pairs into text format
train_data = []
for pair in qa_pairs:
    prompt = f"Question: {pair['question']}\nAnswer: {pair['answer']}\n"
    train_data.append(prompt)

# Convert the text data into Hugging Face Dataset
train_dataset = Dataset.from_dict({"text": train_data})

# Tokenize the data
def tokenize_function(examples):
    encodings = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    encodings['labels'] = encodings['input_ids']  # Labels are the same as input_ids for causal language modeling
    return encodings

tokenized_dataset = train_dataset.map(tokenize_function, batched=True)

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type="CAUSAL_LM",
    inference_mode=False,
    lora_alpha=16,
    lora_dropout=0.05
)

# Wrap the model with LoRA
lora_model = get_peft_model(model, lora_config)

# Print trainable parameters
lora_model.print_trainable_parameters()

# Define training arguments
training_args = TrainingArguments(
    output_dir="./model",
    overwrite_output_dir=True,
    num_train_epochs=1,
    learning_rate=1e-4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    max_steps=50,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2
)

# Initialize the Trainer
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model with LoRA adapters
lora_model.save_pretrained("./model")
tokenizer.save_pretrained("./model")