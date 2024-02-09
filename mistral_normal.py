import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
      # Change to "cuda" if available, else "cpu"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device=0 if torch.cuda.is_available() else -1,  # Change to 0 if cuda is available, else -1
)

prompt = "Can you explain the concept of regularization in machine learning?"

sequences = pipe(
    prompt,
    max_length=100,  # Change to max_length instead of max_new_tokens
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    num_return_sequences=1,
)

print(sequences[0]['generated_text'])
