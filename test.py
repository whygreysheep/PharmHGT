from transformers import AutoModel, AutoTokenizer
 
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
 
inputs = tokenizer("A new method for protein structure prediction.", return_tensors="pt")
outputs = model(**inputs)

from transformers import BertTokenizer, BertForQuestionAnswering

model_name = 'allenai/scibert_scivocab_uncased'  # 这是一个已经微调过的模型
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

question = "What is the capital of France?"
context = "Paris is the capital and most populous city of France."

inputs = tokenizer.encode_plus(question, context, return_tensors="pt", add_special_tokens=True, max_length=512, truncation=True)

outputs = model(**inputs)

import torch
answer_start = torch.argmax(outputs.start_logits)
answer_end = torch.argmax(outputs.end_logits) + 1

answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))