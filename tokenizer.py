from transformers import GPT2Tokenizer
import json

# Load tokenizer with special tokens if needed
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Optional: Add special tokens if you're building a chat model
# special_tokens_dict = {'additional_special_tokens': ['<USER>', '<ASSISTANT>']}
# tokenizer.add_special_tokens(special_tokens_dict)

def tokenize_data(input_file, output_file):
    with open(input_file, "r") as f:
        # More memory efficient line-by-line reading
        data = [json.loads(line) for line in f]

    tokenized_data = []
    for entry in data:
        try:
            prompt = entry["prompt"]  # Note: Typo here ('prompt' vs 'prompt')
            response = entry["response"]

            # Better approach: tokenize separately then combine
            tokenized_prompt = tokenizer(prompt, truncation=True, max_length=256)  # Reserve space for response
            tokenized_response = tokenizer(response, truncation=True, max_length=256)

            # Combine with attention mask
            input_ids = tokenized_prompt.input_ids + tokenized_response.input_ids
            attention_mask = tokenized_prompt.attention_mask + tokenized_response.attention_mask

            # Pad if needed
            if len(input_ids) < 512:
                padding_length = 512 - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
            else:
                input_ids = input_ids[:512]
                attention_mask = attention_mask[:512]

            tokenized_data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                # You might want to add labels for LM training
                "labels": input_ids.copy()  # For causal LM
            })

        except KeyError as e:
            print(f"Missing key in data: {e}")
            continue

    with open(output_file, "w") as f:
        json.dump(tokenized_data, f, ensure_ascii=False, indent=4)

tokenize_data("synthetic_econ_data.jsonl", "tokenized_data.json")
