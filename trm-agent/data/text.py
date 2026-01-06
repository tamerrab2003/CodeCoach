import json
import mlx.core as mx
import tiktoken
import numpy as np
from pathlib import Path

def load_custom_jsonl(data_path, batch_size, max_len=512):
    """
    Loads a JSONL file and yields batches of tokenized data.
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Load all data into memory (simplification for "Tiny" model)
    data = []
    with open(data_path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                # Combine input + reasoning + output for training
                # Format: Input: ... \n Reasoning: ... \n Output: ...
                full_text = f"Input: {obj.get('input', '')}\nReasoning: {obj.get('reasoning', '')}\nOutput: {obj.get('output', '')}"
                tokens = tokenizer.encode(full_text, allowed_special="all")
                if len(tokens) > max_len:
                    tokens = tokens[:max_len]
                else:
                    # Pad
                    tokens = tokens + [tokenizer.eot_token] * (max_len - len(tokens))
                data.append(tokens)
            except Exception:
                continue
                
    data = np.array(data)
    
    class CustomDataLoader:
        def __init__(self, data, batch_size):
            self.data = data
            self.batch_size = batch_size
            
        def __iter__(self):
            indices = np.random.permutation(len(self.data))
            for i in range(0, len(self.data), self.batch_size):
                batch_indices = indices[i : i + self.batch_size]
                if len(batch_indices) < self.batch_size:
                    continue # Skip incomplete batches
                batch_tokens = self.data[batch_indices]
                
                yield {
                    "input_ids": mx.array(batch_tokens),
                    "labels": mx.array(batch_tokens)
                }

        def reset(self):
            pass

    # Meta info
    meta = {
        "steps_per_epoch": len(data) // batch_size,
        "vocab_size": tokenizer.n_vocab
    }
    
    return CustomDataLoader(data, batch_size), CustomDataLoader(data, batch_size), meta
