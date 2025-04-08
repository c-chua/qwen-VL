import torch
import torch.nn.functional as F

class VLMInference:
    def __init__(self, model, processor, device="cuda"):
        self.model = model
        self.processor = processor
        self.device = device
    
    def generate(self, inputs, max_new_tokens=512):
        return self.model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    def get_generation_embedding(self, inputs, generated_ids):
        gen_tokens = generated_ids[:, inputs.input_ids.shape[-1]:]

        # Combine input + generated to get full token stream
        full_input_ids = torch.cat([inputs.input_ids, gen_tokens], dim=1)
        attention_mask = torch.ones_like(full_input_ids)

        # Run through model to get hidden states
        with torch.no_grad():
            outputs = self.model.model(
                input_ids=full_input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

        # Extract hidden states of the generated part
        hidden_states = outputs.hidden_states[-1]  # Last layer: (1, total_len, hidden_dim)
        gen_hidden_states = hidden_states[:, inputs.input_ids.shape[-1]:, :]  # Just generated

        # Mean pooling
        gen_embedding = gen_hidden_states.mean(dim=1)  # Shape: (1, hidden_dim)

        # Normalize for cosine similarity
        gen_embedding = F.normalize(gen_embedding, p=2, dim=1)

        return gen_embedding  # Tensor of shape (1, hidden_dim)