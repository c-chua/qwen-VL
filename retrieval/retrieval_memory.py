import torch
import torch.nn.functional as F
from sentence_transformers import util
import json

class RetrievalMemory:
    def __init__(self, memory_path, device="cuda"):
        self.device = device
        self.memory_path = memory_path
        self.keys = []
        self.actions = []
        self.embeddings = None
        self.load_memory()

    def load_memory(self):
        with open(self.memory_path, "r") as f:
            memory = json.load(f)

        self.keys = list(memory.keys())
        self.actions = [memory[k]["action"] for k in self.keys]
        self.embeddings = torch.stack([
            torch.tensor(memory[k]["embedding"], dtype=torch.float32).to(self.device)
            for k in self.keys
        ])

    def query(self, qwen_embedding, top_k=1):
        qwen_embedding = F.normalize(qwen_embedding.squeeze(0).to(torch.float32), p=2, dim=0)
        sims = util.pytorch_cos_sim(qwen_embedding, self.embeddings)[0]
        topk = torch.topk(sims, k=top_k)

        results = []
        for idx, score in zip(topk.indices, topk.values):
            results.append({
                "situation_key": self.keys[idx],
                "similarity": round(score.item(), 3),
                "action": self.actions[idx]
            })

        return results
    
    def save_new_situation(self, embedding_tensor, situation_text, full_action_text):
        # Load current memory
        with open(self.memory_path, "r") as f:
            memory = json.load(f)

        # Determine next situation key
        existing_keys = list(memory.keys())
        existing_nums = [int(k.split("_")[1]) for k in existing_keys if k.startswith("situation_")]
        next_index = max(existing_nums) + 1 if existing_nums else 1
        next_key = f"situation_{next_index}"

        # Determine action from action text
        action_text_lower = full_action_text.lower()
        if "change lane right" in action_text_lower:
            action = "Change lane right"
        elif "change lane left" in action_text_lower:
            action = "Change lane left"
        else:
            action = "Continue"

        # Save entry
        memory[next_key] = {
            "embedding": embedding_tensor.squeeze(0).tolist(),
            "action": action,
            "situation": situation_text.strip()
        }

        # Write back to file
        with open(self.memory_path, "w") as f:
            json.dump(memory, f, indent=2)

        print(f"Saved: {next_key} → Action: {action}")