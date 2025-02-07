import os.path as osp
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class DurationPredictor(nn.Module):
    def __init__(self, base_model_path):
        super(DurationPredictor, self).__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True)
        self.predictor_head = nn.Linear(self.base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        logits = self.predictor_head(last_hidden_state[:, -1, :]).squeeze(-1)
        return logits

    def save_model(self, hf_save_path, model_state_dict, safe_serialization=True):
        self.base_model.save_pretrained(hf_save_path, state_dict=model_state_dict,
                                        safe_serialization=safe_serialization)
        torch.save(self.predictor_head.state_dict(), osp.join(hf_save_path, "predictor_head.pth"))

    @classmethod
    def load_model(cls, load_path):
        model = cls(base_model_path=load_path)
        predictor_head_state_dict = torch.load(osp.join(load_path, "predictor_head.pth"))
        model.predictor_head.load_state_dict(predictor_head_state_dict)
        return model
