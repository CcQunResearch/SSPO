import os
import os.path as osp
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from template import *
from utils import *


class SFTDataset(Dataset):
    def __init__(self, file_path, template="default"):
        assert template in ["default", "qwen", "glm4", "llama3"], "template must be in ['default', 'qwen', 'glm4', 'llama3]"
        self.file_path = file_path
        self.template = template

        self.load_data()

    def load_data(self):
        temp = default_template
        if self.template == "qwen":
            temp = qwen_template
        elif self.template == "glm4":
            temp = glm4_template
        elif self.template == "llama3":
            temp = llama3_template

        self.all_data = json.load(open(self.file_path, "r", encoding="utf-8"))
        self.all_data = [{"instruction": temp.format(data["instruction"]), "output": data["output"]} for data in
                         self.all_data]

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        return self.all_data[index]


class DPODataset(Dataset):
    def __init__(self, file_path, add_kl_penalty=True, template="default"):
        assert template in ["default", "qwen", "glm4", "llama3"], "template must be in ['default', 'qwen', 'glm4', 'llama3]"
        self.file_path = file_path
        self.template = template
        self.add_kl_penalty = add_kl_penalty

        self.load_data()

    def load_data(self):
        temp = default_template
        if self.template == "qwen":
            temp = qwen_template
        elif self.template == "glm4":
            temp = glm4_template
        elif self.template == "llama3":
            temp = llama3_template

        self.raw_data = json.load(open(self.file_path, "r", encoding="utf-8"))
        self.all_data = []
        for data in self.raw_data:
            src_prompt = data["src_prompt"]
            for sample in data["sampling_records"]:
                if sample["same_flag"] is not True:
                    one_data = {"instruction": temp.format(src_prompt) + sample["temp_prompt"] + sample["src"] + "(",
                                "chosen": sample["chosen"], "rejected": sample["rejected"]}
                    if self.add_kl_penalty:
                        one_data["complete"] = data["single"]["accept"]
                    self.all_data.append(one_data)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        return self.all_data[index]
