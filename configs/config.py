# config/config.py

import yaml

class Config:
    def __init__(self, path="configs/yaml/config.yaml"):
        with open(path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    def __getitem__(self, item):
        return self.cfg[item]


# 全局单例
cfg = Config()