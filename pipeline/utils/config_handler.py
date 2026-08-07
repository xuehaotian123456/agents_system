"""YAML 配置加载器"""
import os
import yaml
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "config"
SETTINGS = {}

def _load_config():
    global SETTINGS
    if SETTINGS:
        return SETTINGS
    path = CONFIG_DIR / "settings.yaml"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            SETTINGS = yaml.safe_load(f)
    return SETTINGS

_load_config()

def get_llm_config():
    return SETTINGS.get("llm", {})

def get_chroma_config():
    return SETTINGS.get("chroma", {})

def get_hybrid_config():
    return SETTINGS.get("hybrid", {})

def get_data_config():
    return SETTINGS.get("data", {})

def get_crawler_config():
    return SETTINGS.get("crawler", {})

def get_agent_config():
    return SETTINGS.get("agent", {})
