# utils/config_loader.py

import os
import yaml
from ruamel.yaml import YAML

# プロジェクトルートを自動検出
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# デフォルトのYAML パス
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
DEFAULT_HPARAM_PATH = os.path.join(PROJECT_ROOT, "configs", "hyperparameter.yaml")
DEFAULT_RUNTIME_PATH = os.path.join(PROJECT_ROOT, "configs", "runtime.yaml")
DEFAULT_COMPARE_PATH = os.path.join(PROJECT_ROOT, "configs", "compare.yaml")

yaml_ruamel = YAML()    # ruamel 用インスタンス


def resolve_yaml_path(exp_dir, filename, default_path):
    if exp_dir is not None:
        return os.path.join(exp_dir, filename)
    return default_path


def load_config(exp_dir=None):
    path = resolve_yaml_path(exp_dir, "config.yaml", DEFAULT_CONFIG_PATH)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_hyperparameter(exp_dir=None):
    path = resolve_yaml_path(exp_dir, "hyperparameter.yaml", DEFAULT_HPARAM_PATH)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_runtime(exp_dir=None):
    path = resolve_yaml_path(exp_dir, "runtime.yaml", DEFAULT_RUNTIME_PATH)
    with open(path, "r") as f:
        return yaml_ruamel.load(f)


def save_runtime(runtime, exp_dir=None):
    path = resolve_yaml_path(exp_dir, "runtime.yaml", DEFAULT_RUNTIME_PATH)
    with open(path, "w") as f:
        yaml_ruamel.dump(runtime, f)


def load_compare():
    path = DEFAULT_COMPARE_PATH
    with open(path, "r") as f:
        return yaml.safe_load(f)