"""
utils.py

Fonctions utilitaires transverses au projet :
- chargement de la configuration (config.yaml) ;
- création des dossiers de sortie ;
- configuration du logger ;
- génération d'un générateur de nombres aléatoires (RNG) reproductible.
"""

import os
import logging

import numpy as np
import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Charge le fichier de configuration YAML du pipeline."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def ensure_output_dirs(base: str = "outputs") -> None:
    """Crée les sous-dossiers de sortie s'ils n'existent pas déjà."""
    for sub in ("figures", "tables", "reports", "logs"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)


def setup_logger(log_path: str = "outputs/logs/run_all.log") -> logging.Logger:
    """Configure un logger écrivant à la fois dans un fichier et sur la sortie standard."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger("mc_pricing")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(fmt)
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Retourne un générateur de nombres aléatoires NumPy (Generator API), reproductible via seed."""
    return np.random.default_rng(seed)
