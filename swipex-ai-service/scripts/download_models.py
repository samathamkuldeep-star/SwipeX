#!/usr/bin/env python3
"""Model pre-fetch script for offline/docker build execution."""

import sys
import logging
from sentence_transformers import SentenceTransformer
import spacy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("download_models")


def main():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    logger.info(f"Downloading sentence-transformer model: {model_name}")
    try:
        SentenceTransformer(model_name)
        logger.info("SentenceTransformer model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to download sentence-transformer model: {e}")
        sys.exit(1)

    spacy_model = "en_core_web_sm"
    logger.info(f"Checking spaCy model: {spacy_model}")
    try:
        spacy.load(spacy_model)
        logger.info(f"spaCy model {spacy_model} is available.")
    except OSError:
        logger.info(f"Downloading spaCy model {spacy_model}...")
        from spacy.cli import download
        try:
            download(spacy_model)
            logger.info(f"spaCy model {spacy_model} downloaded successfully.")
        except Exception as e:
            logger.warning(f"Could not download {spacy_model}: {e}. Fallback to spacy.blank('en') will be used at runtime.")


if __name__ == "__main__":
    main()
