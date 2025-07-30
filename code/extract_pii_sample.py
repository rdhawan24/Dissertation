import math
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import SpacyNlpEngine
import spacy

def setup_tools():
    tokenizer = AutoTokenizer.from_pretrained("ab-ai/pii_model")
    model = AutoModelForTokenClassification.from_pretrained("ab-ai/pii_model")
    pii_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    spacy.load("en_core_web_lg")
    config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}]
    }
    analyzer = AnalyzerEngine(nlp_engine=SpacyNlpEngine(config), supported_languages=["en"])

    card_patterns = [
        Pattern("VISA", r"\b4[0-9]{12}(?:[0-9]{3})?\b", 0.8),
        Pattern("MASTERCARD", r"\b5[1-5][0-9]{14}\b", 0.8),
        Pattern("AMEX", r"\b3[47][0-9]{13}\b", 0.8),
        Pattern("DISCOVER", r"\b6(?:011|5[0-9]{2})[0-9]{12}\b", 0.8),
    ]
    card_recognizer = PatternRecognizer(supported_entity="CREDIT_CARD", patterns=card_patterns)
    analyzer.registry.add_recognizer(card_recognizer)

    money_pattern = Pattern("MONEY", r"\b(?:USD|GBP|\$|£|₹|Rs\.?)\s?\d{1,3}(?:[.,]?\d{3})*(?:\.\d{2})?\b", 0.6)
    money_recognizer = PatternRecognizer(supported_entity="MONEY", patterns=[money_pattern])
    analyzer.registry.add_recognizer(money_recognizer)

    return pii_pipeline, analyzer

_tools = None
def get_tools():
    global _tools
    if _tools is None:
        _tools = setup_tools()
    return _tools

def process_chunk(data_slice: pd.DataFrame) -> pd.DataFrame:
    pii_pipeline, analyzer = get_tools()
    results = []

    for _, row in data_slice.iterrows():
        body = row["body"]
        pii_entities = pii_pipeline(body)
        firstnames = [e["word"] for e in pii_entities if e["entity_group"] == "FIRSTNAME"]
        lastnames = [e["word"] for e in pii_entities if e["entity_group"] == "LASTNAME"]
        presidio_results = analyzer.analyze(text=body, entities=["CREDIT_CARD", "MONEY"], language="en")
        card_numbers = [body[r.start:r.end] for r in presidio_results if r.entity_type == "CREDIT_CARD"]
        currencies = [body[r.start:r.end] for r in presidio_results if r.entity_type == "MONEY"]

        results.append({
            "file": row["file"],
            "body": body,
            "firstnames": firstnames,
            "lastnames": lastnames,
            "card_numbers": card_numbers,
            "currencies": currencies
        })

    return pd.DataFrame(results)

def extract_pii_sample(csv_path: Path, sample_size: int = 20, workers: int = 4) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["file", "body"]).head(sample_size)
    chunk_size = math.ceil(len(df) / workers)
    slices = [(i, min(chunk_size, len(df) - i)) for i in range(0, len(df), chunk_size)]

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_chunk, df.iloc[start:start+count])
            for start, count in slices
        ]
        for f in futures:
            results.append(f.result())

    return pd.concat(results, ignore_index=True)

# Example usage:
    result_df = extract_pii_sample(Path("emails_clean.csv"), sample_size=20, workers=4)
    print(result_df)

