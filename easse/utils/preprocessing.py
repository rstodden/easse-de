from importlib import import_module

import sacremoses

_TOKENIZERS = {
    "none": "tokenizer_none.NoneTokenizer",  # "tokenizer_base.BaseTokenizer",
    "13a": "tokenizer_13a.Tokenizer13a",
    "intl": "tokenizer_intl.TokenizerV14International",
}


def _get_tokenizer(name: str):
    """Dynamically import tokenizer as importing all is slow."""
    module_name, class_name = _TOKENIZERS[name].rsplit(".", 1)
    return getattr(import_module(f".tokenizers.{module_name}", "sacrebleu"), class_name)


def tokenize_spacy(sent: str, nlp, return_obj=False):
    if nlp == None:
        raise ValueError("tokenizer_obj mising.")
    if return_obj:
        return nlp(sent)
    else:
        return " ".join([tok.text for tok in nlp(sent)])


def normalize(sentence: str, lowercase: bool = True, tokenizer: str = "13a", return_str: bool = True,
              tokenizer_obj=None, return_obj=False):
    if lowercase:
        sentence = sentence.lower()

    if tokenizer in ["13a", "intl", "none"]:
        tokenizer_obj = _get_tokenizer(name=tokenizer)()
        normalized_sent = tokenizer_obj(sentence)
    elif tokenizer == "moses":
        normalized_sent = sacremoses.MosesTokenizer().tokenize(sentence, return_str=True, escape=False)
    elif tokenizer == "penn":
        normalized_sent = sacremoses.MosesTokenizer().penn_tokenize(sentence, return_str=True)
    elif tokenizer_obj and tokenizer == "spacy":
        normalized_sent = tokenize_spacy(sentence, tokenizer_obj, return_obj)
    else:
        normalized_sent = sentence
    if not return_str:
        normalized_sent = normalized_sent.split()

    return normalized_sent
