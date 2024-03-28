from typing import List

import numpy as np
import textstat

import easse.utils.preprocessing as utils_prep
from easse.utils.constants import LANGUAGE



def corpus_fre(
    sys_sents: List[str],
    language: str = LANGUAGE,
):
    """Available for English, German, Spanish, French, Italian, Dutch, Polish, and Russian. """
    textstat.textstat.set_lang(language)
    fre_score = textstat.textstat.flesch_reading_ease(" ".join(sys_sents))
    return fre_score


def sentence_fre(
    sys_sent: str,
    language: str = LANGUAGE,
):
    return corpus_fre(
        [sys_sent],
        language=language,
    )


def corpus_averaged_sentence_fre(
    sys_sents: List[str],
    language: str = LANGUAGE,
):

    scores = []
    for sys_sent in sys_sents:
        scores.append(
            sentence_fre(
                sys_sent,
                languge=language,
            )
        )
    return np.mean(scores)


# German only metrics

def corpus_wiener(
    sys_sents: List[str],
    language: str = LANGUAGE,
    number: int = 1,
):
    textstat.textstat.set_lang(language)
    try:
        wiener_score = textstat.textstat.wiener_sachtextformel(" ".join(sys_sents), number)
    except ZeroDivisionError:
        wiener_score = np.nan
    return wiener_score


def sentence_wiener(
    sys_sent: str,
    language: str = LANGUAGE,
    number: int = 1,
):
    return corpus_wiener([sys_sent], number=number,

                         language=language,
                         )


def corpus_averaged_sentence_wiener(
    sys_sents: List[str],
    language: str = LANGUAGE,
    number: int = 1,
):
    textstat.textstat.set_lang(language)
    scores = list()
    for sent in sys_sents:
        scores.append(sentence_wiener(sent, number=number, 
                                      # lowercase=lowercase, tokenizer=tokenizer, tokenizer_obj=tokenizer_obj, 
                                      language=language,))
    return round(np.nanmean(scores),4)


def corpus_wiener_1(
    sys_sents: List[str],
    language: str = LANGUAGE,
):
    return corpus_wiener(sys_sents, 
                         language, 1, 
                        )


def corpus_wiener_2(
    sys_sents: List[str],
    language: str = LANGUAGE,
):
    return corpus_wiener(sys_sents, 
                         language, 2, 
                        )


def corpus_wiener_3(
    sys_sents: List[str],
    language: str = LANGUAGE,
):
    return corpus_wiener(sys_sents, 
                         language, 3, 
                        )


def corpus_wiener_4(
    sys_sents: List[str],
    language: str = LANGUAGE,
):
    return corpus_wiener(sys_sents, 
                         language, 4, 
                        )


def sent_wiener_1(
    sys_sents: List[str],
    language: str = LANGUAGE,
):
    return corpus_averaged_sentence_wiener(sys_sents, 
                                           language, 1, 
                                          )


def sent_wiener_2(
    sys_sents: List[str],
    language: str = LANGUAGE,
):
    return corpus_averaged_sentence_wiener(sys_sents, 
                                           language, 2, 
                                          )


def sent_wiener_3(
    sys_sents: List[str],
    language: str = LANGUAGE,
):
    return corpus_averaged_sentence_wiener(sys_sents, 
                                           language, 3, 
                                          )


def sent_wiener_4(
    sys_sents: List[str],
    language: str = LANGUAGE,
):
    return corpus_averaged_sentence_wiener(sys_sents, 
                                           language, 4, 
                                          )
    


## Spanish only metrics    
# >>> textstat.fernandez_huerta(test_data)
# >>> textstat.szigriszt_pazos(test_data)
# >>> textstat.gutierrez_polini(test_data)
# >>> textstat.crawford(test_data)

def corpus_fernandez_huerta(
    sys_sents: List[str],
    language: str = LANGUAGE,
):
    textstat.textstat.set_lang(language)
    fre_score = textstat.textstat.fernandez_huerta(" ".join(sys_sents))
    return fre_score


def sentence_fernandez_huerta(
    sys_sent: str,
    language: str = LANGUAGE,
):
    return corpus_fernandez_huerta(
        [sys_sent],
        language=language,
    )


def corpus_averaged_sentence_fernandez_huerta(
    sys_sents: List[str],
    language: str = LANGUAGE,
):

    scores = []
    for sys_sent in sys_sents:
        scores.append(
            sentence_fernandez_huerta(
                sys_sent,
                language=language,
            )
        )
    return np.mean(scores)

def corpus_szigriszt_pazos(
    sys_sents: List[str],
    language: str = LANGUAGE,
):
    textstat.textstat.set_lang(language)
    fre_score = textstat.textstat.szigriszt_pazos(" ".join(sys_sents))
    return fre_score


def sentence_szigriszt_pazos(
    sys_sent: str,
    language: str = LANGUAGE,
):
    return corpus_szigriszt_pazos(
        [sys_sent],
        language=language,
    )


def corpus_averaged_sentence_szigriszt_pazos(
    sys_sents: List[str],
    language: str = LANGUAGE,
):

    scores = []
    for sys_sent in sys_sents:
        scores.append(
            sentence_szigriszt_pazos(
                sys_sent,
                language=language,
            )
        )
    return np.mean(scores)


def corpus_gutierrez_polini(
    sys_sents: List[str],
    language: str = LANGUAGE,
):
    textstat.textstat.set_lang(language)
    fre_score = textstat.textstat.gutierrez_polini(" ".join(sys_sents))
    return fre_score


def sentence_gutierrez_polini(
    sys_sent: str,
    language: str = LANGUAGE,
):
    return corpus_gutierrez_polini(
        [sys_sent],
        language=language,
    )


def corpus_averaged_sentence_gutierrez_polini(
    sys_sents: List[str],
    language: str = LANGUAGE,
):

    scores = []
    for sys_sent in sys_sents:
        scores.append(
            sentence_gutierrez_polini(
                sys_sent,
                language=language,
            )
        )
    return np.mean(scores)
    
    
def corpus_crawford(
    sys_sents: List[str],
    language: str = LANGUAGE,
):
    textstat.textstat.set_lang(language)
    fre_score = textstat.textstat.crawford(" ".join(sys_sents))
    return fre_score


def sentence_crawford(
    sys_sent: str,
    language: str = LANGUAGE,
):
    return corpus_crawford(
        [sys_sent],
        language=language,
    )


def corpus_averaged_sentence_crawford(
    sys_sents: List[str],
    language: str = LANGUAGE,
):

    scores = []
    for sys_sent in sys_sents:
        scores.append(
            sentence_crawford(
                sys_sent,
                language=language,
            )
        )
    return np.mean(scores)



## Arabic only metrics
# textstat.osman(text)

def corpus_osman(
    sys_sents: List[str],
    language: str = LANGUAGE,
):
    textstat.textstat.set_lang(language)
    fre_score = textstat.textstat.osman(" ".join(sys_sents))
    return fre_score


def sentence_osman(
    sys_sent: str,
    language: str = LANGUAGE,
):
    return corpus_osman(
        [sys_sent],
        language=language,
    )


def corpus_averaged_sentence_osman(
    sys_sents: List[str],
    language: str = LANGUAGE,
):

    scores = []
    for sys_sent in sys_sents:
        scores.append(
            sentence_osman(
                sys_sent,
                language=language,
            )
        )
    return np.mean(scores)


## Italian only metrics    
# textstat.gulpease_index(text)


def corpus_gulpease_index(
    sys_sents: List[str],
    language: str = LANGUAGE,
):
    textstat.textstat.set_lang(language)
    fre_score = textstat.textstat.gulpease_index(" ".join(sys_sents))
    return fre_score


def sentence_gulpease_index(
    sys_sent: str,
    language: str = LANGUAGE,
):
    return corpus_gulpease_index(
        [sys_sent],
        language=language,
    )


def corpus_averaged_sentence_gulpease_index(
    sys_sents: List[str],
    language: str = LANGUAGE,
):

    scores = []
    for sys_sent in sys_sents:
        scores.append(
            sentence_gulpease_index(
                sys_sent,
                language=language,
            )
        )
    return np.mean(scores)
