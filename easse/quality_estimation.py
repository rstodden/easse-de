from typing import List

import numpy as np

from tseval.feature_extraction import (
	get_compression_ratio,
	count_sentence_splits,
	get_levenshtein_similarity,
	is_exact_match,
	get_wordrank_score,
	wrap_single_sentence_vectorizer,
	count_words_per_sentence,
	count_syllables_per_word,
	get_unchanged_words_proportion,
	get_added_words_proportion,
	get_deleted_words_proportion,
	get_rewritten_words_proportion,
	get_kept_words_proportion,
	get_unchanged_words_proportion,
	get_parse_tree_height,
	get_ratio_clauses,
	max_pos_in_freq_table,
	average_pos_in_freq_table
)

from easse.utils.constants import LANGUAGE
from easse.utils.preprocessing import normalize


def get_average_pair(vectorizer, orig_sentences, sys_sentences, language):
    cumsum = 0
    count = 0
    for orig_sentence, sys_sentence in zip(orig_sentences, sys_sentences):
        cumsum += vectorizer(orig_sentence, sys_sentence, language)
        count += 1
    return cumsum / count


def get_average_sent(vectorizer, sentences, language):
    cumsum = 0
    count = 0
    for sent in sentences:
        cumsum += vectorizer(sent, language)
        count += 1
    return cumsum / count
    

def corpus_quality_estimation(
    orig_sentences: List[str], sys_sentences: List[str], lowercase: bool = False, tokenizer: str = '13a',
		language=LANGUAGE, tokenizer_obj=None,
):
    orig_sentences_joined = [" ".join([token.text for token in sent]) for sent in orig_sentences]
    sys_sentences_joined = [" ".join([token.text for token in sent]) for sent in sys_sentences]
    return {
        'Compression ratio': get_average_pair(get_compression_ratio, orig_sentences, sys_sentences, language),
        'Sentence splits': get_average_pair(count_sentence_splits, orig_sentences, sys_sentences, language),
        'Levenshtein similarity': get_average_pair(get_levenshtein_similarity, orig_sentences, sys_sentences, language),
        'Exact copies': get_average_pair(is_exact_match, orig_sentences_joined, sys_sentences_joined, language),
        
        'Added lemmas proportion': get_average_pair(get_added_words_proportion, orig_sentences, sys_sentences, language),
        'Deleted lemmas proportion': get_average_pair(get_deleted_words_proportion, orig_sentences, sys_sentences, language),
        'Kept lemmas proportion': get_average_pair(get_kept_words_proportion, orig_sentences, sys_sentences, language),  # based on lemma
        'Kept words proportion': get_average_pair(get_unchanged_words_proportion, orig_sentences, sys_sentences, language),  # based on token
        'Rewritten words proportion': get_average_pair(get_rewritten_words_proportion, orig_sentences, sys_sentences, language),
        # 'Lexical complexity score': get_average_pair(
        #     wrap_single_sentence_vectorizer(get_wordrank_score), orig_sentences, sys_sentences, language
        # ),
        'Avg. sentence length (in words)': get_average_sent(count_words_per_sentence, sys_sentences, language),
        'Avg. number syllables per word': get_average_sent(count_syllables_per_word, sys_sentences, language),
        'Avg. parse tree height': get_average_sent(get_parse_tree_height, sys_sentences, language),
        'Avg. number clauses': get_average_sent(get_ratio_clauses, sys_sentences, language),
        'Avg. max. pos. freq.': get_average_sent(max_pos_in_freq_table, sys_sentences, language),
        'Avg. avg. pos. freq.': get_average_sent(average_pos_in_freq_table, sys_sentences, language),
        'Lexical complexity score': get_average_sent(get_wordrank_score, sys_sentences, language
        ),
    }
