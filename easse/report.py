import html
from collections import OrderedDict
from typing import List
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.express as px
from tseval.feature_extraction import get_levenshtein_similarity, get_compression_ratio, count_sentences
from yattag import Doc, indent

from easse.annotation.lcs import get_lcs
# from easse.bertscore import corpus_bertscore
from easse.bleu import corpus_bleu
from easse.fkgl import corpus_fkgl
from easse.quality_estimation import corpus_quality_estimation
from easse.sari import corpus_sari
# from easse.dsari import corpus_document_sari
# from easse.bartscore import corpus_bartscore
# from easse.rouge import corpus_rouge
from easse.textstat_metrics import (corpus_averaged_sentence_fre, corpus_fre,
                                    corpus_wiener_1, corpus_wiener_2, corpus_wiener_3, corpus_wiener_4,
                                    sent_wiener_1, sent_wiener_2, sent_wiener_3, sent_wiener_4,
                                    corpus_averaged_sentence_osman, corpus_osman,
                                    corpus_averaged_sentence_fernandez_huerta, corpus_fernandez_huerta,
                                    corpus_averaged_sentence_szigriszt_pazos, corpus_szigriszt_pazos,
                                    corpus_averaged_sentence_gutierrez_polini, corpus_gutierrez_polini,
                                    corpus_averaged_sentence_crawford, corpus_crawford,
                                    corpus_averaged_sentence_gulpease_index, corpus_gulpease_index)
from easse.utils.constants import DEFAULT_METRICS, LANGUAGE
from easse.utils.helpers import add_dicts
from easse.utils.text import to_words, count_words
from easse.utils.preprocessing import normalize
import spacy 

def get_all_scores(
    orig_sents_tok: List[str],
    sys_sents_tok: List[str],
    refs_sents_tok: List[List[str]],
    orig_sents_nlp: List[str],
    sys_sents_nlp: List[str],
    refs_sents_nlp: List[List[str]],
    lowercase: bool = False,
    tokenizer: str = '13a',
    metrics: List[str] = DEFAULT_METRICS,
    language: str = LANGUAGE,
    tokenizer_obj=None,
):
    scores = OrderedDict()
    if 'bleu' in metrics:
        scores['BLEU'] = corpus_bleu(sys_sents_tok, refs_sents_tok, force=True,)  #  tokenizer=tokenizer, lowercase=lowercase,
                                     # tokenizer_obj=tokenizer_obj,)
    if 'sari' in metrics:
        scores['SARI'] = corpus_sari(orig_sents_tok, sys_sents_tok, refs_sents_tok,) # tokenizer=tokenizer, lowercase=lowercase,
                                     # tokenizer_obj=tokenizer_obj,)
    # if 'document_sari' in metrics:
    #     scores['DSARI'] = corpus_document_sari(orig_sents, sys_sents, refs_sents, tokenizer=tokenizer, lowercase=lowercase,
    #                                  tokenizer_obj=tokenizer_obj,)
    if 'samsa' in metrics:
        from easse.samsa import corpus_samsa

        scores['SAMSA'] = corpus_samsa(orig_sents_tok, sys_sents_tok, 
                                       # tokenizer=tokenizer, 
                                       verbose=True, 
                                       # lowercase=lowercase,
                                       # tokenizer_obj=tokenizer_obj,
                                      )
    if 'fkgl' in metrics:
        scores['FKGL'] = corpus_fkgl(sys_sents_tok, 
                                     # tokenizer=tokenizer, 
                                     # lowercase=lowercase,
                                     # tokenizer_obj=tokenizer_obj,
                                    )

    # if "rouge" in metrics:
    #     scores["rouge"] = corpus_rouge([sys_sents], refs_sents, tokenizer=tokenizer, lowercase=lowercase, tokenizer_obj=tokenizer_obj,
    #                   language=language)

    if "bertscore" in metrics:
        from easse.bertscore import corpus_bertscore
        (
            scores["bertscore_precision"],
            scores["bertscore_recall"],
            scores["bertscore_f1"],
        ) = corpus_bertscore(sys_sents_tok, refs_sents_tok, language=language,)
                            #  tokenizer=tokenizer, lowercase=lowercase, language=language,
                            # tokenizer_obj=tokenizer_obj,)

    # if "bartscore" in metrics:
    #     from easse.bartscore import corpus_bartscore
    #     (
    #         scores["bartscore_precision"],
    #         scores["bartscore_recall"],
    #         scores["bartscore_f1"],
    #     ) = corpus_bartscore(orig_sents, sys_sents, refs_sents, tokenizer=tokenizer, lowercase=lowercase, language=language,
    #                         tokenizer_obj=tokenizer_obj,)

    # if 'fre_sent' in metrics:
    #     scores["sent_FRE"] = corpus_averaged_sentence_fre(sys_sents, tokenizer=tokenizer, lowercase=lowercase, language=language,
    #                                        tokenizer_obj=tokenizer_obj,)
    if 'fre' in metrics:
        scores["FRE"] = corpus_fre(sys_sents_tok, language=language,)
                                           # tokenizer=tokenizer, lowercase=lowercase, tokenizer_obj=tokenizer_obj,)
    # if 'wiener_sachtextformel_1_sent' in metrics:
    #     scores["Wiener-Sachtextformel-1-sent"] = sent_wiener_1(sys_sents, tokenizer=tokenizer, lowercase=lowercase,
    #                                                                 language=language,tokenizer_obj=tokenizer_obj,)
    # if 'wiener_sachtextformel_2_sent' in metrics:
    #     scores["Wiener-Sachtextformel-2-sent"] = sent_wiener_2(sys_sents, tokenizer=tokenizer, lowercase=lowercase,
    #                                                                 language=language,tokenizer_obj=tokenizer_obj,)
    # if 'wiener_sachtextformel_3_sent' in metrics:
    #     scores["Wiener-Sachtextformel-3-sent"] = sent_wiener_3(sys_sents, tokenizer=tokenizer, lowercase=lowercase,
    #                                                                 language=language,tokenizer_obj=tokenizer_obj,)
    # if 'wiener_sachtextformel_4_sent' in metrics:
    #     scores["Wiener-Sachtextformel-4-sent"] = sent_wiener_4(sys_sents, tokenizer=tokenizer, lowercase=lowercase,
    #                                                                 language=language,tokenizer_obj=tokenizer_obj,)
    if 'wiener_sachtextformel_1' in metrics:
        scores["Wiener-Sachtextformel-1"] = corpus_wiener_1(sys_sents_tok, language=language,)
                                           # tokenizer=tokenizer, lowercase=lowercase, tokenizer_obj=tokenizer_obj,)
    if 'wiener_sachtextformel_2' in metrics:
        scores["Wiener-Sachtextformel-2"] = corpus_wiener_2(sys_sents_tok, language=language,)
                                           # tokenizer=tokenizer, lowercase=lowercase, tokenizer_obj=tokenizer_obj,)
    if 'wiener_sachtextformel_3' in metrics:
        scores["Wiener-Sachtextformel-3"] = corpus_wiener_3(sys_sents_tok, language=language,)
                                           # tokenizer=tokenizer, lowercase=lowercase, tokenizer_obj=tokenizer_obj,)
    if 'wiener_sachtextformel_4' in metrics:
        scores["Wiener-Sachtextformel-4"] = corpus_wiener_4(sys_sents_tok, language=language,)
                                           # tokenizer=tokenizer, lowercase=lowercase, tokenizer_obj=tokenizer_obj,)
    # if 'osman_sent' in metrics:
    #     scores["sent_osman"] = corpus_averaged_sentence_osman(sys_sents, tokenizer=tokenizer, lowercase=lowercase, language=language,
    #                                        tokenizer_obj=tokenizer_obj,)
    if 'osman' in metrics:
        scores["osman"] = corpus_osman(sys_sents_tok, language=language,)
                                           # tokenizer=tokenizer, lowercase=lowercase, tokenizer_obj=tokenizer_obj,)
    # if 'fernandez_huerta_sent' in metrics:
    #     scores["sent_fernandez_huerta"] = corpus_averaged_sentence_fernandez_huerta(sys_sents, tokenizer=tokenizer, lowercase=lowercase, language=language,
    #                                        tokenizer_obj=tokenizer_obj,)
    if 'fernandez_huerta' in metrics:
        scores["fernandez_huerta"] = corpus_fernandez_huerta(sys_sents_tok, language=language,)
                                           # tokenizer=tokenizer, lowercase=lowercase, tokenizer_obj=tokenizer_obj,)
    # if 'szigriszt_pazos_sent' in metrics:
    #     scores["sent_szigriszt_pazos"] = corpus_averaged_sentence_szigriszt_pazos(sys_sents, tokenizer=tokenizer, lowercase=lowercase, language=language,
    #                                        tokenizer_obj=tokenizer_obj,)
    if 'szigriszt_pazos' in metrics:
        scores["szigriszt_pazos"] = corpus_szigriszt_pazos(sys_sents_tok, language=language,)
                                           # tokenizer=tokenizer, lowercase=lowercase, tokenizer_obj=tokenizer_obj,)
    # if 'gutierrez_polini_sent' in metrics:
    #     scores["sent_gutierrez_polini"] = corpus_averaged_sentence_gutierrez_polini(sys_sents, tokenizer=tokenizer, lowercase=lowercase, language=language,
    #                                        tokenizer_obj=tokenizer_obj,)
    if 'gutierrez_polini' in metrics:
        scores["gutierrez_polini"] = corpus_gutierrez_polini(sys_sents_tok, language=language,)
                                           # tokenizer=tokenizer, lowercase=lowercase, tokenizer_obj=tokenizer_obj,)
    # if 'crawford_sent' in metrics:
    #     scores["sent_crawford"] = corpus_averaged_sentence_crawford(sys_sents, tokenizer=tokenizer, lowercase=lowercase, language=language,
    #                                        tokenizer_obj=tokenizer_obj,)
    if 'crawford' in metrics:
        scores["crawford"] = corpus_crawford(sys_sents_tok, language=language,)
                                           # tokenizer=tokenizer, lowercase=lowercase, tokenizer_obj=tokenizer_obj,)
    # if 'gulpease_index_sent' in metrics:
    #     scores["sent_gulpease_index"] = corpus_averaged_sentence_gulpease_index(sys_sents, tokenizer=tokenizer, lowercase=lowercase, language=language,
    #                                        tokenizer_obj=tokenizer_obj,)
    if 'gulpease_index' in metrics:
        scores["gulpease_index"] = corpus_gulpease_index(sys_sents_tok, language=language,)
                                           # tokenizer=tokenizer, lowercase=lowercase, tokenizer_obj=tokenizer_obj,)

    quality_estimation_scores = corpus_quality_estimation(
        orig_sents_nlp, sys_sents_nlp, 
        # tokenizer=tokenizer, lowercase=lowercase, 
        language=language, 
        # tokenizer_obj=tokenizer_obj,
    )
    scores = add_dicts(
        scores,
        quality_estimation_scores,
    )
    return {key: round(value, 2) for key, value in scores.items()}


def make_differing_words_bold(orig_sent, sys_sent, make_bold):
    '''Returns the two sentences with differing words in bold'''

    def format_words(words, mutual_words):
        '''Makes all words bold except the mutual ones'''
        words_generator = iter(words)
        formatted_string = ''
        for mutual_word in mutual_words:
            word = next(words_generator)
            bold_text = ''
            while word != mutual_word:
                bold_text += ' ' + word
                word = next(words_generator)
            if bold_text != '':
                formatted_string += ' ' + make_bold(bold_text)
            formatted_string += ' ' + word
        # Add remaining words
        formatted_string += ' ' + make_bold(' '.join(words_generator))
        return formatted_string.strip()

    orig_words = to_words(orig_sent)
    sys_words = to_words(sys_sent)
    mutual_words = get_lcs(orig_words, sys_words)
    return format_words(orig_words, mutual_words), format_words(sys_words, mutual_words)


def make_text_bold_html(text):
    doc = Doc()
    doc.line('strong', text)
    return doc.getvalue()


def get_random_html_id():
    html_id = str(uuid4())
    return 'a' + html_id[1:]  # HTML id can't start with a number


def get_qualitative_examples_html(orig_sents_tok, sys_sents_tok, refs_sents_tok, orig_sents_nlp, sys_sents_nlp, refs_sents_nlp, language=LANGUAGE):
    
    title_key_print = [
        ('Randomly sampled simplifications', lambda c, s, refs: 0, lambda value: ''),
        (
            'Best simplifications according to SARI',
            lambda c, s, refs: -corpus_sari([c], [s], [[ref] for ref in refs], language),
            lambda value: f'SARI={-value:.2f}',
        ),
        (
            'Worst simplifications according to SARI',
            lambda c, s, refs: corpus_sari([c], [s], [[ref] for ref in refs], language),
            lambda value: f'SARI={value:.2f}',
        ),
        # (
        #     'Simplifications with the most compression',
        #     lambda c, s, refs: get_compression_ratio(c, s, language),
        #     lambda value: f'compression_ratio={value:.2f}',
        # ),
        # (
        #     'Simplifications with a high amount of paraphrasing',
        #     lambda c, s, refs: get_levenshtein_similarity(c, s, language) / get_compression_ratio(c, s, language),
        #     lambda value: f'levenshtein_similarity={value:.2f}',
        # ),
        # (
        #     'Simplifications with the most sentence splits (if any)',
        #     lambda c, s, refs: -(count_sentences(s, language) - count_sentences(c, language)),
        #     lambda value: f'#sentence_splits={-value:.2f}',
        # ),
    ]

    def get_one_sample_html(orig_sent_tok, sys_sent_tok, ref_sents_tok, sort_key, print_func):
        def get_one_sentence_html(sentence, system_name):
            doc = Doc()
            with doc.tag('div', klass='row'):
                with doc.tag('div', klass='col-2'):
                    doc.text(system_name)
                with doc.tag('div', klass='col'):
                    doc.asis(sentence)
            return doc.getvalue()
        orig_sent_tok, sys_sent_tok, *ref_sents_tok = [html.escape(sent) for sent in [orig_sent_tok, sys_sent_tok, *ref_sents_tok]]
        doc = Doc()
        # print("sys", sys_sents[:5], "orig", orig_sents[:5], "ref", refs_sents[0][:5])
        with doc.tag('div', klass='mb-2 p-1'):
            # Sort key
            with doc.tag('div', klass='text-muted small'):
                doc.asis(print_func(sort_key(orig_sent_tok, sys_sent_tok, ref_sents_tok)))
            with doc.tag('div', klass='ml-2'):
                orig_sent_bold, sys_sent_bold = make_differing_words_bold(orig_sent_tok, sys_sent_tok, make_text_bold_html)
                # Source
                with doc.tag('div'):
                    # doc.asis("Original: " + orig_sent_bold)
                    doc.asis(get_one_sentence_html(orig_sent_bold, 'Original'))
                # Prediction
                with doc.tag('div'):
                    # doc.asis("System Prediction: " + sys_sent_bold)
                    doc.asis(get_one_sentence_html(sys_sent_bold, 'System Prediction'))
                # References
                collapse_id = get_random_html_id()
                with doc.tag('div', klass='position-relative'):
                    with doc.tag(
                        'a', ('data-toggle', 'collapse'), ('href', f'#{collapse_id}'), klass='stretched-link small'
                    ):
                        doc.text('References')
                    with doc.tag('div', klass='collapse', id=collapse_id):
                        for ref_sent_tok in ref_sents_tok:
                            _, ref_sent_bold = make_differing_words_bold(orig_sent_tok, ref_sent_tok, make_text_bold_html)
                            with doc.tag('div', klass='text-muted'):
                                doc.asis(ref_sent_bold)
        return doc.getvalue()

    doc = Doc()
    for title, sort_key, print_func in title_key_print:
        with doc.tag('div', klass='container-fluid mt-4 p-2 border'):
            collapse_id = get_random_html_id()
            with doc.tag('a', ('data-toggle', 'collapse'), ('href', f'#{collapse_id}')):
                doc.line('h3', klass='m-2', text_content=title)
            # Now lets print the examples
            # Shapes: orig_sents: n_samples, sys_sents: n_samples, refs_sents: (n_refs, n_sample)
            sample_generator = sorted(
                zip(orig_sents_tok, sys_sents_tok, zip(*refs_sents_tok)),
                key=lambda args: sort_key(*args),
            )
            # Samples displayed by default
            with doc.tag('div', klass='collapse', id=collapse_id):
                n_samples = 50
                for i, (orig_sent_tok, sys_sent_tok, refs_tok) in enumerate(sample_generator):
                    if i >= n_samples:
                        break
                    doc.asis(get_one_sample_html(orig_sent_tok, sys_sent_tok, refs_tok, sort_key, print_func))
    return doc.getvalue()


def get_test_set_description_html(test_set, orig_sents_tok, refs_sents_tok, orig_sents_nlp, refs_sents_nlp, lowercase, tokenizer, tokenizer_obj, lang):
    doc = Doc()
    test_set = test_set.capitalize()
    doc.line('h4', test_set)
    orig_sents_tok = np.array(orig_sents_tok)
    refs_sents_tok = np.array(refs_sents_tok)
    df = pd.DataFrame()
    df.loc[test_set, '# of samples'] = str(len(orig_sents_tok))
    df.loc[test_set, '# of references'] = str(len(refs_sents_tok))
    df.loc[test_set, 'Words / source'] = np.average(np.vectorize(count_words)(orig_sents_tok))

    df.loc[test_set, 'Words / reference'] = np.average(np.vectorize(count_words)(refs_sents_tok.flatten()))

    def modified_count_sentences(sent, lang):
        return max(count_sentences(sent, lang), 1)
    sents_length = list()
    for sent in orig_sents_nlp:
        sents_length.append(count_sentences(sent, lang))
    orig_sent_counts = np.array(sents_length)
    # orig_sent_counts = np.vectorize(modified_count_sentences)(orig_sents_nlp, lang)
    expanded_orig_sent_counts = np.expand_dims(orig_sent_counts, 0).repeat(len(refs_sents_tok), axis=0)
    # refs_sent_counts = np.vectorize(modified_count_sentences)(refs_sents_nlp, lang)
    ref_sents_length = list()
    for ref_sents in refs_sents_nlp:
        ref_sent_length = list()
        for sent in ref_sents:
            ref_sent_length.append(count_sentences(sent, lang))
        ref_sents_length.append(ref_sent_length)
    refs_sent_counts = np.array(ref_sents_length)
    ratio = np.average((expanded_orig_sent_counts == 1) & (refs_sent_counts == 1))
    df.loc[test_set, '1-to-1 alignments*'] = f'{ratio*100:.1f}%'
    ratio = np.average((expanded_orig_sent_counts == 1) & (refs_sent_counts > 1))
    df.loc[test_set, '1-to-N alignments*'] = f'{ratio*100:.1f}%'
    ratio = np.average((expanded_orig_sent_counts > 1) & (refs_sent_counts > 1))
    df.loc[test_set, 'N-to-N alignments*'] = f'{ratio*100:.1f}%'
    ratio = np.average((expanded_orig_sent_counts > 1) & (refs_sent_counts == 1))
    df.loc[test_set, 'N-to-1 alignments*'] = f'{ratio*100:.1f}%'
    doc.asis(get_table_html_from_dataframe(df.round(2)))
    doc.line('p', klass='text-muted', text_content='* Alignment detection is not 100% accurate')
    return doc.getvalue()


def get_plotly_html(plotly_figure):
    doc = Doc()
    plot_id = get_random_html_id()
    # Empty div to hold the plot
    with doc.tag('div', id=plot_id):
        # Embedded javascript code that uses plotly to fill the div
        with doc.tag('script'):
            doc.asis(
                f"var plotlyJson = '{plotly_figure.to_json()}'; var figure = JSON.parse(plotlyJson); var plotDiv = document.getElementById('{plot_id}'); Plotly.newPlot(plotDiv, figure.data, figure.layout, {{responsive: true}});"
            )  # noqa: E501
    return doc.getvalue()


def get_plotly_histogram(orig_sents, sys_sents, ref_sents, feature_extractor, feature_name, lang=LANGUAGE):
    '''feature_extractor(orig_sent, sys_sent) -> scalar'''
    data = []
    for orig_sent, sys_sent, ref_sent in zip(orig_sents, sys_sents, ref_sents):
        data.append({'Model': 'System output', feature_name: feature_extractor(orig_sent, sys_sent, lang)})
        data.append({'Model': 'Reference', feature_name: feature_extractor(orig_sent, ref_sent, lang)})
    figure = px.histogram(
        pd.DataFrame(data),
        title=feature_name,
        x=feature_name,
        color='Model',
        nbins=100,
        histnorm=None,
        barmode='overlay',
        opacity=0.7,
        color_discrete_map={'Reference': '#228B22', 'System output': '#B22222'},
        category_orders={'Model': ['System output', 'Reference']},
        width=800,
    )
    figure.layout['hovermode'] = 'x'  # To compare on hover
    figure.data[-1]['marker']['opacity'] = 0.5  # So that the reference is transparent in front of the system output
    return figure


def get_plots_html(orig_sents, sys_sents, ref_sents, lang=LANGUAGE):
    doc = Doc()
    features = {
        'Compression ratio': get_compression_ratio,
        'Levenshtein similarity': get_levenshtein_similarity,
    }
    with doc.tag('div', klass='row'):
        for feature_name, feature_extractor in features.items():
            with doc.tag('div', klass='col-auto shadow-sm p-0 m-2'):
                figure = get_plotly_histogram(orig_sents, sys_sents, ref_sents, feature_extractor, feature_name, lang)
                doc.asis(get_plotly_html(figure))
    return doc.getvalue()


def get_table_html_from_dataframe(df):
    html = df.to_html(classes='table table-bordered table-responsive table-striped')
    return html.replace('<thead>', '<thead class="thead-light">')


def get_scores_by_length_html(
    orig_sents_tok,
    sys_sents_tok,
    refs_sents_tok,
    orig_sents_nlp,
    sys_sents_nlp,
    refs_sents_nlp,
    n_bins=5,
    lowercase: bool = False,
    tokenizer: str = '13a',
    metrics: List[str] = DEFAULT_METRICS,
    tokenizer_obj=None,
):
    def get_intervals_from_limits(limits):
        return list(zip(limits[:-1], limits[1:]))

    def get_equally_populated_intervals(sents, n_bins):
        sent_lengths = sorted([len(sent) for sent in sents])
        n_samples_per_bin = int(len(sent_lengths) / n_bins)
        limits = [sent_lengths[i * n_samples_per_bin] for i in range(n_bins)] + [sent_lengths[-1] + 1]
        return get_intervals_from_limits(limits)

    def split_sents_by_lengths(sents, intervals):
        bins = [[] for _ in range(len(intervals))]
        for sent_idx, sent in enumerate(sents):
            sent_length = len(sent)
            for interval_idx, (interval_start, interval_end) in enumerate(intervals):
                if interval_start <= sent_length and sent_length < interval_end:
                    bins[interval_idx].append(sent_idx)
                    break
        assert sum([len(b) for b in bins]) == len(sents)
        return bins

    def split_sents_by_bins(sents, bins):
        sents = np.array(sents)
        sents_by_bin = []
        for sent_indexes in bins:
            sents_by_bin.append(np.array(sents)[sent_indexes].tolist())
        return sents_by_bin

    intervals = get_equally_populated_intervals(orig_sents_tok, n_bins)
    bins = split_sents_by_lengths(orig_sents_tok, intervals)
    # Split files by bins
    orig_sents_by_bins_tok = split_sents_by_bins(orig_sents_tok, bins)
    sys_sents_by_bins_tok = split_sents_by_bins(sys_sents_tok, bins)
    refs_sents_by_bins_tok = [split_sents_by_bins(ref_sents, bins) for ref_sents in refs_sents_tok]
    orig_sents_by_bins_nlp = split_sents_by_bins(orig_sents_nlp, bins)
    sys_sents_by_bins_nlp = split_sents_by_bins(sys_sents_nlp, bins)
    refs_sents_by_bins_nlp = [split_sents_by_bins(ref_sents, bins) for ref_sents in refs_sents_nlp]
    # Get scores for each bin
    table = []
    for i in range(len(intervals)):
        interval = intervals[i]
        splitted_orig_sents_tok = orig_sents_by_bins_tok[i]
        splitted_sys_sents_tok = sys_sents_by_bins_tok[i]
        splitted_refs_sents_tok = [ref_sents_by_bins[i] for ref_sents_by_bins in refs_sents_by_bins_tok]
        splitted_orig_sents_nlp = orig_sents_by_bins_nlp[i]
        splitted_sys_sents_nlp = sys_sents_by_bins_nlp[i]
        splitted_refs_sents_nlp = [ref_sents_by_bins[i] for ref_sents_by_bins in refs_sents_by_bins_nlp]
        row = get_all_scores(
            splitted_orig_sents_tok,
            splitted_sys_sents_tok,
            splitted_refs_sents_tok,
            splitted_orig_sents_nlp,
            splitted_sys_sents_nlp,
            splitted_refs_sents_nlp,
            lowercase=lowercase,
            tokenizer=tokenizer,
            metrics=metrics,
            tokenizer_obj=tokenizer_obj,
        )
        row['index'] = f'length=[{interval[0]};{interval[1]}]'
        table.append(row)
    df_bins = pd.DataFrame.from_records(table, index='index')
    return get_table_html_from_dataframe(df_bins.round(2))


def get_head_html():
    solarized_css = '''body{background-color:#fdf6e3}#markdown-body{box-sizing:border-box;min-width:200px;max-width:980px;margin:0 auto;padding:45px;font-family:'Source Sans Pro',sans-serif;font-size:110%;color:#43555a}h1,h2,h3,h4{color:#3e4d52}@media (max-width:767px){.markdown-body{padding:15px}}h2{padding-top:20px!important}a{color:#268bd2;text-decoration:none}a:hover{color:#78b9e6;text-decoration:none;text-shadow:none;border:none}.emph{font-style:italic}.mono{color:#000;font-family:'Source Code Pro',monospace}code,pre{color:#000;font-family:'Source Code Pro',monospace}pre{background:rgba(255,255,255,.12);box-shadow:0 0 10px rgba(0,0,0,.15);padding:10px;width:fit-content}img{background:rgba(255,255,255,.12);box-shadow:0 0 10px rgba(0,0,0,.15);padding:10px}.full{max-width:100%}.full-expanded{max-width:none}.katex{color:#000}.left{text-align:left}p,ul{text-align:justify}'''  # noqa: E501
    return f'''
  <head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

    <!-- Bootstrap -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js" integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1" crossorigin="anonymous"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js" integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM" crossorigin="anonymous"></script>
    <!-- Solarized CSS -->
    <style type="text/css">{solarized_css}</style>
    <!-- Plotly js -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  </head>
'''  # noqa


def get_table_html(header, rows, row_names=None):
    def add_header(doc, header):
        with doc.tag('tr'):
            for value in header:
                doc.line('th', value)

    def add_row(doc, values, row_name=None):
        with doc.tag('tr'):
            if row_name is not None:
                doc.line('th', row_name)
            for value in values:
                doc.line('td', value)

    doc = Doc()
    if row_names is not None:
        header.insert(0, '')
    else:
        row_names = [None] * len(rows)
    with doc.tag('table', klass='table table-bordered table-responsive table-striped'):
        with doc.tag('thead', klass='thead-light'):
            add_header(doc, header)
        with doc.tag('tbody'):
            for row, row_name in zip(rows, row_names):
                add_row(doc, [round(val, 2) for val in row], row_name)
    return doc.getvalue()


def get_score_table_html_single_system(orig_sents_tok, sys_sents_tok, refs_sents_tok, orig_sents_nlp, sys_sents_nlp, refs_sents_nlp, lowercase, tokenizer, metrics, tokenizer_obj):
    return get_score_table_html_multiple_systems(
        orig_sents_tok, [sys_sents_tok], refs_sents_tok, orig_sents_nlp, [sys_sents_nlp], refs_sents_nlp, ['System output'], lowercase, tokenizer, metrics, tokenizer_obj
    )


def get_score_table_html_multiple_systems(
    orig_sents_tok, sys_sents_list_tok, refs_sents_tok, orig_sents_nlp, sys_sents_list_nlp, refs_sents_nlp, system_names, lowercase, tokenizer, metrics, tokenizer_obj,
):
    def truncate(sentence):
        # Take first 80% words
        words = to_words(sentence)
        return ' '.join(words[: int(len(words) * 0.8)]) + '.'
    def truncate_nlp(sentence, tokenizer_obj):
        # Take first 80% words
        words = [tok.text for tok in sentence]
        return tokenizer_obj(' '.join(words[: int(len(words) * 0.8)]) + '.')
    
    doc = Doc()
    # We don't want changes to propagate out of this scope
    sys_sents_list_tok = sys_sents_list_tok.copy()
    sys_sents_list_nlp = sys_sents_list_nlp.copy()
    system_names = system_names.copy()
    # Add the identity baseline
    sys_sents_list_tok.append(orig_sents_tok)
    sys_sents_list_nlp.append(orig_sents_nlp)
    system_names.append('Identity baseline (src2src)')
    # Add tgt2tgt baseline
    sys_sents_list_tok.append(refs_sents_tok[0])
    sys_sents_list_nlp.append(refs_sents_nlp[0])
    system_names.append('Reference baseline (tgt2tgt)')
    # Add the truncate baseline
    sys_sents_list_tok.append([truncate(sentence) for sentence in orig_sents_tok])
    sys_sents_list_nlp.append([truncate_nlp(sentence, tokenizer_obj) for sentence in orig_sents_nlp])
    system_names.append('Truncate baseline')
    # Evaluate systems
    sys_scores_list = []
    t = 0
    for sys_sents_tok, sys_sents_nlp in zip(sys_sents_list_tok, sys_sents_list_nlp): 
        # print(t, "model, src2src, tgt2tgt, trunc")
        t+=1
        sys_scores_list.append(get_all_scores(orig_sents_tok, sys_sents_tok, refs_sents_tok, orig_sents_nlp, sys_sents_nlp, refs_sents_nlp, lowercase=lowercase, tokenizer=tokenizer, metrics=metrics, tokenizer_obj=tokenizer_obj,))
        
    
    rows = [sys_scores.values() for sys_scores in sys_scores_list]
    if len(refs_sents_tok) > 1:
        # Evaluate the first reference against all the others (the second reference is duplicated to have the same number of reference as for systems).
        # TODO: Ideally the system and references should be evaluated with exactly the same number of references.
        ref_scores = get_all_scores(
            orig_sents_tok,
            refs_sents_tok[0],
            [refs_sents_tok[1]] + refs_sents_tok[1:],
            orig_sents_nlp,
            refs_sents_nlp[0],
            [refs_sents_nlp[1]] + refs_sents_nlp[1:],
            lowercase=lowercase,
            tokenizer=tokenizer,
            metrics=metrics,
            tokenizer_obj=tokenizer_obj,
        )
        assert all([sys_scores.keys() == ref_scores.keys() for sys_scores in sys_scores_list])
        rows.append(ref_scores.values())
        system_names.append('Reference*')
    doc.asis(
        get_table_html(
            header=list(sys_scores_list[0].keys()),
            rows=rows,
            row_names=system_names,
        )
    )
    doc.line(
        'p',
        klass='text-muted',
        text_content=(
            '* The Reference row represents one of the references (picked randomly) evaluated' ' against the others.'
        ),
    )
    return doc.getvalue()


def get_html_report(
    orig_sents: List[str],
    sys_sents: List[str],
    refs_sents: List[List[str]],
    test_set: str,
    lowercase: bool = False,
    tokenizer: str = '13a',
    metrics: List[str] = DEFAULT_METRICS,
    tokenizer_obj=None,
	language=LANGUAGE,
):
    # print("sys", sys_sents[:5], "orig", orig_sents[:5], "ref", refs_sents[0][:5])
    doc = Doc()
    doc.asis('<!doctype html>')
    orig_sents_tok = [normalize(sent, lowercase, tokenizer, return_str=True, tokenizer_obj=tokenizer_obj) for sent in orig_sents]
    sys_sents_tok = [normalize(sent, lowercase, tokenizer, return_str=True, tokenizer_obj=tokenizer_obj) for sent in sys_sents]
    refs_sents_tok = [[normalize(sent, lowercase, tokenizer, tokenizer_obj=tokenizer_obj, return_str=True) for sent in ref_sents] for ref_sents in refs_sents]
    orig_sents_nlp = [normalize(sent, lowercase, "spacy", return_obj=True, tokenizer_obj=tokenizer_obj) for sent in orig_sents]
    sys_sents_nlp = [normalize(sent, lowercase, "spacy", return_obj=True, tokenizer_obj=tokenizer_obj) for sent in sys_sents]
    refs_sents_nlp = [[normalize(sent, lowercase, "spacy", tokenizer_obj=tokenizer_obj, return_obj=True) for sent in ref_sents] for ref_sents in refs_sents]

    with doc.tag('html', lang='en'):
        doc.asis(get_head_html())
        with doc.tag('body', klass='container-fluid m-2 mb-5'):
            doc.line('h1', 'EASSE report', klass='mt-4')
            with doc.tag('a', klass='btn btn-link', href='https://forms.gle/J8KVkJsqYe8GvYW46'):
                doc.text('Any feedback welcome!')
            doc.stag('hr')
            doc = add_settings_to_html(doc, lowercase, tokenizer, test_set, metrics)
            doc.line('h2', 'Test set')
            doc.stag('hr')
            with doc.tag('div', klass='container-fluid'):
                doc.asis(
                    get_test_set_description_html(
                        test_set=test_set,
                        orig_sents_tok=orig_sents_tok,
                        refs_sents_tok=refs_sents_tok,
                        orig_sents_nlp=orig_sents_nlp,
                        refs_sents_nlp=refs_sents_nlp,
                        lowercase=lowercase,
                        tokenizer=tokenizer,
                        tokenizer_obj=tokenizer_obj, 
                        lang=language,
                    )
                )
            doc.line('h2', 'Scores')
            doc.stag('hr')
            with doc.tag('div', klass='container-fluid'):
                doc.line('h3', 'System vs. Reference')
                doc.stag('hr')
                doc.asis(
                    get_score_table_html_single_system(orig_sents_tok, sys_sents_tok, refs_sents_tok, orig_sents_nlp, sys_sents_nlp, refs_sents_nlp, lowercase, tokenizer, metrics,
                                                       tokenizer_obj)
                )
                doc.line('h3', 'By sentence length (characters)')
                doc.stag('hr')
                doc.asis(get_scores_by_length_html(orig_sents_tok, sys_sents_tok, refs_sents_tok, orig_sents_nlp, sys_sents_nlp, refs_sents_nlp, tokenizer=tokenizer, metrics=metrics, lowercase=lowercase, tokenizer_obj=tokenizer_obj))
            doc.line('h2', 'Plots')
            doc.stag('hr')
            with doc.tag('div', klass='container-fluid'):
                doc.asis(get_plots_html(orig_sents_nlp, sys_sents_nlp, refs_sents_nlp[0]), language)
            doc.line('h2', 'Qualitative evaluation')
            doc.stag('hr')
            with doc.tag('div', klass='container-fluid'):
                doc.asis(get_qualitative_examples_html(orig_sents_tok, sys_sents_tok, refs_sents_tok, orig_sents_nlp, sys_sents_nlp, refs_sents_nlp, language))
    return indent(doc.getvalue())


def write_html_report(filepath, *args, **kwargs):
    with open(filepath, 'w') as f:
        f.write(get_html_report(*args, **kwargs) + '\n')


def get_multiple_systems_qualitative_examples_html(orig_sents_tok, sys_sents_list_tok, refs_sents_tok, system_names):
    def get_relative_sari(orig_sent, sys_sents, refs_sents, system_idx):
        saris = [corpus_sari([orig_sent], [sys_sent], [[ref] for ref in refs_sents]) for sys_sent in sys_sents]
        return saris[system_idx] / np.average(saris)

    def get_one_sample_html(orig_sent_tok, sys_sents_tok, ref_sents_tok, system_names, sort_key, print_func):
        def get_one_sentence_html(sentence, system_name):
            doc = Doc()
            with doc.tag('div', klass='row'):
                with doc.tag('div', klass='col-2'):
                    doc.text(system_name)
                with doc.tag('div', klass='col'):
                    doc.asis(sentence)
            return doc.getvalue()

        doc = Doc()
        with doc.tag('div', klass='mb-2 p-1'):
            # Sort key
            with doc.tag('div', klass='text-muted small'):
                doc.asis(print_func(sort_key(orig_sent_tok, sys_sents_tok, ref_sents_tok)))
            with doc.tag('div', klass='ml-2'):
                # Source
                with doc.tag('div'):
                    doc.asis(get_one_sentence_html(orig_sent_tok, 'Original'))
                    # Predictions
                    for sys_sent, system_name in zip(sys_sents_tok, system_names):
                        _, sys_sent_bold = make_differing_words_bold(orig_sent_tok, sys_sent, make_text_bold_html)
                        doc.asis(get_one_sentence_html(sys_sent_bold, system_name))
                # References
                collapse_id = get_random_html_id()
                with doc.tag('div', klass='position-relative'):
                    with doc.tag(
                        'a', ('data-toggle', 'collapse'), ('href', f'#{collapse_id}'), klass='stretched-link small'
                    ):
                        doc.text('References')
                    with doc.tag('div', klass='collapse', id=collapse_id):
                        for ref_sent_tok in ref_sents_tok:
                            _, ref_sent_bold = make_differing_words_bold(orig_sent_tok, ref_sent_tok, make_text_bold_html)
                            with doc.tag('div', klass='text-muted'):
                                doc.asis(ref_sent_bold)
        return doc.getvalue()

    title_key_print = [('Randomly sampled simplifications', lambda c, s, refs: 0, lambda value: ''),] + [
        (
            f'Worst relative simplifications (SARI) for {system_names[i]}',
            # lambda c, sys_sents, refs_sents: get_relative_sari(c, sys_sents_tok, refs_sents_tok, system_idx=i),
            lambda c, sys_sents, refs_sents: get_relative_sari(c, sys_sents, refs_sents, system_idx=i),
            lambda value: f'Relative SARI={value:.2f}',
        )
        for i in range(len(system_names))
    ]
    doc = Doc()
    for title, sort_key, print_func in title_key_print:
        with doc.tag('div', klass='container-fluid mt-4 p-2 border'):
            collapse_id = get_random_html_id()
            with doc.tag('a', ('data-toggle', 'collapse'), ('href', f'#{collapse_id}')):
                doc.line('h3', klass='m-2', text_content=title)
            # Now lets print the examples
            sample_generator = sorted(
                zip(orig_sents_tok, zip(*sys_sents_list_tok), zip(*refs_sents_tok)),
                key=lambda args: sort_key(*args),
            )
            # Samples displayed by default
            with doc.tag('div', klass='collapse', id=collapse_id):
                n_samples = 50
                for i, (orig_sent_tok, sys_sents_tok, refs_tok) in enumerate(sample_generator):
                    if i >= n_samples:
                        break
                    doc.asis(get_one_sample_html(orig_sent_tok, sys_sents_tok, refs_tok, system_names, sort_key, print_func))
    return doc.getvalue()


def get_multiple_systems_html_report(
    orig_sents, sys_sents_list, refs_sents, system_names, test_set, lowercase, tokenizer, metrics, tokenizer_obj,language,
):
    orig_sents_tok = [normalize(sent, lowercase, tokenizer, return_str=True, tokenizer_obj=tokenizer_obj) for sent in orig_sents]
    sys_sents_list_tok = [[normalize(sent, lowercase, tokenizer, tokenizer_obj=tokenizer_obj, return_str=True) for sent in sys_sents] for sys_sents in sys_sents_list]
    refs_sents_tok = [[normalize(sent, lowercase, tokenizer, tokenizer_obj=tokenizer_obj, return_str=True) for sent in ref_sents] for ref_sents in refs_sents]
    orig_sents_nlp = [normalize(sent, lowercase, "spacy", return_obj=True, tokenizer_obj=tokenizer_obj) for sent in orig_sents]
    refs_sents_nlp = [[normalize(sent, lowercase, "spacy", tokenizer_obj=tokenizer_obj, return_obj=True) for sent in ref_sents] for ref_sents in refs_sents]
    sys_sents_list_nlp = [[normalize(sent, lowercase, "spacy", tokenizer_obj=tokenizer_obj, return_obj=True) for sent in sys_sents] for sys_sents in sys_sents_list]
    doc = Doc()
    doc.asis('<!doctype html>')
    with doc.tag('html', lang='en'):
        doc.asis(get_head_html())
        with doc.tag('body', klass='container-fluid m-2 mb-5'):
            doc.line('h1', 'EASSE report', klass='mt-4')
            with doc.tag('a', klass='btn btn-link', href='https://forms.gle/J8KVkJsqYe8GvYW46'):
                doc.text('Any feedback welcome!')
            doc.stag('hr')
            doc = add_settings_to_html(doc, lowercase, tokenizer, test_set, metrics)
            doc.line('h2', 'Test set')
            doc.stag('hr')
            with doc.tag('div', klass='container-fluid'):
                doc.asis(
                    get_test_set_description_html(
                        test_set=test_set,
                        orig_sents_tok=orig_sents_tok,
                        refs_sents_tok=refs_sents_tok,
                        orig_sents_nlp=orig_sents_nlp,
                        refs_sents_nlp=refs_sents_nlp,
                        lowercase=lowercase,
                        tokenizer=tokenizer,
                        tokenizer_obj=tokenizer_obj, 
                        lang=language,
                    )
                )
            doc.line('h2', 'Scores')
            doc.stag('hr')
            with doc.tag('div', klass='container-fluid'):
                doc.line('h3', 'System vs. Reference')
                doc.stag('hr')
                # todo specify orig version here
                doc.asis(
                    get_score_table_html_multiple_systems(
                        orig_sents_tok, sys_sents_list_tok, refs_sents_tok, orig_sents_nlp, sys_sents_list_nlp, refs_sents_nlp, system_names, lowercase, tokenizer, metrics, tokenizer_obj,
                    )
                )
            doc.line('h2', 'Qualitative evaluation')
            doc.stag('hr')
            with doc.tag('div', klass='container-fluid'):
                # todo specify orig version here
                doc.asis(
                    get_multiple_systems_qualitative_examples_html(orig_sents_tok, sys_sents_list_tok, refs_sents_tok, system_names)
                )
    # return indent(doc.getvalue())
    return doc.getvalue()


def write_multiple_systems_html_report(filepath, *args, **kwargs):
    with open(filepath, 'w') as f:
        f.write(get_multiple_systems_html_report(*args, **kwargs) + '\n')


def add_settings_to_html(doc, lowercase, tokenizer, test_set, metrics):
    doc.line('h2', 'Settings:')
    with doc.tag('div', klass='container-fluid'):
        with doc.tag('ul', id='settings-list'):
            doc.line('li', 'lowercasing: ' + str(lowercase))
            doc.line('li', 'tokenizer: ' + tokenizer)
            doc.line('li', 'test set: ' + test_set)
            doc.line('li', 'metrics: ' + ",".join(metrics))
            doc.line('li', 'language:'+ LANGUAGE)
    doc.stag('hr')
    return doc
