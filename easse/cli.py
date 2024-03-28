from pathlib import Path

import click
import spacy

from easse.bleu import corpus_bleu, corpus_averaged_sentence_bleu
from easse.compression import corpus_f1_token
from easse.fkgl import corpus_fkgl
from easse.quality_estimation import corpus_quality_estimation
from easse.report import write_html_report, write_multiple_systems_html_report
from easse.sari import corpus_sari, get_corpus_sari_operation_scores
# from easse.dsari import corpus_document_sari, get_corpus_document_sari_operation_scores
# from easse.rouge import corpus_rouge
# from easse.bartscore import corpus_bartscore
from easse.textstat_metrics import (corpus_averaged_sentence_fre, corpus_fre,
                                    corpus_wiener_1, corpus_wiener_2, corpus_wiener_3, corpus_wiener_4,
                                    sent_wiener_1, sent_wiener_2, sent_wiener_3, sent_wiener_4, 
                                    corpus_averaged_sentence_osman, corpus_osman,
                                    corpus_averaged_sentence_fernandez_huerta, corpus_fernandez_huerta,
                                    corpus_averaged_sentence_szigriszt_pazos, corpus_szigriszt_pazos,
                                    corpus_averaged_sentence_gutierrez_polini, corpus_gutierrez_polini,
                                    corpus_averaged_sentence_crawford, corpus_crawford,
                                    corpus_averaged_sentence_gulpease_index, corpus_gulpease_index)
from easse.utils.constants import (
    VALID_TEST_SETS_SENTENCE_LEVEL,
    VALID_TEST_SETS_DOCUMENT_LEVEL,
    VALID_METRICS,
    DEFAULT_METRICS,
    SPACY_MODEL_NAME,
    LANGUAGE,
)
from easse.utils.helpers import read_lines
from easse.utils.resources import get_orig_sents, get_refs_sents
from easse.utils.preprocessing import normalize


def check_testset_for_textlevel(testset, input_level):
    if input_level == "sentence-level":
        if testset in VALID_TEST_SETS_SENTENCE_LEVEL:
            return True
        else:
            raise ValueError("Invalid value for '--test_set' / '-t': "+testset+" is not one of the valid testsets of sentence-level: "+", ".join(VALID_TEST_SETS_SENTENCE_LEVEL))
    elif input_level == "document-level":
        if testset in VALID_TEST_SETS_DOCUMENT_LEVEL:
            return True
        else:
            raise ValueError("Invalid value for '--test_set' / '-t': "+testset+" is not one of the valid testsets of document-level: "+", ".join(VALID_TEST_SETS_DOCUMENT_LEVEL))
    raise ValueError("Wrong input level, please choose either document or sentence level.")


def get_sys_sents(test_set, sys_sents_path=None):
    # Get system sentences to be evaluated
    print(sys_sents_path)
    if sys_sents_path is not None:
        return read_lines(sys_sents_path)
    else:
        # read the system output
        with click.get_text_stream("stdin", encoding="utf-8") as system_output_file:
            return system_output_file.read().splitlines()


def get_orig_and_refs_sents(test_set, orig_sents_path=None, refs_sents_paths=None, input_level="sentence-level"):
    # Get original and reference sentences
    if test_set == "custom":
        assert orig_sents_path is not None
        assert refs_sents_paths is not None
        print("orig", orig_sents_path, "ref", refs_sents_paths)
        if type(refs_sents_paths) == str:
            refs_sents_paths = refs_sents_paths.split(",")
        orig_sents = read_lines(orig_sents_path)
        refs_sents = [read_lines(ref_sents_path) for ref_sents_path in refs_sents_paths]
    else:
        orig_sents = get_orig_sents(test_set, input_level)
        refs_sents = get_refs_sents(test_set, input_level)
    # Final checks
    assert all(
        [len(orig_sents) == len(ref_sents) for ref_sents in refs_sents]
    ), f'Not same number of lines for test_set={test_set}, orig_sents_path={orig_sents_path}, refs_sents_paths={refs_sents_paths}'  # noqa: E501
    return orig_sents, refs_sents


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option()
def cli():
    pass

def common_options(function):
    function = click.option(
        "--test_set",
        "-t",
        type=click.Choice(VALID_TEST_SETS_SENTENCE_LEVEL + VALID_TEST_SETS_DOCUMENT_LEVEL),
        required=True,
        help="Test set to use.",
    )(function)
    function = click.option(
        "--orig_sents_path",
        type=click.Path(),
        default=None,
        help='Path to the source sentences. Only used when test_set == "custom".',
    )(function)
    function = click.option(
        "--refs_sents_paths",
        type=str,
        default=None,
        help='Comma-separated list of path(s) to the references(s). Only used when test_set == "custom".',
    )(function)
    function = click.option(
        "--lowercase/--no-lowercase",
        "-lc/--no-lc",
        default=True,
        help="Compute case-sensitive scores for all metrics. ",
    )(function)
    function = click.option(
        "--tokenizer",
        "-tok",
        type=click.Choice(["13a", "intl", "moses", "penn", "spacy", "none"]),
        default="spacy",
        help="Tokenization method to use.",
    )(function)
    function = click.option(
        "--metrics",
        "-m",
        type=str,
        default=",".join(DEFAULT_METRICS),
        help=(
            f'Comma-separated list of metrics to compute. Valid: {",".join(VALID_METRICS)}'
            " (SAMSA is disabled by default for the sake of speed)."
        ),
    )(function)
    function = click.option(
        "--input_level",
        "-il",
        type=click.Choice(["sentence-level", "document-level"]),
        default="sentence-level",
        help='Level of the input data, whether document-level or sentence-level.',
    )(function)
    function = click.option(
        "--language",
        "-lang",
        type=str,
        default=LANGUAGE,
        help='Language of the data.',
    )(function)
    return function


@cli.command("evaluate")
@common_options
@click.option(
    "--analysis",
    "-a",
    is_flag=True,
    help=f"Perform word-level transformation analysis.",
)
@click.option(
    "--quality_estimation",
    "-q",
    is_flag=True,
    help="Compute quality estimation features.",
)
@click.option(
    "--sys_sents_path",
    "-i",
    type=click.Path(),
    default=None,
    help="Path to the system predictions input file that is to be evaluated.",
)
def _evaluate_system_output(*args, **kwargs):
    check_testset_for_textlevel(kwargs["test_set"], kwargs["input_level"])
    kwargs["metrics"] = kwargs.pop("metrics").split(",")
    metrics_scores = evaluate_system_output(*args, **kwargs)

    def recursive_round(obj):
        def is_castable_to_float(obj):
            try:
                float(obj)
            except (ValueError, TypeError):
                return False
            return True

        if is_castable_to_float(obj):
            return round(obj, 3)
        if type(obj) is dict:
            return {key: recursive_round(value) for key, value in obj.items()}
        return obj

    print(recursive_round(metrics_scores))


def evaluate_system_output(
    test_set,
    sys_sents_path=None,
    orig_sents_path=None,
    refs_sents_paths=None,
    tokenizer="13a",
    tokenizer_obj=None,
    lowercase=True,
    metrics=DEFAULT_METRICS,
    analysis=False,
    quality_estimation=False,
    input_level="sentence-level",
    language=LANGUAGE,
):
    """
    Evaluate a system output with automatic metrics.
    """
    for metric in metrics:
        assert metric in VALID_METRICS, f'"{metric}" is not a valid metric. Choose among: {VALID_METRICS}'
    sys_sents = get_sys_sents(test_set, sys_sents_path)
    orig_sents, refs_sents = get_orig_and_refs_sents(test_set, orig_sents_path, refs_sents_paths, input_level)
    # if tokenizer == "spacy":
    if not spacy.util.is_package(language + SPACY_MODEL_NAME):
        print(
            "Please download a model of language corresponding to your language manually with ```python -m spacy download <modelname>```")
    print("LOADING Spacy model.")
    nlp = spacy.load(language + SPACY_MODEL_NAME)
    tokenizer_obj = nlp
    print("Finished loading Spacy model.")
    # else:
    # tokenizer_obj = None

    orig_sents_tok = [normalize(sent, lowercase, tokenizer, tokenizer_obj=tokenizer_obj, return_str=True) for sent in orig_sents]
    # ref_sents_tok = [normalize(sent, lowercase, tokenizer, tokenizer_obj=tokenizer_obj, return_str=True) for sent in refs_sents]
    refs_sents_tok = [[normalize(sent, lowercase, tokenizer, tokenizer_obj=tokenizer_obj, return_str=True) for sent in ref_sents] for ref_sents in refs_sents]
    sys_sents_tok = [normalize(sent, lowercase, tokenizer, tokenizer_obj=tokenizer_obj, return_str=True) for sent in sys_sents]
    orig_sents_nlp = [normalize(sent, lowercase, "spacy", tokenizer_obj=tokenizer_obj, return_str=True, return_obj=True) for sent in orig_sents]
    # ref_sents_nlp = [normalize(sent, lowercase, tokenizer, tokenizer_obj=tokenizer_obj, return_str=False) for sent in refs_sents]
    refs_sents_nlp = [[normalize(sent, lowercase, "spacy", tokenizer_obj=tokenizer_obj, return_str=True, return_obj=True) for sent in ref_sents] for ref_sents in refs_sents]
    sys_sents_nlp = [normalize(sent, lowercase, "spacy", tokenizer_obj=tokenizer_obj, return_str=True, return_obj=True) for sent in sys_sents]

    # compute each metric
    metrics_scores = {}
    if "bleu" in metrics:
        metrics_scores["bleu"] = corpus_bleu(
            sys_sents_tok,
            refs_sents_tok,
            force=True,
            # tokenizer=tokenizer,
            # lowercase=lowercase,
            # tokenizer_obj=tokenizer_obj,
        )

    if "sent_bleu" in metrics:
        metrics_scores["sent_bleu"] = corpus_averaged_sentence_bleu(
            sys_sents_tok, refs_sents_tok,  # tokenizer=tokenizer, lowercase=lowercase, tokenizer_obj=tokenizer_obj,
        )

    if "sari" in metrics:
        metrics_scores["sari"] = corpus_sari(
            orig_sents,
            sys_sents_tok,
            refs_sents_tok,
            tokenizer=tokenizer,
            lowercase=lowercase,
            tokenizer_obj=tokenizer_obj,
        )

    if "sari_legacy" in metrics:
        metrics_scores["sari_legacy"] = corpus_sari(
            orig_sents,
            sys_sents_tok,
            refs_sents_tok,
            tokenizer=tokenizer,
            lowercase=lowercase,
            legacy=True,
            tokenizer_obj=tokenizer_obj,
        )

    if "sari_by_operation" in metrics:
        (
            metrics_scores["sari_add"],
            metrics_scores["sari_keep"],
            metrics_scores["sari_del"],
        ) = get_corpus_sari_operation_scores(
            orig_sents_tok,
            sys_sents_tok,
            refs_sents_tok,
            tokenizer=tokenizer,
            lowercase=lowercase,
            tokenizer_obj=tokenizer_obj,
        )

    # if "dsari_by_operation" in metrics:
    #     (
    #         metrics_scores["dsari_add"],
    #         metrics_scores["dsari_keep"],
    #         metrics_scores["dsari_del"],
    #     ) = get_corpus_dsari_operation_scores(
    #         orig_sents,
    #         sys_sents,
    #         refs_sents,
    #         tokenizer=tokenizer,
    #         lowercase=lowercase,
    #         tokenizer_obj=tokenizer_obj,
    #     )
    #
    # if "dsari" in metrics:
    #     metrics_scores["dsari"] = corpus_dsari(
    #         orig_sents,
    #         sys_sents,
    #         refs_sents,
    #         tokenizer=tokenizer,
    #         lowercase=lowercase,
    #         tokenizer_obj=tokenizer_obj,
    #     )

    if "samsa" in metrics:
        from easse.samsa import corpus_samsa

        metrics_scores["samsa"] = corpus_samsa(
            orig_sents_tok,
            sys_sents_tok,
            refs_sents_tok,
            # tokenizer=tokenizer,
            # lowercase=lowercase,
            # tokenizer_obj=tokenizer_obj,
            verbose=True,
        )

    if "fkgl" in metrics:
        metrics_scores["fkgl"] = corpus_fkgl(sys_sents_tok, 
                                             # tokenizer=tokenizer, 
                                             # lowercase=lowercase, 
                                             # tokenizer_obj=tokenizer_obj,
                                            )

    if "f1_token" in metrics:
        metrics_scores["f1_token"] = corpus_f1_token(sys_sents_tok, 
                                                     refs_sents_tok, 
                                                     # tokenizer=tokenizer, 
                                                     # lowercase=lowercase,
                                                     # tokenizer_obj=tokenizer_obj,
                                                    )

    # if "rouge" in metrics:
    #     metrics_scores["rouge"] = corpus_rouge([sys_sents], refs_sents, tokenizer=tokenizer, lowercase=lowercase, tokenizer_obj=tokenizer_obj,
    #                   language=language)

    if "bertscore" in metrics:
        from easse.bertscore import corpus_bertscore  # Inline import to use EASSE without installing all dependencies

        (
            metrics_scores["bertscore_precision"],
            metrics_scores["bertscore_recall"],
            metrics_scores["bertscore_f1"],
        ) = corpus_bertscore(sys_sents_tok, 
                             refs_sents_tok, 
                             # tokenizer=tokenizer, 
                             # lowercase=lowercase,
                             # tokenizer_obj=tokenizer_obj,
                             language=language)
    # if "bartscore" in metrics:
    #     from easse.bartscore import corpus_bartscore  # Inline import to use EASSE without installing all dependencies
    #
    #     (
    #         metrics_scores["bartscore_precision"],
    #         metrics_scores["bartscore_recall"],
    #         metrics_scores["bartscore_f1"],
    #     ) = corpus_bartscore(orig_sents, sys_sents, refs_sents, tokenizer=tokenizer, lowercase=lowercase, tokenizer_obj=tokenizer_obj,
    #                          language=language)

    # if 'fre_sent' in metrics:
    #     metrics_scores["sent_FRE"] = corpus_averaged_sentence_fre(sys_sents, tokenizer=tokenizer, lowercase=lowercase, language=language, tokenizer_obj=tokenizer_obj,)
    if 'fre' in metrics:
        metrics_scores["FRE"] = corpus_fre(sys_sents_tok, 
                                           # tokenizer=tokenizer, 
                                           # lowercase=lowercase, 
                                           language=language, 
                                           # tokenizer_obj=tokenizer_obj,
                                          )
    # if 'wiener_sachtextformel_1_sent' in metrics:
    #     metrics_scores["Wiener-Sachtextformel-1-sent"] = sent_wiener_1(sys_sents, tokenizer=tokenizer, lowercase=lowercase,                          language=language,tokenizer_obj=tokenizer_obj,)
    # if 'wiener_sachtextformel_2_sent' in metrics:
    #     metrics_scores["Wiener-Sachtextformel-2-sent"] = sent_wiener_2(sys_sents, tokenizer=tokenizer, lowercase=lowercase,                          language=language,tokenizer_obj=tokenizer_obj,)
    # if 'wiener_sachtextformel_3_sent' in metrics:
    #     metrics_scores["Wiener-Sachtextformel-3-sent"] = sent_wiener_3(sys_sents, tokenizer=tokenizer, lowercase=lowercase,                          language=language,tokenizer_obj=tokenizer_obj,)
    # if 'wiener_sachtextformel_4_sent' in metrics:
    #     metrics_scores["Wiener-Sachtextformel-4-sent"] = sent_wiener_4(sys_sents, tokenizer=tokenizer, lowercase=lowercase,                          language=language,tokenizer_obj=tokenizer_obj,)
    if 'wiener_sachtextformel_1' in metrics:
        metrics_scores["Wiener-Sachtextformel-1"] = corpus_wiener_1(sys_sents_tok, 
                                           # tokenizer=tokenizer, 
                                           # lowercase=lowercase, 
                                           language=language, 
                                           # tokenizer_obj=tokenizer_obj,
                                          )
    if 'wiener_sachtextformel_2' in metrics:
        metrics_scores["Wiener-Sachtextformel-2"] = corpus_wiener_2(sys_sents_tok, 
                                           # tokenizer=tokenizer, 
                                           # lowercase=lowercase, 
                                           language=language, 
                                           # tokenizer_obj=tokenizer_obj,
                                          )
    if 'wiener_sachtextformel_3_corpus' in metrics:
        metrics_scores["Wiener-Sachtextformel-3"] = corpus_wiener_3(sys_sents_tok, 
                                           # tokenizer=tokenizer, 
                                           # lowercase=lowercase, 
                                           language=language, 
                                           # tokenizer_obj=tokenizer_obj,
                                          )
    if 'wiener_sachtextformel_4_corpus' in metrics:
        metrics_scores["Wiener-Sachtextformel-4"] = corpus_wiener_4(sys_sents_tok, 
                                           # tokenizer=tokenizer, 
                                           # lowercase=lowercase, 
                                           language=language, 
                                           # tokenizer_obj=tokenizer_obj,
                                          )
    # if 'osman_sent' in metrics:
    #     metrics_scores["sent_osman"] = corpus_averaged_sentence_osman(sys_sents, tokenizer=tokenizer, lowercase=lowercase, language=language, tokenizer_obj=tokenizer_obj,)
    if 'osman' in metrics:
        metrics_scores["osman"] = corpus_osman(sys_sents_tok, 
                                           # tokenizer=tokenizer, 
                                           # lowercase=lowercase, 
                                           language=language, 
                                           # tokenizer_obj=tokenizer_obj,
                                          )
    # if 'fernandez_huerta_sent' in metrics:
    #     metrics_scores["sent_fernandez_huerta"] = corpus_averaged_sentence_fernandez_huerta(sys_sents, tokenizer=tokenizer, lowercase=lowercase, language=language, tokenizer_obj=tokenizer_obj,)
    if 'fernandez_huerta' in metrics:
        metrics_scores["fernandez_huerta"] = corpus_fernandez_huerta(sys_sents_tok, 
                                           # tokenizer=tokenizer, 
                                           # lowercase=lowercase, 
                                           language=language, 
                                           # tokenizer_obj=tokenizer_obj,
                                          )
    # if 'szigriszt_pazos_sent' in metrics:
    #     metrics_scores["sent_szigriszt_pazos"] = corpus_averaged_sentence_szigriszt_pazos(sys_sents, tokenizer=tokenizer, lowercase=lowercase, language=language, tokenizer_obj=tokenizer_obj,)
    if 'szigriszt_pazos' in metrics:
        metrics_scores["szigriszt_pazos"] = corpus_szigriszt_pazos(sys_sents_tok, 
                                           # tokenizer=tokenizer, 
                                           # lowercase=lowercase, 
                                           language=language, 
                                           # tokenizer_obj=tokenizer_obj,
                                          )
    # if 'gutierrez_polini_sent' in metrics:
    #     metrics_scores["sent_gutierrez_polini"] = corpus_averaged_sentence_gutierrez_polini(sys_sents, tokenizer=tokenizer, lowercase=lowercase, language=language, tokenizer_obj=tokenizer_obj,)
    if 'gutierrez_polini' in metrics:
        metrics_scores["gutierrez_polini"] = corpus_gutierrez_polini(sys_sents_tok, 
                                           # tokenizer=tokenizer, 
                                           # lowercase=lowercase, 
                                           language=language, 
                                           # tokenizer_obj=tokenizer_obj,
                                          )
    # if 'crawford_sent' in metrics:
    #     metrics_scores["sent_crawford"] = corpus_averaged_sentence_crawford(sys_sents, tokenizer=tokenizer, lowercase=lowercase, language=language, tokenizer_obj=tokenizer_obj,)
    if 'crawford' in metrics:
        metrics_scores["crawford"] = corpus_crawford(sys_sents_tok, 
                                           # tokenizer=tokenizer, 
                                           # lowercase=lowercase, 
                                           language=language, 
                                           # tokenizer_obj=tokenizer_obj,
                                          )
    # if 'gulpease_index_sent' in metrics:
    #     metrics_scores["sent_gulpease_index"] = corpus_averaged_sentence_gulpease_index(sys_sents, tokenizer=tokenizer, lowercase=lowercase, language=language, tokenizer_obj=tokenizer_obj,)
    if 'gulpease_index' in metrics:
        metrics_scores["gulpease_index"] = corpus_gulpease_index(sys_sents_tok, 
                                           language=language,
                                           # tokenizer=tokenizer, 
                                           # lowercase=lowercase,  
                                           # tokenizer_obj=tokenizer_obj,
                                          )
    if analysis:
        from easse.annotation.word_level import (
            WordOperationAnnotator,
        )  # Inline import to use EASSE without installing all dependencies

        word_operation_annotator = WordOperationAnnotator(tokenizer=tokenizer, lowercase=lowercase, verbose=True, tokenizer_obj=tokenizer_obj)
        metrics_scores["word_level_analysis"] = word_operation_annotator.analyse_operations(
            orig_sents, sys_sents, refs_sents, as_str=True
        )

    if quality_estimation:
        print(orig_sents_nlp[0], orig_sents_tok[0])
        metrics_scores["quality_estimation"] = corpus_quality_estimation(
            orig_sents_nlp, sys_sents_nlp, tokenizer=tokenizer, lowercase=lowercase, tokenizer_obj=tokenizer_obj,
        )

    return metrics_scores


@cli.command("report")
@common_options
@click.option(
    "--sys_sents_path",
    "-i",
    type=click.Path(),
    default=None,
    help="""Path to the system predictions input file that is to be evaluated.
              You can also input a comma-separated list of files to compare multiple systems.""",
)
@click.option(
    "--report_path",
    "-p",
    type=click.Path(),
    default="easse_report.html",
    help="Path to the output HTML report.",
)
def _report(*args, **kwargs):
    check_testset_for_textlevel(kwargs["test_set"], kwargs["input_level"])
    kwargs["metrics"] = kwargs.pop("metrics").split(",")
    if kwargs["sys_sents_path"] is not None and len(kwargs["sys_sents_path"].split(",")) > 1:
        # If we got multiple systems as input, split the paths and rename the key
        kwargs["sys_sents_paths"] = kwargs.pop("sys_sents_path").split(",")
        multiple_systems_report(*args, **kwargs)
    else:
        report(*args, **kwargs)


def report(
    test_set,
    sys_sents_path=None,
    orig_sents_path=None,
    refs_sents_paths=None,
    report_path="easse_report.html",
    tokenizer="13a",
    lowercase=True,
    metrics=DEFAULT_METRICS,
    input_level="sentence-level",
    tokenizer_obj=None,
    language=LANGUAGE,
):
    """
    Create a HTML report file with automatic metrics, plots and samples.
    """
    # if tokenizer == "spacy":
    if not spacy.util.is_package(language + SPACY_MODEL_NAME):
        print("Please download a model of language corresponding to your language manually with ```python -m spacy download <modelname>```")
    print("LOADING Spacy model.")
    nlp = spacy.load(language + SPACY_MODEL_NAME)
    tokenizer_obj = nlp
    print("Finished loading Spacy model.")
    # else:
    #     tokenizer_obj = None
    sys_sents = get_sys_sents(test_set, sys_sents_path)
    orig_sents, refs_sents = get_orig_and_refs_sents(test_set, orig_sents_path, refs_sents_paths, input_level)
    write_html_report(
        report_path,
        orig_sents,
        sys_sents,
        refs_sents,
        test_set=test_set,
        lowercase=lowercase,
        tokenizer=tokenizer,
        metrics=metrics,
        tokenizer_obj=tokenizer_obj,
		language=LANGUAGE,
    )


def multiple_systems_report(
    test_set,
    sys_sents_paths,
    orig_sents_path=None,
    refs_sents_paths=None,
    report_path="easse_report.html",
    tokenizer="13a",
    lowercase=True,
    metrics=DEFAULT_METRICS,
    system_names=None,
    input_level="sentence-level",
    tokenizer_obj=None,
    language=LANGUAGE,
):    
    # if tokenizer == "spacy":
    if not spacy.util.is_package(language + SPACY_MODEL_NAME):
        print("Please download a model of language corresponding to your language manually with ```python -m spacy download <modelname>```")
    print("LOADING Spacy model.")
    nlp = spacy.load(language + SPACY_MODEL_NAME)
    tokenizer_obj = nlp
    print("Finished loading Spacy model.")
    # else:
    #     tokenizer_obj = None
    """
    Create a HTML report file comparing multiple systems with automatic metrics, plots and samples.
    """
    sys_sents_list = [read_lines(path) for path in sys_sents_paths]
    orig_sents, refs_sents = get_orig_and_refs_sents(test_set, orig_sents_path, refs_sents_paths, input_level)
    if system_names is None:
        system_names = [Path(path).name for path in sys_sents_paths]
    write_multiple_systems_html_report(
        report_path,
        orig_sents,
        sys_sents_list,
        refs_sents,
        system_names=system_names,
        test_set=test_set,
        lowercase=lowercase,
        tokenizer=tokenizer,
        metrics=metrics,
        tokenizer_obj=tokenizer_obj,
        language=LANGUAGE,
    )
