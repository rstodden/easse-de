from pathlib import Path


# Paths
PACKAGE_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = PACKAGE_DIR / "resources"
TOOLS_DIR = RESOURCES_DIR / "tools"
DATA_DIR = RESOURCES_DIR / "data"
DATA_DIR_TEST_PATH_SENTENCE_LEVEL = DATA_DIR / "test_sets/sentence_level"
DATA_DIR_TEST_PATH_DOCUMENT_LEVEL = DATA_DIR / "test_sets/document_level"

STANFORD_CORENLP_DIR = TOOLS_DIR / "stanford-corenlp-full-2018-10-05"
UCCA_DIR = TOOLS_DIR / "ucca-bilstm-1.3.10"
UCCA_PARSER_PATH = UCCA_DIR / "models/ucca-bilstm"
TEST_SETS_PATHS_DOCUMENT_LEVEL = {
    ('20Minuten_test', 'orig'): DATA_DIR_TEST_PATH_DOCUMENT_LEVEL / f'20Minuten/20Minuten_test.src.txt',
    ('20Minuten_test', 'refs'): [DATA_DIR_TEST_PATH_DOCUMENT_LEVEL / f'20Minuten/20Minuten_test.tgt.txt'],
    ('20Minuten_valid', 'orig'): DATA_DIR_TEST_PATH_DOCUMENT_LEVEL / f'20Minuten/20Minuten_dev.src.txt',
    ('20Minuten_valid', 'refs'): [DATA_DIR_TEST_PATH_DOCUMENT_LEVEL / f'20Minuten/20Minuten_dev.tgt.txt'],
    ('klexikon_test', 'orig'): DATA_DIR_TEST_PATH_DOCUMENT_LEVEL / f'klexikon/klexikon_test.src.txt',
    ('klexikon_test', 'refs'): [DATA_DIR_TEST_PATH_DOCUMENT_LEVEL / f'klexikon/klexikon_test.tgt.txt'],
    ('klexikon_valid', 'orig'): DATA_DIR_TEST_PATH_DOCUMENT_LEVEL / f'klexikon/klexikon_validation.src.txt',
    ('klexikon_valid', 'refs'): [DATA_DIR_TEST_PATH_DOCUMENT_LEVEL / f'klexikon/klexikon_validation.tgt.txt'],
    ('hda_easy_to_read_langauge_test', 'orig'): DATA_DIR_TEST_PATH_DOCUMENT_LEVEL / f'hda_easy_to_read_langauge/hda_easy_to_read_langauge_test.src.txt',
    ('hda_easy_to_read_langauge_test', 'refs'): [DATA_DIR_TEST_PATH_DOCUMENT_LEVEL / f'hda_easy_to_read_langauge/hda_easy_to_read_langauge_test.tgt.txt'],
    # hda_easy_to_read_langauge_test.src
}

SYSTEM_OUTPUTS_DIR = DATA_DIR / "system_outputs"
SYSTEM_OUTPUTS_DIR_DOCUMENT_LEVEL = SYSTEM_OUTPUTS_DIR / "document_level"
SYSTEM_OUTPUTS_DIRS_MAP_DOCUMENT_LEVEL = {
    "zest_test": SYSTEM_OUTPUTS_DIR_DOCUMENT_LEVEL / "klexikon/test",
}

TEST_SETS_PATHS_SENTENCE_LEVEL = {
    ('zest_test', 'orig'): DATA_DIR_TEST_PATH_SENTENCE_LEVEL / f'ZEST/geolino.test.src',
    ('zest_test', 'refs'): [DATA_DIR_TEST_PATH_SENTENCE_LEVEL / f'ZEST/geolino.test.tgt'],
    ('zest_valid', 'orig'): DATA_DIR_TEST_PATH_SENTENCE_LEVEL / f'ZEST/geolino.valid.src',
    ('zest_valid', 'refs'): [DATA_DIR_TEST_PATH_SENTENCE_LEVEL / f'ZEST/geolino.valid.tgt'],
    ('textcomplexityde_test', 'orig'): DATA_DIR_TEST_PATH_SENTENCE_LEVEL / f'TextComplexityDE/TextComplexityDE_test.src.txt',
    ('textcomplexityde_test', 'refs'): [DATA_DIR_TEST_PATH_SENTENCE_LEVEL / f'TextComplexityDE/TextComplexityDE_test.tgt.txt'],
    ('apa_lha-or-a2_test', 'orig'): DATA_DIR_TEST_PATH_SENTENCE_LEVEL / f'APA_LHAor-a2/APA_LHAor-a2_test.tok.src.txt',
    ('apa_lha-or-a2_test', 'refs'): [DATA_DIR_TEST_PATH_SENTENCE_LEVEL / f'APA_LHAor-a2/APA_LHAor-a2_test.tok.tgt.txt'],
    ('apa_lha-or-a2_valid', 'orig'): DATA_DIR_TEST_PATH_SENTENCE_LEVEL / f'APA_LHAor-a2/APA_LHAor-a2_dev.tok.src.txt',
    ('apa_lha-or-a2_valid', 'refs'): [DATA_DIR_TEST_PATH_SENTENCE_LEVEL / f'APA_LHAor-a2/APA_LHAor-a2_dev.tok.tgt.txt'],
    ('apa_lha-or-b1_test', 'orig'): DATA_DIR_TEST_PATH_SENTENCE_LEVEL / f'APA_LHAor-b1/APA_LHAor-b1_test.tok.src.txt',
    ('apa_lha-or-b1_test', 'refs'): [DATA_DIR_TEST_PATH_SENTENCE_LEVEL / f'APA_LHAor-b1/APA_LHAor-b1_test.tok.tgt.txt'],
    ('apa_lha-or-b1_valid', 'orig'): DATA_DIR_TEST_PATH_SENTENCE_LEVEL / f'APA_LHAor-b1/APA_LHAor-b1_dev.tok.src.txt',
    ('apa_lha-or-b1_valid', 'refs'): [DATA_DIR_TEST_PATH_SENTENCE_LEVEL / f'APA_LHAor-b1/APA_LHAor-b1_dev.tok.tgt.txt'],
    ('deasy-share_test', 'orig'): DATA_DIR_TEST_PATH_SENTENCE_LEVEL / f'deasy-share/DEASY-share_test.tok.src.txt',
    ('deasy-share_test', 'refs'): [DATA_DIR_TEST_PATH_SENTENCE_LEVEL / f'deasy-share/DEASY-share_test.tgt.txt'],
    # ('deasy-share_valid', 'orig'): DATA_DIR_TEST_PATH_SENTENCE_LEVEL / f'deasy-share/DEASY-share_dev.src.txt',
    # ('deasy-share_valid', 'refs'): [DATA_DIR_TEST_PATH_SENTENCE_LEVEL / f'deasy-share/DEASY-share_dev.tgt.txt'],
    ('turkcorpus_test', 'orig'): DATA_DIR / f'test_sets/turkcorpus/test.truecase.detok.orig',
    ('turkcorpus_test', 'refs'): [DATA_DIR / f'test_sets/turkcorpus/test.truecase.detok.simp.{i}' for i in range(8)],
    ('turkcorpus_valid', 'orig'): DATA_DIR / f'test_sets/turkcorpus/tune.truecase.detok.orig',
    ('turkcorpus_valid', 'refs'): [DATA_DIR / f'test_sets/turkcorpus/tune.truecase.detok.simp.{i}' for i in range(8)],
    ('turkcorpus_test_legacy', 'orig'): DATA_DIR / f'test_sets/turkcorpus/legacy/test.8turkers.tok.norm',
    ('turkcorpus_test_legacy', 'refs'): [
        DATA_DIR / f'test_sets/turkcorpus/legacy/test.8turkers.tok.turk.{i}' for i in range(8)
    ],
    ('turkcorpus_valid_legacy', 'orig'): DATA_DIR / f'test_sets/turkcorpus/legacy/tune.8turkers.tok.norm',
    ('turkcorpus_valid_legacy', 'refs'): [
        DATA_DIR / f'test_sets/turkcorpus/legacy/tune.8turkers.tok.turk.{i}' for i in range(8)
    ],
    ('qats_test', 'orig'): DATA_DIR / f'test_sets/qats/qats.test.orig',
    ('qats_test', 'refs'): [DATA_DIR / f'test_sets/qats/qats.test.simp'],
}
SYSTEM_OUTPUTS_DIR = DATA_DIR / "system_outputs"
SYSTEM_OUTPUTS_DIR_SENTENCE_LEVEL = SYSTEM_OUTPUTS_DIR / "sentence_level"
SYSTEM_OUTPUTS_DIRS_MAP_SENTENCE_LEVEL = {
    "turkcorpus_test": SYSTEM_OUTPUTS_DIR / "turkcorpus/test",
    "zest_test": SYSTEM_OUTPUTS_DIR_SENTENCE_LEVEL / "ZEST/test",
    "zest_valid": SYSTEM_OUTPUTS_DIR_SENTENCE_LEVEL / "ZEST/valid",
}

# Constants
VALID_TEST_SETS_SENTENCE_LEVEL = list(set([test_set for test_set, language in TEST_SETS_PATHS_SENTENCE_LEVEL.keys()])) + ['custom']
VALID_TEST_SETS_DOCUMENT_LEVEL = list(set([test_set for test_set, language in TEST_SETS_PATHS_DOCUMENT_LEVEL.keys()])) + ['custom']
VALID_METRICS = [
    'bleu',
    'sari',
    'samsa',
    'fkgl',
    'sent_bleu',
    'f1_token',
    'sari_legacy',
    'sari_by_operation',
    'bertscore',
    'fre',
    'wiener_sachtextformel_1',
    'wiener_sachtextformel_2',
    'wiener_sachtextformel_3',
    'wiener_sachtextformel_4',
    'osman',
    'fernandez_huerta',
    'szigriszt_pazos',
    'gutierrez_polini',
    'crawford',
    'gulpease_index',
    # 'document_sari',
    # 'document_sari_by_operation',

]
DEFAULT_METRICS = ['bleu', 'sari', 'fre', 'wiener_sachtextformel_1']

LANGUAGE = "de"  # "en"
SPACY_MODEL_NAME = "_core_news_sm"  # "_core_web_sm" # 
