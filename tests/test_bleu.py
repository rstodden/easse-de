import pytest

from easse import sari
from easse.utils.helpers import read_lines
from easse.utils.resources import get_orig_sents, get_refs_sents, get_system_outputs_dir


def test_corpus_sari():
    orig_sents = get_orig_sents('turkcorpus_test', input_level="sentence-level")
    refs_sents = get_refs_sents('turkcorpus_test', input_level="sentence-level")
    system_outputs_dir = get_system_outputs_dir('turkcorpus_test', input_level="sentence-level")
    hyp_sents = read_lines(system_outputs_dir / "ACCESS")
    sari_score = sari.corpus_sari(orig_sents, hyp_sents, refs_sents)
    assert sari_score == pytest.approx(41.381013)  # Scores from MUSS https://arxiv.org/abs/2005.00352
