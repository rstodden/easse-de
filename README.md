# EASSE
[**EASSE**](https://www.aclweb.org/anthology/D19-3009/) (**E**asier **A**utomatic **S**entence **S**implification **E**valuation) is a Python 3 package aiming to facilitate and standardise automatic evaluation and comparison of Sentence Simplification systems. ([*What is Sentence Simplification?*](https://www.mitpressjournals.org/doi/full/10.1162/coli_a_00370))

We edited EASSE to evaluate German simplification systems. We made the following changes to the original:

- uploaded German [sentence-level test sets](easse/resources/data/test_sets/sentence_level) and [document-level test sets](easse/resources/data/test_sets/document_level)
- uploaded German [sentence-level system outputs](easse/resources/data/system_outputs/sentence_level)
- uploaded [human judgements](easse/resources/data/human_judgements) regarding German test sets (so far only on gold simplifications)
- uploaded [evaluation reports](easse/reports) for German sentence-level experiments 
- added [reproduction scripts](easse/scripts/reproduction/) for available German TS systems 
- added script to add German resources  [easse/install.sh](easse/install.sh) on sentence and document level
- removed some English resources (turkcorpus, qats and pwkp kept for tests)
- added evaluation metrics for German data
  - Flesch Reading Ease adapted by Amstad (1978)
  - Wiener Sachtextformeln following Bamberger & Vaneck (1984)
- added an [interpretation sheet](easse/Metric_Interpretation.md) for readability metrics
- added identity baseline (source text copied) 
- added multi-lingual tokenizer (i.e., spacy)
- changed tseval from English version (https://github.com/facebookresearch/text-simplification-evaluation) to multilingual version (https://github.com/rstodden/text-simplification-evaluation)


### Features

- Automatic evaluation metrics (e.g. SARI<sup>1</sup>, BLEU, SAMSA, etc.).
- Commonly used [**evaluation sets**](https://github.com/feralvam/easse/tree/master/easse/resources/data/test_sets).
- Literature [**system outputs**](https://github.com/feralvam/easse/tree/master/easse/resources/data/system_outputs) to compare to.
- Word-level transformation analysis.
- Referenceless Quality Estimation features.
- Straightforward access to commonly used evaluation datasets.
- Comprehensive HTML report for quantitative and qualitative evaluation of a simplification output.

[1]: The SARI version in EASSE fixes inconsistencies and bugs in the original version. See the dedicated section for more details.

## Installation
### Requirements

Latest tested with Python 3.12.

### Installing from Source

Install EASSE by running:

```
git clone https://github.com/rstodden/easse-de.git
cd easse
pip install -e .
```

This will make `easse` available on your system but it will use the sources from the local clone
you made of the source repository.

Pick the fasttextmodel of your language from here https://fasttext.cc/docs/en/crawl-vectors.html, unzip and save into the tseval/resources/fasttext-vectors/cc.<lang>.300.vec folder.
Download spacy model of your language from here https://spacy.io/models/.
    
Please specify the language of your interest in [./easse/utils/constants.py](./easse/utils/constants.py).

## Running EASSE

### CLI
Once EASSE has been installed, you can run the command-line interface with the `easse` command.

```
$ easse
Usage: easse [OPTIONS] COMMAND [ARGS]...

Options:
  --version   Show the version and exit.
  -h, --help  Show this message and exit.

Commands:
  evaluate  Evaluate a system output with automatic metrics.
  report    Create a HTML report file with automatic metrics, plots and samples.
```

#### easse evaluate
```
$ easse evaluate -h
Usage: easse evaluate [OPTIONS]

Options:
  -m, --metrics TEXT              Comma-separated list of metrics to compute.
                                  Valid: bleu,sari,samsa,fkgl,fre (SAMSA is
                                  disabled by default for the sake of speed)
  -tok, --tokenizer [13a|intl|moses|penn|none|spacy]
                                  Tokenization method to use.
  --refs_sents_paths TEXT         Comma-separated list of path(s) to the
                                  references(s). Only used when test_set ==
                                  "custom"
  --orig_sents_path PATH          Path to the source sentences. Only used when
                                  test_set == "custom"
  --sys_sents_path PATH           Path to the system predictions input file
                                  that is to be evaluated.
  -t, --test_set [zest_(test|valid)|textcomplexityde_test|apa_lha-or-a2_(test|valid)|
                  apa_lha-or-b1()test|valid)|deasy-share_test|custom]
                                  test set to use.  [required]
  -a, --analysis                  Perform word-level transformation analysis.
  -q, --quality_estimation        Perform quality estimation.
  -h, --help                      Show this message and exit.
```
Example with the [ACCESS](https://github.com/facebookresearch/access) system outputs:
```
easse evaluate -t turkcorpus_test -m 'bleu,sari,fkgl' -q < easse/resources/data/system_outputs/turkcorpus/test/ACCESS
```

<img src="https://github.com/feralvam/easse/blob/master/demo/evaluate.gif">

#### easse report
```
$ easse report -h
Usage: easse report [OPTIONS]

Options:
  -m, --metrics TEXT              Comma-separated list of metrics to compute.
                                  Valid: bleu,sari,samsa,fkgl,fre_sent (SAMSA is
                                  disabled by default for the sake of speed
  -tok, --tokenizer [13a|intl|moses|penn|none|spacy]
                                  Tokenization method to use.
  --refs_sents_paths TEXT         Comma-separated list of path(s) to the
                                  references(s). Only used when test_set ==
                                  "custom"
  --orig_sents_path PATH          Path to the source sentences. Only used when
                                  test_set == "custom"
  --sys_sents_path PATH           Path to the system predictions input file
                                  that is to be evaluated.
  -t, --test_set [zest_(test|valid)|textcomplexityde_test|apa_lha-or-a2_(test|valid)|
                  apa_lha-or-b1()test|valid)|deasy-share_test|custom]
                                  test set to use.  [required]
  -p, --report_path PATH          Path to the output HTML report.
  -h, --help                      Show this message and exit.
 ```


German example DEplain-web:
```
shell
easse report -m "sari,fre,bleu,bertscore" --no-lowercase -tok "spacy" -t custom --orig_sents_path ./resources/data/test_sets/sentence_level/DEplain-web/manual-public/DEplain-web-manual-public.test.org --refs_sents_paths ./resources/data/test_sets/sentence_level/DEplain-web/manual-public/DEplain-web-manual-public.test.simp --sys_sents_path ./resources/data/system_outputs/sentence_level/DEplain-web/test/DEplain_trimmed_mbart_sents_apa_web.txt,./resources/data/system_outputs/sentence_level/DEplain-web/test/mt5_DEplain-APA.txt -p test.html
```
The results in the report should be the same as in Table 6 of Stodden (2024a, https://aclanthology.org/2024.determit-1.1/).

<img src="https://github.com/feralvam/easse/blob/master/demo/report.gif">



### Python

You can also use the different functions available in EASSE from your Python code.

```python
from easse.sari import corpus_sari

corpus_sari(orig_sents=["About 95 species are currently accepted.", "The cat perched on the mat."],  
            sys_sents=["About 95 you now get in.", "Cat on mat."], 
            refs_sents=[["About 95 species are currently known.", "The cat sat on the mat."],
                        ["About 95 species are now accepted.", "The cat is on the mat."],  
                        ["95 species are now accepted.", "The cat sat."]])
Out[1]: 33.17472563619544
```

## Metrics
[Interpretation of metrics.](easse/Metric_Interpretation.md)

## Differences with original SARI implementation

The version of SARI fixes inconsistencies and bugs that were present in the original implementation. The main differences are:
1) The original SARI implementation applies normalisation (NIST style tokenization and rejoin ‘s, ‘re ...) only on the prediction and references but not on the source sentence (see STAR.java file). This results in incorrect ngram additions or deletions. EASSE applies the same normalization to source, prediction and references.
2) The original SARI implementation takes tokenized text as input that are then tokenized a second time. This also causes discrepancies between the tokenization of the training set and the evaluation set. EASSE uses untokenized text that is then tokenized uniformly at runtime, during evaluation. This allows for training models on raw text without worrying about matching the evaluation tokenizer.
3) The original JAVA implementation had a silent overflow bug where ngram statistics would go beyond the maximum limit for integers and silently start over from the minimum value. This caused incorrect SARIs when rating too many sentences but did not raise an error.

## Information to Tokenizers
- _13a_: sacrebleu.tokenizers.tokenizer_13a.Tokenizer13a
  - "Tokenizes an input line using a relatively minimal tokenization that is however equivalent to mteval-v13a, used by WMT."
- _intl_: sacrebleu.tokenizers.tokenizers_intl.TokenizerV14International
  - "Tokenizes a string following the official BLEU implementation."
- _moses_: sacremoses.MosesTokenizer()
- _penn_: sacremoses.MosesTokenizer().penn_tokenize()
- _none_: sacrebleu.tokenizers.tokenizers_none.NoneTokenizer.()
  - No tokenization applied.
  - please choose this option, if your data is already tokenized.
- _spacy_
  - tokenization based on spacy models. 
  - Can be specified for German or other languages.
  - in comparison to the other tokenizers, rather slow

## Licence
EASSE is licenced under the GNU General Public License v3.0.

## Citation

If you use EASSE-multi in your research, please cite the original EASSE paper and the EASSE-DE paper. If you use the system generations or reproducability code please also cite our reproduction paper.

### Reproduction paper:

Regina Stodden. 2024a. [Reproduction & Benchmarking of German Text Simplification Systems](https://aclanthology.org/2024.determit-1.1/). In Proceedings of the Workshop on DeTermIt! Evaluating Text Difficulty in a Multilingual Context @ LREC-COLING 2024, pages 1–15, Torino, Italia. ELRA and ICCL.
```
@inproceedings{stodden-2024-reproduction,
    author = {Regina Stodden},
    title = {{Reproduction \& Benchmark of German Text Simplification Systems}},
    booktitle = "Proceedings of the 1st Workshop on Evaluating Text Difficulty in a Multilingual Context (DeTermIt!)",
    year = {2024 (to appear)},
   address = "Turino, Italy"
}
```

### EASSE-DE:

Regina Stodden. 2024b. [EASSE-DE & EASSE-multi: Easier Automatic Sentence Simplification Evaluation for German & Multiple Languages](https://aclanthology.org/2024.tsar-1.11/). In Proceedings of the Third Workshop on Text Simplification, Accessibility and Readability (TSAR 2024), pages 107–116, Miami, Florida, USA. Association for Computational Linguistics.
```
@misc{stodden-2024-easse,
    author = {Regina Stodden},
      title={{EASSE-DE: Easier Automatic Sentence Simplification Evaluation for German}}, 
      year={2024},
      eprint={2404.03563},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
    url = {https://arxiv.org/abs/2404.03563},
    note = {arXiv preprint, arXiv:2404.03563}
}
```

### Original EASSE paper: 
Fernando Alva-Manchego, Louis Martin, Carolina Scarton, and Lucia Specia. 2019. [EASSE: Easier Automatic Sentence Simplification Evaluation](https://aclanthology.org/D19-3009/). In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP): System Demonstrations*, pages 49–54, Hong Kong, China. Association for Computational Linguistics.

```
@inproceedings{alva-manchego-etal-2019-easse,
    title = "{EASSE}: {E}asier Automatic Sentence Simplification Evaluation",
    author = "Alva-Manchego, Fernando  and
      Martin, Louis  and
      Scarton, Carolina  and
      Specia, Lucia",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP): System Demonstrations",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-3009",
    doi = "10.18653/v1/D19-3009",
    pages = "49--54",
}
```
