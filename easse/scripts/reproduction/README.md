# Reproduction of Models

In this directory, we provide our code which we used to reproduce the German text simplification models.
We have reproduced the following models:

| model name | reference | code | comment |
| -----------|-----------|------|---------|
|hda_LS | [Siegel et al. (2019)](https://doi.org/10.1109/QoMEX.2019.8743173)|[https://github.com/rstodden/easy-to-understand_language](https://github.com/rstodden/easy-to-understand_language) |  We have slightly updated the original code. |
| Sockeye-APA-LHA | [Spring et al. (2021)](https://aclanthology.org/2021.ranlp-1.150/)| [https://github.com/ZurichNLP/RANLP2021-German-ATS](https://github.com/ZurichNLP/RANLP2021-German-ATS)| We haven't changed the original code, please follow the instructions of the original authors.|
| trimmed_mbart_sents_apa  | [Stodden et al. (2023)](https://aclanthology.org/2023.acl-long.908/)| [reproduction-based-on-checkpoints.ipynb](reproduction-based-on-checkpoints.ipynb) | model loaded from Huggingface checkpoint |
|  trimmed_mbart_sents_apa_web | [Stodden et al. (2023)](https://aclanthology.org/2023.acl-long.908/)| [reproduction-based-on-checkpoints.ipynb](reproduction-based-on-checkpoints.ipynb) | model loaded from Huggingface checkpoint |
| BLOOM zero-shot| [Ryan et al. (2023)](https://aclanthology.org/2023.acl-long.269/) | [reproduction_bloom_by_ryan-eta-al-2023.ipynb](reproduction_bloom_by_ryan-eta-al-2023.ipynb) | We have slightly updated the original code. |
| BLOOM 10-random-shot| [Ryan et al. (2023)](https://aclanthology.org/2023.acl-long.269/) | [reproduction_bloom_by_ryan-eta-al-2023.ipynb](reproduction_bloom_by_ryan-eta-al-2023.ipynb) | We have slightly updated the original code. |
| BLOOM 10-similarity-shot| [Ryan et al. (2023)](https://aclanthology.org/2023.acl-long.269/) | [reproduction_bloom_by_ryan-eta-al-2023.ipynb](reproduction_bloom_by_ryan-eta-al-2023.ipynb) | We have slightly updated the original code. |
| customer-decoder-ats| [Anschütz et al. (2023)](https://aclanthology.org/2023.findings-acl.74/)| [reproduction-based-on-checkpoints.ipynb](reproduction-based-on-checkpoints.ipynb) | model loaded from Huggingface checkpoint |
| mT5 models | Stodden et al. (2024) | [mt5-models-loop.py](mt5-models-loop.py) | mT5 models fine-tuned using the provided code |

Additionally, we have fine-tuned mT5 on the simple-german-corpus and on DEplain-APA. The corresponding code is also linked in this repository.