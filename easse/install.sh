#todo: copy all readmes?

mkdir -p easse/resources/data/test_sets/sentence_level/
mkdir -p easse/resources/data/test_sets/document_level/
mkdir -p easse/resources/data/system_outputs/document_level/
mkdir -p easse/resources/data/system_outputs/sentence_level/

mkdir -p raw_data
cd raw_data
#
#### SENTENCE LEVEL
# ZEST (parallel, children)
git clone https://github.com/Jmallins/ZEST-data.git
rm -r -f ZEST-data/.git
mv ZEST-data/ ../easse/resources/data/test_sets/sentence_level/ZEST/

# TextComplexityDE
wget https://raw.githubusercontent.com/babaknaderi/TextComplexityDE/master/TextComplexityDE19/parallel_corpus.csv
wget https://raw.githubusercontent.com/babaknaderi/TextComplexityDE/master/LICENSE
cd ..
python easse/utils/download_files.py -test_set_name "TextComplexityDE" -data_path raw_data/parallel_corpus.csv
mv raw_data/LICENSE easse/resources/data/test_sets/sentence_level/TextComplexityDE/LICENSE
cd raw_data
rm parallel_corpus.csv



### ask for APA data
git clone https://github.com/ZurichNLP/RANLP2021-German-ATS.git
cd RANLP2021-German-ATS/
mkdir -p "tools"
git clone https://github.com/facebookresearch/fairseq.git "tools/fairseq"
git clone https://github.com/moses-smt/mosesdecoder.git "tools/mosesdecoder"
mkdir -p data/aligned
# Download the ZIP file from https://zenodo.org/record/5148163
mv ../APA_sentence-aligned_LHA.zip data/APA_sentence-aligned_LHA.zip
cd data
unzip APA_sentence-aligned_LHA.zip -d aligned/
cd ..
mkdir -p "logs"
bash "preprocess/job_preprocess_data.sh" -i "data/aligned/" -r .
##>&2 echo "extracting raw text..."
##extract_raw_text "data/aligned" "data/raw"
##>&2 echo "deduplicating..."
##deduplicate_text "data/raw" "data/deduplicated" "tools/fairseq/examples/backtranslation" "tools/mosesdecoder/scripts"
##>&2 echo "creating train, dev and test..."
##split_train_dev_test "data/deduplicated" "data/splits"
cd ../..
python easse/utils/download_files.py -test_set_name "APA_LHA" -data_path raw_data/RANLP2021-German-ATS/data/splits
rm -r -f raw_data/RANLP2021-German-ATS



# hda easy-to-understand_language (document-level corpus)
wget https://github.com/hdaSprachtechnologie/easy-to-understand_language/raw/master/leichte_sprache_corpus.db
cd ..
python easse/utils/download_files.py -test_set_name hda_easy-to-understand -data_path raw_data/leichte_sprache_corpus.db
rm raw_data/leichte_sprache_corpus.db