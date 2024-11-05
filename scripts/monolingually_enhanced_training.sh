#!/usr/bin/env bash
# -*- coding: utf-8 -*-


python3 "./scripts/assignment_3/add_monolingual_data.py" \
    --in-dir "data/en-fr" \
    --src-lang "fr" \
    --tgt-lang "en" \
    --out-dir "data/en-fr-enhanced" \
    --mono-data "data/monolingual/MONO-en-TED-20000.txt" \
    --mono-size 10000 \
    --shuffle \
    --random-sample

# python ".\scripts\assignment_3\add_monolingual_data.py" --in-dir "data\en-fr" --src-lang "fr" --tgt-lang "en" --out-dir "data\en-fr-enhanced" --mono-data "data\monolingual\MONO-en-TED-20000.txt" --mono-size 10000 --shuffle --random-sample

# binarize data for model training
bash scripts/preprocess_data.sh data/en-fr-enhanced/raw/ fr en

# python $base/preprocess.py \
#     --source-lang $src_lang \
#     --target-lang $tgt_lang \
#     --dest-dir $prepared \
#     --train-prefix $preprocessed/train \
#     --valid-prefix $preprocessed/valid \
#     --test-prefix $preprocessed/test \
#     --tiny-train-prefix $preprocessed/tiny_train \
#     --threshold-src 1 \
#     --threshold-tgt 1 \
#     --num-words-src 4000 \
#     --num-words-tgt 4000

python3 train.py \
    --data data/en-fr-enhanced/prepared \
    --source-lang fr \
    --target-lang en \
    --log-file assignments/03/mono_enhanced/log.log \
    --save-dir assignments/03/mono_enhanced/checkpoints \
    --cuda 

--data data/en-fr-enhanced/prepared --source-lang fr --target-lang en --log-file assignments/03/mono_enhanced/log.log --save-dir assignments/03/mono_enhanced/checkpoints --cuda

python3 translate.py \
    --data data/en-fr-enhanced/prepared \
    --dicts data/en-fr-enhanced/prepared \
    --output assignments/03/mono_enhanced/translations.txt
    --checkpoint-path assignments/03/mono_enhanced/checkpoints
    --batch-size 25
    --cuda

echo "done."