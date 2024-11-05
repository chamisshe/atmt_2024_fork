import sentencepiece as spm

spm.SentencePieceTrainer.train(input=r"data\en-fr-bpe\raw\tiny_train.en", model_prefix='assignments/03/bpe/m', vocab_size=1000, model_type="bpe")

# spm_train --input=<input> --model_prefix=<model_name> --vocab_size=8000 --character_coverage=1.0 --model_type=<type>