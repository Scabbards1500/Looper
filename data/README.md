# Data

gsm8k (`openai/gsm8k`, main, train), AQuA-RAT (`deepmind/aqua_rat`, raw, train)
and MATH (`nlile/hendrycks-MATH-benchmark`, train) are downloaded from the
Hugging Face hub by `sophia/tasks.py`. The first 500 training examples of
each are used.

LogiQA is read from the original LogiQA release's `Train.txt`
(https://github.com/lgw863/LogiQA-dataset). Put it at `data/logiqa_train.txt`,
or point `SOPHIA_LOGIQA` / `--logiqa-path` at it. Each example is 8 lines: a
blank line, the answer letter, the context, the question and four options.
