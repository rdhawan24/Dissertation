cd /path/to/Dissertation/directory/


fpe/
├── code/
│   ├── main.py            # orchestrator – slicing, workers, progress bar
│   ├── chunk_worker.py    # loads one slice, extracts bodies, batched NER
│   └── email_processing.py# batched PII NER helpers (ab-ai/pii_model)
└── dataset/
    └── emails.csv


python3 -m venv ~/envs/enron
source ~/envs/enron/bin/activate
pip3 install polars tqdm pyarrow

pip3 install torch==2.2.2 "transformers>=4.39"

(enron) mohan@Garfield:~/Software/Personal/Dissertation$ python - <<'PY'
import torch, transformers, polars, tqdm, numpy, platform
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
print("numpy", numpy.__version__)
print("transformers", transformers.__version__)
print("polars", polars.__version__)
print("CPU:", platform.processor())
PY
torch 2.3.1+cpu | CUDA: False
numpy 2.3.2
transformers 4.54.1
polars 1.31.0
CPU: x86_64

see email for token

mkdir -p ./hf_models/pii_model
cd hf_models/pii_model/


(enron) mohan@Garfield:~/Software/Personal/Dissertation/hf_models/pii_model$ huggingface-cli login
⚠️  Warning: 'huggingface-cli login' is deprecated. Use 'hf auth login' instead.

    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

    To log in, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Enter your token (input will not be visible): 
Add token as git credential? (Y/n) n
Token is valid (permission: fineGrained).
The token `FPE` has been saved to /home/mohan/.cache/huggingface/stored_tokens
Your token has been saved to /home/mohan/.cache/huggingface/token
Login successful.
The current active token is: `FPE`


(enron) mohan@Garfield:~/Software/Personal/Dissertation/hf_models/pii_model$ huggingface-cli download ab-ai/pii_model --local-dir ~/hf_models/pii_model --local-dir-use-symlinks False


time python3 ./code/main.py ./dataset/emails.csv -j 4