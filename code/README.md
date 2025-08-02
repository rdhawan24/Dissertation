cd /path/to/Dissertation/directory/
python3 -m venv ~/envs/enron
source ~/envs/enron/bin/activate
pip3 install polars tqdm pyarrow
time python3 ./code/main.py ./dataset/emails.csv -j 12




python -m vllm.entrypoints.openai.api_server   --model deepseek-ai/deepseek-llm-7b-chat --port 8000

supports 4096 tokens

errors out with max-tokens=8192

time python3 ./code/main.py /cs/student/projects2/sec/2024/rdhawan/FPEProject/datasets/emails_cards_encrypted_test.csv -m ./hf_models/pii_model/gliner_multi_pii_onnx -n 50 -j 1 -e --partial-dir ./tmp1/enron_parts --slice-rows 10


