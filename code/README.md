cd /path/to/Dissertation/directory/
python3 -m venv ~/envs/enron
source ~/envs/enron/bin/activate
pip3 install polars tqdm pyarrow
time python3 ./code/main.py ./dataset/emails.csv -j 12