import argparse
import json
from transformers import AutoTokenizer
import numpy as np

parser = argparse.ArgumentParser(description="Compute EAGLE speedup ratio from answer files.")
parser.add_argument("--ea-file", type=str,
                    default="llama-2-chat-70b-fp16-ea-in-temperature-0.0.jsonl",
                    help="jsonl answer file produced by a gen_ea_answer_* script")
parser.add_argument("--base-file", type=str,
                    default="llama-2-chat-70b-fp16-base-in-temperature-0.0.jsonl",
                    help="jsonl answer file produced by a gen_baseline_answer_* script")
parser.add_argument("--tokenizer", type=str, default="/home/lyh/weights/hf/llama2chat/13B/",
                    help="tokenizer path (used to count baseline tokens)")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
jsonl_file = args.ea_file
jsonl_file_base = args.base_file
data = []
with open(jsonl_file, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)



speeds=[]
for datapoint in data:
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    tokens=sum(datapoint["choices"][0]['new_tokens'])
    times = sum(datapoint["choices"][0]['wall_time'])
    speeds.append(tokens/times)


data = []
with open(jsonl_file_base, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)

total_time=0
total_token=0
speeds0=[]
for datapoint in data:
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    tokens = 0
    for i in answer:
        tokens += (len(tokenizer(i).input_ids) - 1)
    times = sum(datapoint["choices"][0]['wall_time'])
    speeds0.append(tokens / times)
    total_time+=times
    total_token+=tokens



# print('speed',np.array(speeds).mean())
# print('speed0',np.array(speeds0).mean())
print("ratio",np.array(speeds).mean()/np.array(speeds0).mean())
