import json
!ls
!ls
with open("train_rand_split.jsonl", "r") as f:
    json_list = list(f)
for json_str in json_list:
    result = json.loads(json_str)
    # print(f"result: {result}")
    # print(isinstance(result, dict))
result
import pandas as pd
df = pd.DataFrame({"id":[], "question":[], "concept":[], "true_answer":[], "wrong1":[], "wrong2":[], "wrong3":[], "wrong4":[]})
df
def get_instance(res):
    d = {}
    d["id"] = res['id']
    d["question"] = res['question']
    d["true_answer"] = res['choices'][ord(res['answerKey']) - ord('A')]['text']
    for i in range(5):
        if i != ord(res['answerKey']) - ord('A'):
            d[f"wrong{i}"] = res['choices'][i]['text']
    return d
get_instance(result)
def get_instance(res):
    d = {}
    d["id"] = res['id']
    d["question"] = res['question']['stem']
    d["true_answer"] = res['question']['choices'][ord(res['answerKey']) - ord('A')]['text']
    for i in range(5):
        if i != ord(res['answerKey']) - ord('A'):
            d[f"wrong{i}"] = res['question']['choices'][i]['text']
    return d
get_instance(result)
!ls
!ls
for json_str in json_list:
    result = json.loads(json_str)
    d = get_instance(result)
    df.append(d, ignore_index = True)
df
for json_str in json_list:
    result = json.loads(json_str)
    d = get_instance(result)
    df = pd.concat([df, pd.DataFrame(d)], ignore_index = True)
pd.DataFrame(d)
d
pd.DataFrame(d, ignore_index=True)
df
d_all = {"id":[], "question":[], "concept":[], "true_answer":[], "wrong1":[], "wrong2":[], "wrong3":[], "wrong4":[]}
for json_str in json_list:
    result = json.loads(json_str)
    d = get_instance(result)
    for key, val in d.items():
        d_all[key].append(val)
def get_instance(res):
    d = {}
    d["id"] = res['id']
    d["question"] = res['question']['stem']
    d["true_answer"] = res['question']['choices'][ord(res['answerKey']) - ord('A')]['text']
    for i in range(5):
        if i != ord(res['answerKey']) - ord('A'):
            d[f"wrong{i+1}"] = res['question']['choices'][i]['text']
    return d
for json_str in json_list:
    result = json.loads(json_str)
    d = get_instance(result)
    for key, val in d.items():
        d_all[key].append(val)
def get_instance(res):
    d = {}
    d["id"] = res['id']
    d["question"] = res['question']['stem']
    d["true_answer"] = res['question']['choices'][ord(res['answerKey']) - ord('A')]['text']
    cnt = 0
    for i in range(5):
        if i != ord(res['answerKey']) - ord('A'):
            d[f"wrong{cnt+1}"] = res['question']['choices'][i]['text']
            cnt += 1
    return d
d_all = {"id":[], "question":[], "concept":[], "true_answer":[], "wrong1":[], "wrong2":[], "wrong3":[], "wrong4":[]}
for json_str in json_list:
    result = json.loads(json_str)
    d = get_instance(result)
    for key, val in d.items():
        d_all[key].append(val)
d_all
df = pd.DataFrame(d_all)
[(key, len(val)) for key, val in d_all.items()]
d_all.pop('concept')
d_all.keys()
df = pd.DataFrame(d_all)
df.head()
df.to_csv("train.csv", index=False)
lls
!ls
with open("dev_rand_split.jsonl", "r") as f:
    json_list = list(f)
d_all_dev = {"id":[], "question":[], "concept":[], "true_answer":[], "wrong1":[], "wrong2":[], "wrong3":[], "wrong4":[]}
d_all_dev
for json_str in json_list:
    result = json.loads(json_str)
    d = get_instance(result)
    for key, val in d.items():
        d_all_dev[key].append(val)
df_dev = pd.DataFrame(d_all_dev)
d_all_dev.pop('concept')
df_dev = pd.DataFrame(d_all_dev)
df_dev.to_csv("dev.csv", index=False)
!wc -l *
d_all_tst = {"id":[], "question":[], "concept":[], "ans1":[], "ans2":[], "ans3":[], "ans4":[], "ans5":[]}
with open("test_rand_split_no_answers.jsonl", "r") as f:
    json_list = list(f)
def get_test_instance(res):
    d = {}
    d["id"] = res['id']
    d["question"] = res['question']['stem']
    # d["true_answer"] = res['question']['choices'][ord(res['answerKey']) - ord('A')]['text']
    # cnt = 0
    for i in range(5):
        #if i != ord(res['answerKey']) - ord('A'):
        d[f"ans{i+1}"] = res['question']['choices'][i]['text']
        # cnt += 1
    return d
for json_str in json_list:
    result = json.loads(json_str)
    d = get_test_instance(result)
    for key, val in d.items():
        d_all_tst[key].append(val)
len(d_all_tst)
len(d_all_tst["id"])
d_all_tst.pop('concept')
df_tst = pd.DataFrame(d_all_tst)
df_tst.head()
df_tst.to_csv("test.csv", index=False)
%hist -f proc.py
