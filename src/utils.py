import os
import json
import pickle

global debug
debug = False

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f]

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def dump_jsonl(data, filename):
    if debug:
        print("Debug Mode: Skip dumping jsonl")
        return
    with open(filename, "w") as f:
        for line in data:
            json.dump(line, f)
            f.write("\n")

def dump_pickle(data, filename):
    if debug:
        print("Debug Mode: Skip dumping pickle")
        return
    with open(filename, "wb") as f:
        pickle.dump(data, f)
