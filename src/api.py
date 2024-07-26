from openai import OpenAI
import os
import json
from tqdm import tqdm
import time
import re

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def infer_llm(engine, instruction, exemplars, query, answer_num=5, max_tokens=2048):
    """
    Args:
        instruction: str
        exemplars: list of dict {"query": str, "answer": str}
        query: str
    Returns:
        answers: list of str
    """

    messages = [{"role": "system", "content": "You are a helpful AI assistant.."},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": "OK, I'm ready to help."},
        ]
    
    if exemplars is not None:
        for i, exemplar in enumerate(exemplars):
            messages.append({"role": "user", "content": exemplar['query']})
            messages.append({"role": "assistant", "content": exemplar['answer']})
    
    messages.append({"role": "user", "content": query})

    while True:
        time.sleep(1)
        answers = client.chat.completions.create(
            engine=engine,
            messages=messages,
            temperature=0.8,
            max_tokens=max_tokens,
            top_p=0.95,
            n=answer_num,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        return [response["message"]["content"] for response in answers["choices"] if response['finish_reason'] != 'length']

def infer_llm_completion(engine, instruction, exemplars, query, answer_num=5, max_tokens=2048):
    retry_times = 0
    while True:
        time.sleep(1)
        try:
            answers = client.completions.create(
                engine=engine,
                prompt=instruction+query,
                temperature=0.8,
                max_tokens=max_tokens,
                top_p=0.95,
                n=answer_num,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["<|im_end|>", "<|im_sep|>"],
            )
            return [response["text"] for response in answers["choices"] if response['finish_reason'] != 'length']
        except Exception as e:
            print(e)
            try:
                sleep_time = re.findall(r'Please retry after (\d+) seconds.', e.user_message)
                time.sleep(int(sleep_time[0]))
            except Exception as e:
                time.sleep(10)
            retry_times += 1

