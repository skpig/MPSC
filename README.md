# Enhancing Large Language Models in Coding Through Multi-Perspective Self-Consistency [ACL 2024]

## Environment
1. Install required package: `pip install -r requirements.txt`
2. Download benchmark dataset from [google drive](https://drive.google.com/drive/folders/1-Rnwa6vutKdpmnrDpRp5ZJEkh8DzlbsF?usp=sharing) to `data` dir
3. Download auther generated outputs from google drive **[Available Soon!]** to `runtime` dir
4. Update `api.py` to your own OpenAI config

## Directory Structure
```python
|-- data # four code generation datasets
|-- runtime # runtime files including LLM generated results and inter-consistency measurements
|-- src
    |-- pipeline.py # the entry point for LLM sampling & inter-consistency measurements. All results will be saved in `runtime`.
    |-- graph.py # the entry point of MPSC
    |-- evaluation.py, _evaluation.py # evaluation metrics
    |-- execution.py, _execution.py # execution process for inter-consistency measurements
    |-- api.py # OpenAI api 
    |-- exemplars # ICL exemplars for test case generation
```

## Reproduction

- Directly apply author provided LLM generated results for MPSC
    ```
    python3 graph.py
    ```
- MPSC from scratch (Warning: may cause a large number of OpenAI API calls)
    ```
    python3 pipeline.py
    python3 graph.py
    ```

## Usage of MPSC
We also provide a code snippet of MPSC for other tasks in `MPSC` dir.