# Multi-Perspective Self-Consistency(MPSC)
This code implementation of MPSC is consist of three stages.

## Stage 1: Query LLM (Not included in the code base)
Given several pre-defined perspectives (for example, `code, spec, testcase`), LLM is ask to generate a group of outputs (for example, `[code_0, ..., code_N]`) from each perspective. The results is a `List[str]` for each perspective.

The generation process is not included in the code base. One can use arbituary LLMs and prompts to generate outputs.

Save the generated outputs in one json file for each perspective. (See `data/vertex/code.json`)

## Stage 2: Inter-consistency Measurement (Not included in the code base)
Given the generated outputs from multiple perspectives, one should design inter-consistency measurement for each pair of perspectives. For example, for `code` and `spec` perspective, one should assign each pair of `(code_i, spec_j)` with a score `s[code_i][spec_j]`. The results is a `Dict[Dict[float]]` for each pair of perspectives.

The verification process is not included in the code base.

Save the verification results in one json file for each pair of perspectives. (See `data/edge/code-spec.json`)

## Stage 3: Graph Ranking
Given the results from the above two stages, one can run the graph.py to rank outputs within each perspective.

1. Load results of Stage 1 to initialize different `VertexGroup()`, each corresponds to the generated outputs from one perspective.
2. Load results of Stage 2 to initialize different `EdgeGroup()`, each corresponds to the inter-consistency results from a pair of perspectives.
3. Pass the `vertex_group_lst` and `edge_group_lst` to construct a `NPartiteGraph()` for MPSC ranking.