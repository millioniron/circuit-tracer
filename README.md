```markdown
# circuit-tracer + Graph-LLM Extension

This repository is a **fork and extension** of [`circuit-tracer`](https://github.com/safety-research/circuit-tracer), a library for mechanistic interpretability in large language models (LLMs) using **cross-layer transcoders**. We build upon the foundational methodology introduced in:

- **Ameisen et al. (2025)** – [*Circuit Tracing: Revealing Computational Graphs in Language Models*](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)  
- **Lindsey et al. (2025)** – [*On the Biology of a Large Language Model*](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)

Our extension focuses on **applying and adapting circuit tracing to graph-structured inputs**, as explored in our paper:

> **Guan et al. (2025)** – [*Attention Mechanisms Perspective: Exploring LLM Processing of Graph-Structured Data*](https://arxiv.org/abs/2505.02130)  
> **[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2505.02130)**

We investigate **why LLMs struggle to “think in graphs”** using a **dual-diagnosis framework**:  
🔹 **Macro-level**: attention pattern analysis  
🔹 **Micro-level**: circuit tracing with attribution graphs

---

## 🧠 Core Capabilities (Inherited from `circuit-tracer`)

1. **Circuit Discovery**  
   Computes attribution graphs showing how transcoder features, error nodes, and input tokens causally influence each other and the output logits.

2. **Interactive Visualization**  
   Explore and annotate circuits via a web-based interface (same as Anthropic’s original frontend).

3. **Feature Interventions**  
   Perturb or steer transcoder features to validate causal hypotheses (e.g., suppress “Texas” features to change capital prediction).

---

## 🌐 Our Extension: Graph-Structured Data Analysis

We extend `circuit-tracer` to diagnose **how LLMs process graph-structured data serialized as text**, revealing two key failure modes:

- **Premature abstraction**: Fine-grained topological features (e.g., node degree, neighbor sets) are compressed into coarse statistical summaries in early layers.
- **Destructive cross-modal fusion**: Linguistic semantics override graph structure in deeper layers, causing predictions to prioritize fluency over relational accuracy.

### Key Scripts & Notebooks

| File | Purpose |
|------|--------|
| `LLM_atten_initial.py` | Baseline LLM processing of graph tasks |
| `LLM_atten_pad.py` | Enhanced version with padding/shuffling |
| `LLM_atten_atten.py` | Adds custom attention mechanisms |
| `LLM_cengji_A.ipynb` | Reproduces **Section A** (attention analysis) |
| `LLM_cengji_B.ipynb` | Reproduces **Section B** (circuit tracing on graph inputs) |

### Datasets
We provide preprocessed graph datasets (e.g., `Roman-Empire`, `Amazon-Ratings`) used in our study:  
📥 [Download from Google Drive](https://drive.google.com/drive/folders/1gIguSsAhqqEeQor2pfxvzH-d4tzWADZF?usp=sharing)

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
conda create --name Exploration python=3.8
conda activate Exploration

# Install PyTorch + Geometric
pip install torch==2.0.0+cu118 torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
pip install torch_geometric pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install LLM & interpretability deps
pip install transformers==4.31.0 peft accelerate bitsandbytes sentencepiece protobuf
pip install matplotlib datasets
```

### 2. Install `circuit-tracer`
```bash
git clone https://github.com/your-username/circuit-tracer.git
cd circuit-tracer
pip install .
```

### 3. Try Our Demos
- Run `demos/circuit_tracing_tutorial.ipynb` to replicate state-capital reasoning.
- Use `LLM_cengji_B.ipynb` to trace circuits on graph-serialized inputs (e.g., `node_a connect[node_b, node_c]`).

---

## 🧪 Usage Examples

### CLI: Analyze a Graph Prompt
```bash
circuit-tracer attribute \
  --prompt "node_b connect[node_c, node_d]" \
  --transcoder_set gemma \
  --slug graph-demo \
  --graph_file_dir ./graphs \
  --server
```

### Python: Perform Intervention on Topology Features
```python
from circuit_tracer import ReplacementModel, intervene

model = ReplacementModel.from_pretrained("gemma", "mntss/clt-gemma-2-2b-2.5M")
graph = model.attribute("node_b connect[node_c, node_d]")

# Suppress "high-degree node" features
intervene(model, graph, target_supernode="high_degree", factor=-2.0)
```

---

## 📚 Available Transcoders
We support transcoders for:
- **Gemma-2 (2B)** – PLTs & CLTs ([426K](https://huggingface.co/mntss/clt-gemma-2-2b-426k), [2.5M](https://huggingface.co/mntss/clt-gemma-2-2b-2.5M))
- **Llama-3.2 (1B)** – PLTs & CLTs
- **Qwen-3** – PLTs (0.6B to 14B)

Use `--transcoder_set gemma` or provide HuggingFace repo IDs.

---

## 📝 Citation

If you use this code, please cite both the original `circuit-tracer` and our graph extension:

```bibtex
% Original circuit tracing framework
@article{ameisen2025circuit,
  author = {Ameisen, Emmanuel and Lindsey, Jack and Pearce, Adam and others},
  title = {Circuit Tracing: Revealing Computational Graphs in Language Models},
  journal = {Transformer Circuits Thread},
  year = {2025},
  url = {https://transformer-circuits.pub/2025/attribution-graphs/methods.html}
}

% Our graph-LLM extension
@article{guan2025attention,
  title = {Attention Mechanisms Perspective: Exploring LLM Processing of Graph-Structured Data},
  author = {Guan, Zhong and Wu, Likang and Zhao, Hongke and He, Ming and Fan, Jianpin},
  journal = {arXiv preprint arXiv:2505.02130},
  year = {2025}
}

% circuit-tracer library
@misc{circuit-tracer,
  author = {Hanna, Michael and Piotrowski, Mateusz and Lindsey, Jack and Ameisen, Emmanuel},
  title = {circuit-tracer},
  howpublished = {\url{https://github.com/safety-research/circuit-tracer}},
  year = {2025}
}
```

---

## 🤝 Contributing

We welcome issues, feature requests, and pull requests!  
If you’re extending this work to other modalities (e.g., code, vision-graph hybrids), please reach out.

---

> **Note**: This project is for **research purposes only**. Interpretability insights should be validated with domain experts—especially in high-stakes domains like medicine or management decision-making.
```
