# EdgeDocs Copilot

**A retrieval-augmented generation (RAG) assistant for computer vision, machine learning, and edge-AI documentation.**

EdgeDocs Copilot turns a selected collection of technical documents into a searchable knowledge base. It retrieves relevant passages, reranks them, and uses a language model to produce answers with numbered source citations. The notebook also exposes the evidence behind each answer and includes tools for comparing retrieval methods, chunking strategies, latency, and answer-grounding indicators.

The default corpus covers **PyTorch, ONNX, ONNX Runtime, OpenCV, and OpenVINO**. A representative question is:

> I have trained an image classification model in PyTorch. How can I export it to ONNX and deploy it using ONNX Runtime? Explain the necessary steps and identify potential compatibility issues.

The application explains these workflows from documentation; it does not itself export, train, or deploy the user's models.

**Implementation:** [`EdgeDocsCopilotRAG.ipynb`](EdgeDocsCopilotRAG.ipynb). The interface is a Jupyter/Google Colab notebook with Markdown answers, tables, plots, and expandable source panels. There is no separate web server or packaged command-line application in the supplied notebook.

**Project status:** An implemented and evaluated notebook prototype. The indexing and retrieval experiments run end to end; the recorded local-model answers still need better citation adherence and answer quality before this can be treated as a reliable documentation assistant.

**Validation status:** Implementation details were reviewed against the original notebook. The [recorded execution results](#recorded-execution-results) below come from the executed copy (80 cells) and the [exported CSV/JSON files](#exported-results-and-evaluation-data). Function and class definitions match the original after ignoring comments and docstrings. These are observed outputs from one CPU run, not an independently repeated benchmark. The end-to-end evaluation was limited to three questions.

### Project at a glance

- Built a document ingestion pipeline with structural parsing, two chunking strategies, and incremental Qdrant indexing.
- Compared keyword, dense, and hybrid retrieval, with and without cross-encoder reranking, using explicit evidence labels.
- Indexed 8 accessible documents into 243 parent–child chunks or 208 semantic chunks; one requested source failed to download.
- Found that semantic + dense retrieval achieved the highest recorded Recall@5 (0.656), while added reranking did not consistently improve retrieval quality.
- Identified generation as the main CPU latency bottleneck and missing citations as an unresolved output-quality issue.

Start with the [architecture](#architecture), [quick start](#quick-start), and [recorded results](#recorded-execution-results). Detailed implementation and evaluation notes follow for readers who want to inspect the design.

## Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [How document ingestion works](#how-document-ingestion-works)
- [How retrieval and answering work](#how-retrieval-and-answering-work)
- [Using your own documents](#using-your-own-documents)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Recorded execution results](#recorded-execution-results)
- [Exported results and evaluation data](#exported-results-and-evaluation-data)
- [Outputs and persistence](#outputs-and-persistence)
- [Code map](#code-map)
- [Limitations and troubleshooting](#limitations-and-troubleshooting)
- [Reproducibility and licensing](#reproducibility-and-licensing)

## Features

| Capability | Implementation |
| --- | --- |
| Multiple input formats | URLs, PDFs, HTML, Markdown, plain text, and recursive local folders |
| Structure-aware ingestion | Preserves heading paths, code blocks, lists, tables, and PDF page metadata where extraction permits |
| Two chunking strategies | Parent–child chunking and embedding-based semantic chunking |
| Incremental indexing | Source hashes, content-addressed chunk IDs, reuse of existing chunks, and deletion of stale chunks |
| Three retrieval modes | Keyword, dense, and hybrid retrieval over Qdrant |
| Two-stage ranking | Retrieve 20 candidates, then use a cross-encoder to select 5 passages by default |
| Compound questions | Optional LLM query decomposition and fusion of results across subqueries |
| Grounded generation | Numbered `[S1]` citations, explicit coverage gaps, and instructions to identify source conflicts |
| Evidence inspection | Expandable panels showing retrieved text, generation context, source location, and relevance score |
| Abstention | Skip answer generation when the best selected passage falls below a configurable relevance threshold |
| Evaluation | Recall, MRR, nDCG, stage latency, token usage, estimated API cost, and citation diagnostics |
| Local or hosted generation | Ollama by default; optional Anthropic or OpenAI-compatible APIs |

## Architecture

The indexing path prepares documents once and updates them when their content changes. The question-answering path searches the resulting index.

```mermaid
flowchart TD
    A[URLs and local documents] --> B[Extract, clean, and preserve structure]
    B --> C{Chunking strategy}
    C --> D[Parent sections and child chunks]
    C --> E[Semantic chunks]
    D --> F[Dense and sparse vectors]
    E --> F
    F --> G[Qdrant collections and manifests]
    Q[User question] --> I[Optional query decomposition]
    I --> R[Keyword, dense, or hybrid retrieval]
    G --> R
    R --> X[Cross-encoder reranking and selection]
    X --> K[Deduplicate and budget source context]
    X --> T{Evidence threshold met?}
    T -->|No| N[Abstention with closest passages]
    T -->|Yes| L[Generate cited answer if LLM is available]
    K --> L
    L --> V[Citation diagnostics and source inspection]
```

### Default components

| Component | Default |
| --- | --- |
| Dense embedding model | `BAAI/bge-small-en-v1.5` through Sentence Transformers |
| Dense similarity | Cosine similarity over normalized embeddings |
| Sparse retrieval | Custom identifier-aware BM25-style encoder with Qdrant IDF weighting |
| Vector database | Embedded Qdrant, stored under `cfg.workdir` |
| Fusion | Reciprocal Rank Fusion (RRF), executed by Qdrant by default |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Answer model | `qwen2.5:3b` served by Ollama |
| Presentation | IPython Markdown/HTML, pandas, and Matplotlib |

“Local” means the machine running the notebook. In Colab, the models and embedded database run inside the Colab VM, not on the user's own computer. Ollama calls have no per-token API charge; compute, hosting, and storage costs are outside the notebook's cost accounting.

## Quick start

### Google Colab

1. Upload `EdgeDocs_Copilot-3.ipynb` to Colab.
2. (optional) Select **Runtime → Change runtime type → T4 GPU**, if available. CPU execution is supported by the model-loading code but can be substantially slower.
3. Run the installation cell and the definition cells in notebook order through section 3.
4. In section **4.0**, choose the LLM provider. Leave the defaults for Ollama, or use `cfg.llm_provider = "none"` for retrieval-only operation.
5. Run **4.1** to initialize embeddings, the reranker, the LLM client, and Qdrant. With Ollama selected, `setup_ollama()` installs/starts Ollama, downloads the configured model when necessary, and attempts a smoke test.
6. Review `SOURCES` in **4.2**, then run **4.3** to ingest the documents. The supplied demo builds both chunking strategies.
7. Inspect the corpus in **4.4**, compare retrieval in **4.5**, and ask questions in **4.6**.
8. Run **4.7–4.9** when ready to inspect incremental updates and evaluations.

The original full-system evaluation cell runs the complete evaluation set by default; the executed copy uses `limit=3`. Keep this limit for an initial trial, especially with CPU inference or a paid API.

### Local Jupyter

Use a modern Python environment; Python 3.10+ is a practical starting point, but this notebook does not provide a pinned, tested environment specification.

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in Windows PowerShell, then install the notebook dependencies:

```bash
python -m pip install jupyterlab numpy ipython qdrant-client sentence-transformers pymupdf pymupdf4llm beautifulsoup4 lxml markdownify requests pandas matplotlib
jupyter lab EdgeDocs_Copilot-3.ipynb
```

The notebook installation cell contains the core package list; NumPy, IPython, and PyTorch are also imported or used at runtime. Check that the installed PyTorch build supports your hardware if you intend to use CUDA.

For local LLM use, install Ollama for your operating system and start its service before initializing the notebook. Its automatic installer is written for a Linux/Colab environment and invokes `apt-get` and the Ollama shell installer; it is not a cross-platform installation routine. Once Ollama is installed, the notebook can pull the configured model through its API.

### Credentials and network access

- The default embedded Qdrant setup does not require an account or API key.
- The default public embedding and reranking models do not require a Hugging Face token in the notebook.
- Ollama does not require an LLM API key. Initial dependency/model downloads and remote source ingestion require network access.
- Hosted generation requires the relevant SDK and `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`.
- `get_secret()` checks Colab Secrets first, then environment variables. Section 4.0 can prompt for a key using hidden input.

After dependencies and model weights are available, local-document retrieval and local Ollama generation can operate without fetching online documentation, provided the configured services are accessible locally.

## How document ingestion works

### 1. Fetch and extract

`expand_sources()` accepts individual URLs, file paths, and folders. Folder scanning recursively includes `.pdf`, `.html`, `.htm`, `.md`, `.markdown`, and `.txt` files. URLs are fetched individually: the application does not crawl linked pages or execute website JavaScript.

`load_source()` dispatches extraction based on the extension and HTTP content type:

| Input | Extraction behavior |
| --- | --- |
| PDF | Attempts page-level Markdown with `pymupdf4llm`; falls back to sorted plain text with PyMuPDF |
| HTML | Selects main content, removes navigation and other page noise, and converts the result to Markdown |
| Markdown | Removes YAML front matter and HTML comments, retaining the document structure |
| Plain text | Heuristically promotes headings and recognizes some indented code blocks |

HTML extraction preserves code through temporary placeholders and attempts to retain tables as Markdown pipe tables. HTTP redirects and short HTML meta-refresh redirect pages are handled. Original source identifiers and final URLs are recorded separately.

The extracted text is parsed into `Block` objects of type `heading`, `paragraph`, `list`, `table`, or `code`. `build_sections()` groups these under hierarchical heading paths.

### 2. Clean and normalize

The cleaning stage removes common boilerplate, empty blocks, duplicate content, and stray formatting. Prose and code follow separate normalization rules to avoid flattening code indentation.

For PDFs with at least four pages, repeated headers/footers are detected near page edges. PDF paragraph processing also rejoins some hyphenated line breaks. Cleaned block text is hashed for duplicate-document detection.

These are heuristic transformations. Unusual layouts, repeated meaningful text, or complex formatting can be misinterpreted, so inspect extracted chunks before trusting an index.

### 3. Chunk documents

| Property | Parent–child | Semantic |
| --- | --- | --- |
| Main idea | Search small passages, supply larger parent sections to the LLM | Split near changes in semantic meaning |
| Default size settings | Parents: 900; children: 220 approximate tokens | Minimum target: 80; maximum target: 350 approximate tokens |
| Overlap | Up to 40 approximate tokens of whole trailing paragraph/list blocks | No explicit sliding overlap |
| Search representation | Heading breadcrumb plus child text | Heading breadcrumb plus semantic chunk text |
| Generation context | Parent text, falling back to matched children when needed | Chunk text |
| Additional work | Structural packing | Sentence-neighborhood embeddings and adjacent cosine distances |

Parent–child chunks stay within heading sections. Large sections are packed into parents, then split into smaller searchable children. Oversized code blocks are re-fenced after splitting; split tables repeat their header rows where possible.

Semantic chunking embeds sentence-like units with neighboring context, measures adjacent embedding distances, and uses the **85th percentile** of within-section distances as the default breakpoint threshold. It also respects section boundaries and size guards. Code or tables can be attached to their introductory text when they fit.

Sizes are estimates based on words and punctuation, not a model tokenizer. They are not strict tokenizer-level limits: the tiny-tail merge in semantic chunking can reach 125% of the configured maximum, and oversized individual units can exceed nominal budgets.

### 4. Generate search vectors

Each indexed chunk receives two named vectors:

- **`dense`:** a normalized embedding of its heading breadcrumb and chunk text. Questions receive the configured BGE query instruction before embedding.
- **`bm25`:** a sparse vector using identifier-aware tokenization, light stemming, term-frequency saturation, and document-length normalization. Qdrant supplies IDF weighting.

The keyword tokenizer expands identifiers such as `torch.onnx.export`, `opset_version`, and `InferenceSession` into full-token and component forms. This helps retrieval match both technical names and natural-language descriptions.

The sparse encoder uses hashed token IDs and a fixed `bm25_avg_len=150.0`, rather than computing the corpus's actual average length. It is a custom BM25-style implementation, not a byte-for-byte reference BM25 implementation.

### 5. Store and incrementally update

Default collections are `edgedocs_parent_child` and `edgedocs_semantic`. Each point contains vectors plus text, source metadata, heading path, chunk ID, and parent context where applicable.

A JSON manifest tracks each source's extracted-content hash, cleaned-content hash, and chunk IDs:

| Change | Ingestion behavior |
| --- | --- |
| Extracted content unchanged | Skip cleaning, chunking, embedding, and indexing after extraction |
| Content changed | Rechunk and compare IDs; embed new IDs, reuse existing IDs, and delete vanished IDs |
| Duplicate cleaned document | Skip the duplicate source |
| Empty cleaned document | Skip it; remove previously indexed content for that source if present |
| Source omitted from the input list | Retain it unless `remove_missing=True` |

“Unchanged” still requires fetching/reading and extracting the source to compute its hash. Parent–child IDs include the parent-content hash, so modifying a parent can invalidate all its children, including children whose own text is unchanged.

**Configuration changes require care:** the manifest does not fingerprint model or preprocessing configuration. `force=True` bypasses the source-hash shortcut but still reuses existing chunk IDs; it does not guarantee re-embedding every vector. For embedding-model or preprocessing changes, use a fresh collection prefix and reingest. Changing the embedding model also requires recreating the `Embedder` instance.

## How retrieval and answering work

### Query interpretation

When enabled and an LLM is available, `interpret_query()` asks for structured JSON containing subqueries, library names, and keywords. At most four subqueries are retained. The retrieval pipeline searches the original question and its subqueries, then combines their candidate lists with RRF.

The extracted library names and keywords are descriptive outputs; the current code does not use them as explicit database filters. If interpretation fails, the pipeline falls back to the original question.

### First-stage retrieval

| Mode | Behavior |
| --- | --- |
| `keyword` | Search sparse vectors for lexical relevance |
| `dense` | Search dense vectors for semantic similarity |
| `hybrid` | Combine dense and keyword rankings with RRF |
| `hybrid_manual` | Perform the hybrid fusion in Python for inspection/comparison |

Hybrid retrieval fetches up to **50 candidates per retrieval leg**, fuses the rankings, and returns **20 candidates** by default. RRF adds reciprocal rank contributions rather than directly averaging incomparable dense and sparse scores:

```text
RRF score(chunk) = sum over rankings of 1 / (rrf_k + rank)
```

The manual path uses one-based ranks and `rrf_k=60`. The Qdrant path attempts configurable RRF when supported, otherwise falls back to the server's default fusion. Consequently, the two paths should not be assumed to produce identical scores or rankings in every environment.

### Cross-encoder reranking and selection

The cross-encoder jointly scores each query–passage pair. With multi-query reranking enabled, a passage's ranking score is its maximum score across the original question and subqueries.

`select_top()` normally keeps five passages. To cover compound questions, it can replace weaker picks with a subquery's best passage when that passage meets `subquery_min_rerank_score`. At least two slots remain reserved for the overall strongest passages.

### Context assembly

`build_contexts()` merges children that share a parent and assigns source labels `[S1]`, `[S2]`, and so on. It tries to fit parent sections into a **4,000 approximate-token** source-text budget. If a parent does not fit, it tries the matched child text instead; otherwise it drops that context.

Five selected chunks can therefore produce fewer than five source contexts. Source labels are specific to each answer, not persistent document IDs. The configured budget covers source text, not all prompt instructions and metadata.

### Evidence gate and generation

If the highest selected reranker score is below `0.005` by default, the application returns an abstention message and closest passages instead of asking the LLM to answer. This threshold needs calibration on the actual corpus; a relevance score is not a calibrated probability of correctness.

When evidence passes the gate, the LLM is instructed to:

1. Use only the supplied sources for technical claims.
2. Cite factual claims using `[S#]` labels.
3. Label reasonable inferences explicitly.
4. Identify missing information instead of filling gaps from memory.
5. Explain conflicts between sources.
6. Treat source text as reference material rather than instructions.

The requested answer sections are **Short answer**, **Steps / explanation**, **Compatibility issues and caveats**, **Not covered by the sources**, and, when needed, **Conflicts between sources**.

If no LLM is available and evidence passes the gate, the application displays retrieved sources with a retrieval-only message. The gate checks relevance before generation, not whether every aspect of a compound question is supported.

### Verification and source inspection

`verify_answer()` reports nonexistent citation IDs, factual-looking uncited sentences, and selected API names, flags, or versions not found in the supplied context or question. These diagnostics flag issues for review; they neither prove entailment nor automatically repair the answer.

Expandable source panels show source URLs/paths, heading paths, PDF page references when available, scores, matched passages, and generation context. Displayed passages are truncated to 3,000 characters by default; inspect `resp.contexts` for the complete stored context.

## Using your own documents

Run the definition cells and initialization first. The following examples are notebook cells, not standalone scripts.

### Index a folder or uploaded files

```python
my_sources = ["./docs"]
report = ingest_sources(my_sources, strategy="parent_child")
display(report)
display(corpus_stats(collection_name("parent_child")))
```

In Colab, `upload_files()` opens a file picker and saves uploads under `cfg.workdir/uploads`:

```python
my_sources = upload_files()
report = ingest_sources(my_sources, strategy="parent_child")
```

Use explicit URLs to add web pages. For GitHub Markdown, use the raw file URL. The supplied corpus includes nine source URLs: a version-specific PyTorch ONNX page, ONNX versioning, four ONNX Runtime resources, an OpenCV tutorial, an OpenVINO page, and the PyTorch paper PDF. These URLs are configuration inputs, not a guarantee of current availability or complete coverage.

### Ask and inspect

After defining the `ask()` helper in section 4.6:

```python
resp = ask(
    "How do I create an inference session in ONNX Runtime with Python?",
    mode="hybrid",
    collection=collection_name("parent_child"),
)

print(resp.answer)
print(resp.verification)
print(resp.timings)
```

For retrieval alone, with no query-decomposition or generation call:

```python
hits = retrieve("InferenceSession execution providers", mode="hybrid", k=20)
ranked = rerank("InferenceSession execution providers", hits)
for hit in ranked[:5]:
    print(hit.rerank_score, hit.heading_path, hit.url)
```

### Compare chunking strategies

```python
ingest_sources(my_sources, strategy="parent_child")
ingest_sources(my_sources, strategy="semantic")

resp = answer_question(
    "How do I create an inference session?",
    collection=collection_name("semantic"),
)
show_response(resp)
```

### Update or remove sources

```python
# Reuse unchanged content and update changed content.
report = ingest_sources(my_sources, strategy="parent_child")

# Synchronize this collection to the COMPLETE desired source list.
report = ingest_sources(
    my_sources,
    strategy="parent_child",
    remove_missing=True,
)
```

`remove_missing=True` deletes indexed sources absent from the list for that collection. Do not use a partial list unless deletion of the other sources is intended. Each chunking collection must be updated separately.

## Configuration

Set configuration before initializing components or building the index. Re-running the `Config` cell recreates `cfg` and resets custom settings.

| Setting | Default | Purpose |
| --- | --- | --- |
| `workdir` | `/content/edgedocs` when `/content` exists; otherwise `./edgedocs_data` | Local state and outputs |
| `collection_prefix` | `edgedocs` | Namespace for strategy collections |
| `chunking_strategy` | `parent_child` | Default ingestion/search collection |
| `embed_batch_size` | `64` | Dense embedding batch size |
| `prefetch_k` | `50` | Per-leg hybrid candidate count |
| `candidates_k` | `20` | First-stage candidate count |
| `final_k` | `5` | Selected passages before context deduplication |
| `use_query_decomposition` | `True` | Enable LLM-assisted subqueries |
| `rerank_multi_query` | `True` | Score against subqueries as well as the question |
| `abstain_min_rerank_score` | `0.005` | Pre-generation abstention threshold |
| `subquery_min_rerank_score` | `0.02` | Minimum score for subquery coverage slots |
| `max_context_tokens` | `4000` | Approximate source-text budget |
| `max_output_tokens` | `1800` | Generation output limit |
| `temperature` | `0.0` | Generation setting |
| `ollama_num_ctx` | `8192` | Requested Ollama context window |
| `ollama_keep_alive` | `30m` | Model retention between calls |
| `llm_timeout_s` | `600` | Ollama chat request timeout |
| `match_on` | `context` | Evaluate against parent context or chunk text |

### Retrieval-only mode

Set this in section 4.0 before initialization:

```python
cfg.llm_provider = "none"
cfg.use_query_decomposition = False
```

Embeddings, reranking, ingestion, and retrieval evaluation still work. Generated answers, LLM query interpretation, automatic evaluation drafting, and the faithfulness judge need an LLM.

### Hosted model

Install the relevant optional SDK, select a valid model available to your account, and configure credentials before initializing:

```python
# Install openai separately if needed.
cfg.llm_provider = "openai"
cfg.llm_model = "YOUR_AVAILABLE_MODEL_ID"
# cfg.llm_base_url = "YOUR_OPENAI_COMPATIBLE_ENDPOINT"
```

For Anthropic, install `anthropic`, use `cfg.llm_provider = "anthropic"`, and configure `ANTHROPIC_API_KEY`. Model-name examples in the notebook are configuration examples, not verified availability claims. Hosted providers use their SDK clients; the Ollama timeout setting is not applied to those SDK calls.

Update `price_in_per_mtok` and `price_out_per_mtok` using your chosen provider's rates before reporting cost. The defaults **3.0** and **15.0 USD per million tokens** are placeholders. Hosted generation sends questions and retrieved context to the configured provider.

### Hosted Qdrant

```python
cfg.qdrant_url = "https://YOUR_QDRANT_ENDPOINT"
cfg.qdrant_api_key = get_secret("QDRANT_API_KEY")
```

Set these before the first `get_qdrant()` call. The client is cached, so changing the fields afterward does not automatically reconnect it. A fresh kernel is the simplest way to switch database configurations. Even with hosted Qdrant, ingestion manifests remain in the notebook's local `workdir`.

## Evaluation

The notebook includes **10 demonstration questions: 8 answerable and 2 intended to be outside the demo corpus**. They cover export, inference sessions, execution providers, validation, OpenCV, OpenVINO, opset support, and a compound workflow. The two negative examples concern Hexagon INT4 and TensorRT INT8 calibration caches.

These labels must be reviewed when sources change; an intended negative may become answerable after adding documentation.

### Relevance labels

Labels use source substrings and evidence phrases rather than chunk IDs, allowing reuse across chunking strategies:

```python
custom_eval = [{
    "id": "session",
    "question": "How do I create an ONNX Runtime inference session?",
    "answerable": True,
    "relevant": [{
        "source_contains": "onnxruntime.ai",
        "must_contain": ["InferenceSession"],
        "grade": 2,
    }],
}]
```

A match requires the source substring and **all** evidence phrases, after case and whitespace normalization. The default `match_on="context"` checks parent text for parent–child chunks; `"chunk"` checks the searched text. This choice changes what the metric measures and should be reported with results.

Each passage credits at most one previously uncredited relevance specification, choosing the highest matching grade. Each specification is credited once.

### Retrieval metrics

| Metric | Meaning in this implementation |
| --- | --- |
| Recall@5 / Recall@10 | Fraction of positive relevance specifications credited within the first 5 or 10 hits |
| MRR | Reciprocal rank of the first positive hit; zero if none is found in the returned ranking |
| nDCG@5 / nDCG@10 | Ranking quality using `2^grade - 1` gain and logarithmic rank discount, normalized against ideal specification grades |
| Retrieval latency | Time spent retrieving candidates, in milliseconds |
| Reranking latency | Time spent in the cross-encoder, in milliseconds |

```python
coll = collection_name("parent_child")
display(check_eval_coverage(custom_eval, coll))
per_query, summary = evaluate_retrieval(custom_eval, collection=coll)
display(summary)
plot_retrieval_summary(summary)
```

Fix unmatched specifications before interpreting scores: a missing source or outdated evidence phrase can make a label impossible to satisfy.

The default comparison includes keyword, dense, hybrid, dense plus cross-encoder, and hybrid plus cross-encoder retrieval. `make_retrievers(..., include_manual=True)` adds manual RRF.

**Retrieval evaluation is distinct from the full answer pipeline.** It does not run LLM query decomposition or coverage-aware final selection. Reranked variants retain the full candidate ranking so Recall@10 can be calculated, even though generation normally uses five selected passages.

```python
# Both collections must already contain the corpus.
comparison = compare_chunking(custom_eval)
display(comparison)
```

### Full-system evaluation

```python
display(rerank_score_report(EVAL_SET, coll))
e2e = evaluate_end_to_end(EVAL_SET, collection=coll, limit=3)
report = summarize_end_to_end(e2e)
display(e2e)
display(report["latency_s"])
display(report["totals"])
plot_latency_breakdown(e2e)
```

This records stage timing, total elapsed time, input/output tokens, estimated cost, abstention, coverage-gap flags, citation diagnostics, and selected-passage recall. Summary latency includes mean, median, and p95.

Important interpretation details:

- Token totals include successful query-interpretation and generation calls recorded by the response. Ollama prompt counts can fall back to the notebook's token estimate.
- Local Ollama calls report zero API cost. No-call responses can have unavailable token/cost fields.
- The `abstained` field records the programmatic evidence gate; a refusal written by the LLM is not automatically classified as gate abstention.
- The `final_recall@5` column uses `cfg.final_k` internally, so its name becomes misleading if `final_k` is changed.
- Total elapsed time includes overhead beyond the four plotted stages; verification and display are outside the main answer timer. The stacked timing plot should not be treated as an exact decomposition of every runtime operation.
- Retrieval-only runs do not establish generated-answer quality. Missing citation diagnostics are not evidence of error-free generation.
- Ten demonstration questions are too few for broad reliability claims, especially tail-latency estimates.

### Optional LLM faithfulness judge

```python
judgement = llm_judge_faithfulness(resp)
display(judgement)
```

The configured LLM splits the answer into claims and judges support from supplied sources. Faithfulness is the fraction of judged claims marked supported; missing verdicts are reported separately. This is an additional model-based check, not independent ground truth or a per-citation entailment guarantee. Judge token/cost reporting is separate from the main response evaluation.

`draft_eval_from_corpus(n=10)` can draft questions and evidence phrases, retaining phrases found in sampled passages and writing `eval_draft.json`. Review these drafts manually before using them as labels. This helper also makes LLM calls.

## Recorded execution results

These tables reproduce the saved outputs of the executed notebook and have been cross-checked against the exported CSV/JSON files. The five retrieval-summary rows also reconcile with the 40 per-query records at the exported rounding precision. Values retain the displayed precision; differences between separate evaluation cells reflect separate calls, not statistical confidence intervals. No experiments were rerun to produce this section.

### Execution environment and coverage

| Item | Recorded setting or observation |
| --- | --- |
| Runtime | Colab-style runtime, working directory `/content/edgedocs` |
| Dense embedder | `BAAI/bge-small-en-v1.5`, 384 dimensions, CPU |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`; default device selection in the CPU run |
| LLM | Ollama `qwen2.5:3b`; placement check explicitly reported CPU-only |
| Storage | Embedded/local Qdrant |
| Sources requested | 9 |
| Sources successfully ingested | 8 per chunking strategy |
| Retrieval evaluation | 8 questions labeled answerable, including the question whose OpenCV source failed |
| Full-system evaluation | First 3 questions only: `export`, `ort_session`, `ort_providers` |
| Model smoke test | Returned `ready` in 9.3 s |

The output does not establish the exact CPU model, RAM, or installed package versions. This is **not a T4 GPU benchmark**. Hugging Face warned about unauthenticated downloads, and local Qdrant warned that the requested keyword payload indexes had no effect in local mode; neither warning prevented the demonstrated run.

### Ingestion performance

| Metric | Parent–child | Semantic |
| --- | ---: | ---: |
| Wall time (s) | 72.22 | 282.03 |
| Indexed chunks | 243 | 208 |
| Wall time per chunk (ms) | 297.20 | 1355.90 |
| Extraction (s) | 9.72 | 7.26 |
| Cleaning (s) | 0.04 | 0.04 |
| Chunking (s) | 0.06 | 197.99 |
| Final chunk embedding (s) | 59.91 | 74.55 |
| Indexing (s) | 2.29 | 2.02 |

Semantic ingestion took approximately **3.91×** the wall time and produced **35 fewer chunks**. Its sentence-neighborhood embeddings are included in the chunking stage, explaining much of the additional time. These timings exclude initial package installation and model setup; failed-source overhead and other orchestration mean stage totals need not exactly equal wall time.

| Successfully ingested source | Parent–child chunks | Semantic chunks |
| --- | ---: | ---: |
| PyTorch ONNX documentation | 24 | 19 |
| ONNX versioning | 24 | 18 |
| ONNX Runtime Python getting started | 21 | 18 |
| ONNX Runtime Python API summary | 78 | 68 |
| ONNX Runtime PyTorch export tutorial | 7 | 7 |
| ONNX Runtime `Versioning.md` | 16 | 13 |
| OpenVINO PyTorch conversion | 16 | 15 |
| PyTorch paper PDF | 57 | 50 |
| **Total** | **243** | **208** |

**Corpus issue:** the OpenCV URL redirected to a versioned page that returned **HTTP 403 Forbidden** in both ingestion runs. The coverage check found **14 relevance specifications, with 2 unmatched**—both OpenCV specifications. The OpenCV question remained in the reported retrieval averages and scored zero for every retriever. These results therefore combine retrieval behavior with incomplete corpus coverage; they are not a clean benchmark over a fully available corpus. The tables below preserve the original averages rather than silently removing that question.

The PDF parser reported missing RapidOCR and used Tesseract OCR on two pages. It also inferred the paper title as `**Sam Gross**`, showing that extracted metadata should be reviewed even when ingestion succeeds.

### Retrieval comparison: parent–child index

Means over the eight answerable-labeled questions, as displayed in the standalone retrieval evaluation. Latencies are in **milliseconds per question** and exclude query decomposition and answer generation.

| Retriever | Recall@5 | Recall@10 | MRR | nDCG@5 | nDCG@10 | Retrieval ms | Rerank ms | Total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Keyword | 0.469 | 0.531 | 0.540 | 0.503 | 0.515 | 6.862 | 0.000 | 6.862 |
| Dense | 0.531 | 0.656 | 0.657 | 0.560 | 0.604 | 43.362 | 0.000 | 43.362 |
| Hybrid (Qdrant RRF) | 0.531 | 0.656 | 0.566 | 0.525 | 0.567 | 52.496 | 0.000 | 52.496 |
| Dense + cross-encoder | 0.469 | 0.656 | 0.648 | 0.538 | 0.588 | 52.641 | 3361.244 | 3413.885 |
| Hybrid + cross-encoder | 0.469 | 0.656 | 0.645 | 0.538 | 0.586 | 69.766 | 3248.956 | 3318.721 |

Dense retrieval had the highest MRR and nDCG values in this parent–child comparison and tied hybrid retrieval on Recall@5/10. Keyword retrieval was fastest. Reranking added roughly 3.2–3.4 seconds on CPU and did not improve the aggregate metrics over dense retrieval in this run.

Per-question results help explain those averages:

- `ort_session`, `ort_providers`, and `openvino` achieved Recall@5 = 1.0 and MRR = 1.0 for every retriever.
- `export` achieved Recall@10 = 0.0 for every retriever, despite matchable PyTorch evidence being indexed. Its label specifically required the PyTorch documentation source; potentially useful export passages from other sources do not satisfy that specification.
- `onnx_check` had Recall@5 = 0.0 for every retriever. Dense, hybrid, and both reranked variants recovered its labeled evidence by rank 10.
- `compound` achieved Recall@5/10 = 0.25 across all variants: one of four evidence specifications was credited. Finding one relevant passage did not ensure complete workflow coverage.
- `opencv` scored zero because its required source was missing.

### Chunking comparison

The following comes from the separate `compare_chunking(EVAL_SET)` call. Its parent–child quality values agree with the standalone evaluation; timing values differ because the searches were run again.

| Chunking | Retriever | Recall@5 | Recall@10 | MRR | nDCG@5 | nDCG@10 | Total ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Parent–child | Keyword | 0.469 | 0.531 | 0.540 | 0.503 | 0.515 | 6.483 |
| Parent–child | Dense | 0.531 | 0.656 | 0.657 | 0.560 | 0.604 | 38.437 |
| Parent–child | Hybrid | 0.531 | 0.656 | 0.566 | 0.525 | 0.567 | 43.254 |
| Parent–child | Dense + cross-encoder | 0.469 | 0.656 | 0.648 | 0.538 | 0.588 | 2779.381 |
| Parent–child | Hybrid + cross-encoder | 0.469 | 0.656 | 0.645 | 0.538 | 0.586 | 2687.687 |
| Semantic | Keyword | 0.438 | 0.531 | 0.534 | 0.478 | 0.510 | 6.495 |
| Semantic | Dense | 0.656 | 0.656 | 0.604 | 0.565 | 0.565 | 41.410 |
| Semantic | Hybrid | 0.531 | 0.656 | 0.501 | 0.479 | 0.515 | 44.868 |
| Semantic | Dense + cross-encoder | 0.469 | 0.688 | 0.656 | 0.538 | 0.612 | 3830.086 |
| Semantic | Hybrid + cross-encoder | 0.469 | 0.594 | 0.652 | 0.538 | 0.580 | 3911.142 |

Across these ten configurations, semantic + dense had the highest Recall@5 (**0.656**) and nDCG@5 (**0.565**). Semantic + dense + cross-encoder had the highest Recall@10 (**0.688**) and nDCG@10 (**0.612**). Parent–child + dense had the highest MRR (**0.657**). There is no single winner across every metric, and the small, partially uncovered evaluation set does not establish general superiority.

### Incremental update demonstration

| Step | Observed behavior |
| --- | --- |
| Initial ingestion | Markdown: 3 new chunks in 0.513 s; text: 2 new chunks in 0.328 s |
| Unchanged rerun | All 5 chunks reused, none newly embedded; displayed source times rounded to 0.000 s |
| Edit one Markdown section | 1 new chunk, 1 deleted chunk, 2 Markdown chunks reused; update took 0.198 s. The 2 text chunks were also reused. |
| Remove the text file from the source list | Text source reported `removed`; Markdown retained all 3 chunks |

This demonstrates the intended incremental behavior on the two-file example. The removal report uses placeholder zero counters, so its displayed `n_deleted=0` should not be interpreted as evidence that the removed document's chunks were retained.

### Full-system evaluation: three questions

This run used hybrid retrieval, parent–child context, query decomposition, and local Qwen 2.5 3B generation. All three evaluated questions were labeled answerable.

| Question ID | Selected recall@5 | Query analysis s | Retrieval s | Rerank s | Generation s | Total s | Input tokens | Output tokens | Citations | Uncited sentences |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `export` | 0.0 | 19.807 | 0.212 | 9.219 | 92.234 | 121.472 | 1636 | 293 | 0 | 7 |
| `ort_session` | 1.0 | 20.974 | 0.238 | 10.708 | 210.008 | 241.930 | 2337 | 345 | 0 | 11 |
| `ort_providers` | 1.0 | 18.934 | 0.206 | 13.913 | 85.325 | 118.380 | 3506 | 427 | 0 | 13 |

| Stage | Mean s | Median s | p95 s |
| --- | ---: | ---: | ---: |
| Query understanding | 19.905 | 19.807 | 20.858 |
| Retrieval | 0.219 | 0.212 | 0.236 |
| Reranking | 11.280 | 10.708 | 13.593 |
| LLM generation | 129.189 | 92.234 | 198.231 |
| Total | 160.594 | 121.472 | 229.884 |

Totals were **7,479 input tokens**, **1,065 output tokens**, and **$0.00 reported local-model API cost**. No gate abstentions occurred. The output reported zero invalid citation IDs and zero terms-not-found flags, but also **zero citations and 31 uncited sentences**. Zero invalid citations therefore does not indicate successful citation compliance: the model did not emit citations to validate.

Mean generation time accounted for approximately **80.4%** of mean response time. The displayed p95 is based on only three observations and is not a dependable service-level tail-latency estimate. No unanswerable questions were included in this limited evaluation, so its negative-case gap/abstention summary was `n/a`.

### Interactive answers, abstention, and judge behavior

Separate interactive calls produced the following diagnostics; they are not additional rows in the three-question evaluation above.

| Interactive question | Total s | Citations | Uncited sentences | Gate abstained |
| --- | ---: | ---: | ---: | --- |
| Compound PyTorch export / ONNX Runtime deployment | 353.172 | 0 | 20 | No |
| Qualcomm Hexagon INT4 | 268.020 | 0 | 7 | No |
| ONNX Runtime execution providers | 411.724 | 0 | 14 | No |

The compound answer was repetitive and did not follow the requested citation format. The Hexagon answer discussed general ONNX/OpenVINO conversion, then acknowledged that the requested INT4/Hexagon procedure was not covered. This was a coverage-gap acknowledgement, **not a successful programmatic abstention**.

The separate calibration report, which disables query decomposition, recorded:

| Question group / example | Best reranker score |
| --- | ---: |
| Hexagon INT4 negative example | 0.0001 |
| TensorRT INT8 negative example | 0.5551 |
| Lowest score among answerable-labeled questions | 0.9903 |
| Configured abstention threshold | 0.005 |

The TensorRT negative example exceeds the configured gate threshold in this calibration path. The OpenCV question also scored **0.9990** despite its required source being unavailable. High reranker relevance therefore does not establish exact evidence coverage.

The Hexagon calibration result differs from the interactive non-abstention. The paths differ: calibration disables decomposition, while interactive retrieval can introduce subqueries and use maximum scores across them. This is a plausible mechanism to investigate, not a confirmed diagnosis from the saved output alone. Threshold calibration should be repeated with the same settings used for answering.

The optional judge returned **faithfulness = 1.0 for 11 claims**, with no unjudged or unsupported claims, using 1,926 input and 429 output tokens at $0 reported API cost. It judged `resp`, the earlier **compound-question answer**, not the last row of the evaluation or every generated answer. Because the same configured model performs judging, and that answer still lacked citations and contained repetition, this result does not establish complete answer quality or citation correctness.

### What this run demonstrates

The saved outputs demonstrate ingestion of eight accessible sources, both chunking indexes, retrieval comparisons, incremental updates, local answer generation, evaluation, and a successful results-export message. They also identify concrete improvements to test: restore OpenCV corpus coverage, improve citation-format adherence, evaluate abstention with decomposition enabled, and measure generation on a faster runtime. The notebook provides these mechanisms; the recorded answers show that reliable cited generation still needs further validation.

## Exported results and evaluation data

Place the six supplied result files in a `results/` directory beside this README and the notebook. The relative links below then work directly on GitHub. Keep the executed notebook under the canonical name `EdgeDocs_Copilot-3.ipynb` so the notebook link at the top opens the version with saved outputs.

| File | Size of exported dataset | What it provides |
| --- | --- | --- |
| [retrieval_summary.csv](results/retrieval_summary.csv) | 5 retriever rows | Mean Recall@5/10, MRR, nDCG@5/10, retrieval latency, reranking latency, and total retrieval-pipeline latency on the parent–child index |
| [retrieval_per_query.csv](results/retrieval_per_query.csv) | 40 rows: 8 questions × 5 retrievers | Individual question scores and timings supporting the aggregate retrieval table |
| [chunking_comparison.csv](results/chunking_comparison.csv) | 10 rows: 2 chunking strategies × 5 retrievers | Quality and timing comparison between parent–child and semantic indexes |
| [end_to_end.csv](results/end_to_end.csv) | 3 question rows | Full-system stage timing, selected recall, token usage, local API cost, abstention/gap indicators, and citation diagnostics |
| [ingestion_times.csv](results/ingestion_times.csv) | 8 metric rows × 2 strategies | Wall time, chunk count, milliseconds per chunk, and extraction/cleaning/chunking/embedding/indexing times |
| [eval_set.json](results/eval_set.json) | 10 questions; 14 evidence specifications | Question IDs, question text, answerability labels, source constraints, required evidence phrases, and relevance grades |

### Repository layout

| Repository path | Purpose |
| --- | --- |
| `README.md` | Project overview, setup, implementation, measured results, and limitations |
| `EdgeDocs_Copilot-3.ipynb` | Executed application and experiment notebook |
| `results/` | The five CSV files and evaluation JSON linked above |

The result files preserve the original exports. Some CSVs have an unnamed first column because pandas exported the row index; this is not an extra experimental variable. The first column of `ingestion_times.csv` contains metric names and should be retained as its index. Aggregate retrieval and chunking tables are rounded to three decimal places, ingestion values to two, while per-query and end-to-end CSVs retain greater precision.

### Inspect the exported results without rerunning models

Run this from the repository root with pandas installed:

```python
import json
from pathlib import Path
import pandas as pd

root = Path("results")
retrieval = pd.read_csv(root / "retrieval_summary.csv", index_col="retriever")
per_query = pd.read_csv(root / "retrieval_per_query.csv", index_col=0)
chunking = pd.read_csv(root / "chunking_comparison.csv")
end_to_end = pd.read_csv(root / "end_to_end.csv", index_col=0)
ingestion = pd.read_csv(root / "ingestion_times.csv", index_col=0)
eval_set = json.loads((root / "eval_set.json").read_text())

print(retrieval)
print(chunking.sort_values("Recall@5", ascending=False))
print("Mean response time (s):", end_to_end["total_s"].mean())
print("Question count:", len(eval_set))

# Reconstruct retrieval averages from individual question records.
metric_columns = ["Recall@5", "Recall@10", "MRR", "nDCG@5", "nDCG@10"]
recomputed = per_query.groupby("retriever")[metric_columns].mean().round(3)
print(recomputed)
```

The exports corroborate the recorded metrics; they are not a second independent run. In particular, `end_to_end.csv` contains only the first three answerable questions, not the entire JSON evaluation set. The OpenCV failure, interactive answers, extraction warnings, incremental-update demonstration, and optional judge verdict are documented in the notebook outputs rather than in these six files. The CSVs do not contain full generated answers or retrieved passages, so the executed notebook remains necessary for qualitative inspection.

## Outputs and persistence

Runtime files are created under `cfg.workdir`:

| Path | Contents |
| --- | --- |
| `qdrant/` | Embedded database, when local mode is used |
| `manifest_<collection>.json` | Source hashes and chunk IDs for incremental updates |
| `uploads/` | Files uploaded through the Colab helper |
| `demo_docs/` | Markdown/text files created by the incremental-update demonstration |
| `ollama_serve.log` | Background server log if the notebook starts Ollama |
| `eval_draft.json` | Optional generated evaluation draft |
| `results/retrieval_summary.csv` | Aggregate retrieval comparisons |
| `results/retrieval_per_query.csv` | Per-question retrieval metrics |
| `results/chunking_comparison.csv` | Metrics across the two chunking strategies |
| `results/end_to_end.csv` | Full-system counters and timing |
| `results/ingestion_times.csv` | Ingestion-stage comparison |
| `results/eval_set.json` | Saved evaluation labels |

The final export cell depends on earlier demo variables, including `summary`, `per_query`, `cmp`, `e2e`, and `ing`. Run the corresponding evaluations before exporting, or adapt the cell to the experiments you ran. Plots are displayed inline; the final cell does not export chart images or full answer transcripts.

Colab VM files are temporary unless explicitly preserved. Back up both the embedded database and its matching manifests if incremental state matters. Ollama model storage is managed separately from `cfg.workdir`. The incremental demo recreates its dedicated `edgedocs_demo` collection on each run.

## Code map

| Notebook area | Main entry points | Responsibility |
| --- | --- | --- |
| Setup | `Config`, `Timer`, `get_secret` | Settings, timing, credentials |
| Data structures | `RawDocument`, `Block`, `Section`, `Chunk` | Extracted and indexed representations |
| Extraction | `load_source`, `extract_pdf`, `extract_html`, `parse_blocks` | Input parsing and structure |
| Cleaning | `clean_document`, `clean_pages`, `clean_blocks` | Noise removal and normalization |
| Chunking | `chunk_parent_child`, `chunk_semantic` | Searchable text units |
| Encoding | `Embedder`, `BM25SparseEncoder` | Dense/sparse vectors |
| Storage | `get_qdrant`, `ensure_collection`, `ingest_sources` | Persistence and incremental ingestion |
| LLM | `LLMClient`, `setup_ollama` | Local/hosted generation |
| Retrieval | `retrieve`, `rrf_fuse`, `rerank`, `select_top` | Candidate retrieval and selection |
| Orchestration | `interpret_query`, `retrieve_and_rerank` | Compound-query handling |
| Answering | `build_contexts`, `answer_question`, `verify_answer` | Prompt context, generation, diagnostics |
| Presentation | `ask`, `show_response`, `show_sources` | Interactive notebook output |
| Evaluation | `evaluate_retrieval`, `compare_chunking`, `evaluate_end_to_end` | Quality and performance experiments |

`RAGResponse` provides `answer`, `contexts`, `retrieval`, `verification`, `timings`, `tokens`, `cost_usd`, and `abstained` for programmatic inspection.

## Limitations and troubleshooting

| Symptom or limitation | Explanation / next step |
| --- | --- |
| Ollama unavailable | Check the setup output and server log, then rerun initialization after resolving the service/model issue. `setup_ollama()` returning `False` is not itself a guarantee that every later generation call will degrade gracefully. |
| Slow answers | Check `ollama_placement()`, use GPU inference where available, reduce evaluation size, or disable decomposition. The recorded CPU timings below are observations from one session, not fixed performance guarantees. |
| Zero or very few chunks | Inspect the source: it may be a redirect stub, blocked page, JavaScript-rendered page, scanned PDF, or unsupported layout. The notebook has no separately configured OCR stage; the parser dependency used Tesseract OCR on two pages in the recorded run. OCR behavior depends on the installed parser and environment. |
| Ingestion tables fail after all downloads fail | Some demo display cells assume successful-ingestion columns exist. Inspect the raw report and fix the sources before proceeding. |
| Empty collection during inspection | The example child/parent inspection calls `max()` on collected hits; populate the collection first. |
| Qdrant storage lock | Embedded mode expects one client for a storage directory. Close competing sessions or use a separate work directory. |
| New model but old results | Reinitialize the model and build a fresh collection. Manifest hashes do not track model settings. |
| Unexpected abstention | Check relevant passages and `rerank_score_report()` before adjusting the threshold. Small score changes can affect the gate. |
| Citations look valid but claims are wrong | Citation existence does not establish factual support. Read the source context and optionally use the judge. |
| Poor Recall/nDCG | Check label coverage, source versions, extraction quality, and `match_on` before attributing the result to retrieval. |
| Truncated answers/context | Adjust source/output budgets and Ollama context size together; approximate token counts can understate actual model tokens. |
| Source details remain stale | Heading/location metadata is not fully encoded in chunk IDs. Some metadata-only changes may reuse existing payloads; use a fresh index when provenance must be rebuilt. |

.










..
