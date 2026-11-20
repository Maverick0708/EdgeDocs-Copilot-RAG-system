# EdgeDocs Copilot

If you've trained a model in PyTorch and want to run it with ONNX Runtime, the export instructions and the Python code for loading the model live on different sites. EdgeDocs Copilot is a Jupyter notebook that searches the PyTorch, ONNX, ONNX Runtime and OpenVINO docs in one place, then asks a small local language model to answer with citations. You can expand every source and read exactly what the model was given.

It's an experiment, and it doesn't fully get there yet. In the saved run, the best search method found about two thirds of the labelled evidence in its top five results, and the model produced a checked answer for only 2 of the 7 questions that reached it. The rest of this page covers what worked, what didn't, and one answer that looked right and wasn't.

[View the notebook with its saved outputs](./EdgeDocsCopilotRAG.ipynb) · [Run in Google Colab](https://colab.research.google.com/github/Maverick0708/EdgeDocs-Copilot-RAG-system/blob/main/EdgeDocsCopilotRAG.ipynb)

*Built with AI coding assistants. There's more on that [at the end](#how-this-was-made).*

## A citation worth opening

One of the test questions is **"Do I have to specify execution providers when creating an ONNX Runtime session?"** The notebook answered yes, and quoted the ONNX Runtime Python API page: "Since ONNX Runtime 1.10, you must explicitly specify the execution provider for your target." The quote is real. But expand the first source and read the next sentence. Running on CPU is the only time you can leave the provider unspecified.

That answer passed validation. The retriever had found the exception, the model had been given it, and the citation pointed at the right page. What got through was the word "must" in a claim that needed a qualification. The validator does have a rule for exactly this, but it only looks at the quoted text, and the model quoted just the first sentence, so the rule never fired.

## How a question moves through the notebook

Before you ask anything, the notebook builds a local index in Qdrant from web pages or local PDF, HTML, Markdown and text files. It tries to keep code next to the text that introduces it: the semantic splitter attaches each code block or table to the sentence before it, as long as the result stays under the size limit, so "the following example" doesn't end up separated from the example.

There are two ways to split documents, and the notebook builds both so you can compare them. Parent–child chunking searches short passages and, when one matches, hands the model the whole section around it. Semantic chunking cuts where the meaning shifts between neighbouring sentences, and it costs more to build because every sentence needs an embedding first. Re-indexing is incremental. In the update demo, a tiny three-chunk Markdown file with parent–child chunking, editing one section replaced one chunk and reused the other two. The notebook doesn't test that on the full corpus.

A name like `torch.onnx.export` is something keyword search (BM25) can match exactly. Someone asking "How do I export my model?" might never type it, which is what dense search with `BAAI/bge-small-en-v1.5` is for. You can run either one alone, combine them with reciprocal rank fusion, or have a MiniLM cross-encoder reread each candidate next to the question. By default the notebook uses dense search over semantic chunks, with reranking and question splitting (decomposition) switched off.

The model gets up to five passages, within a budget of about 2,200 tokens. Before it sees them, a check looks at the best similarity score and at whether a short fixed list of terms (a few framework and hardware names, plus numeric precisions like INT8) appears in the evidence. If the check fails, you see the closest sources and the reason it declined. Otherwise Qwen 2.5 3B, running locally through Ollama, has to return claims, each with a source ID and an exact quote from that source. A rejected draft gets one repair attempt. After that, you get labelled excerpts instead of a made-up answer.

## Results from the saved CPU run

Eight of the nine requested documents ended up in the index. The OpenCV tutorial returned HTTP 403, so nothing in the index could satisfy its two evidence labels, but its question still counts in the scores below. The index isn't only documentation, either. About a quarter of its chunks come from a PyTorch research paper (arXiv 1912.01703, a PDF). The extractor read the paper's title as "Sam Gross", and the paper turned up as the second keyword result for the export-and-deploy question.

The table covers the eight questions labelled answerable, using semantic chunks. Each question has evidence labels, which are a source plus a required term, such as the ONNX Runtime site and `InferenceSession`. Recall@5 is the fraction of a question's labels matched in the first five results, averaged across questions. MRR reflects how early the first match showed up. The labels are strict, so a good passage from a different page scores zero. The export question scores zero at five results with every method, partly for that reason.

| Search method | Recall@5 | Recall@10 | MRR | Mean search time, including reranking |
| --- | ---: | ---: | ---: | ---: |
| Keyword | 0.438 | 0.531 | 0.534 | 10 ms |
| Dense | **0.656** | 0.656 | 0.604 | 71 ms |
| Hybrid | 0.531 | 0.656 | 0.501 | 79 ms |
| Dense + reranker | 0.469 | **0.688** | **0.656** | 6.4 s |
| Hybrid + reranker | 0.469 | 0.594 | 0.652 | 6.2 s |

Five results is the cutoff that matters, because that's all the model sees. The reranker moved the first match closer to the top on average (MRR 0.656 against 0.604), but it took about 6 seconds a query on CPU instead of 71 milliseconds, and with only five slots it covered less of the labelled evidence than plain dense search (0.469 against 0.656). Leave out the OpenCV question and dense Recall@5 is 0.750 across the remaining seven.

Semantic chunking was the slower build: about eight minutes for 208 chunks, against just under two minutes for 243 parent–child chunks. With dense search it gained at five results (0.531 to 0.656) and tied at ten. It didn't help the other methods at five, and parent–child had the better dense MRR (0.657 against 0.604). The comparison isn't perfectly even either. Parent–child hits are matched against their whole parent section, while semantic hits are matched on the chunk alone.

### What came back to the reader

Across the ten questions, **two answers passed the checks, five fell back to excerpts, and three stopped before generation.** Both passing answers were about ONNX Runtime sessions, including the provider example above. One of the fallbacks was the ONNX validation question. The final context held the expected evidence, but the model still couldn't produce a draft that passed. All ten questions took about 41 minutes on a Colab CPU, almost all of it generation, which came to about 5.8 minutes for each of the seven that reached the model.

The three abstentions were the missing OpenCV tutorial and two requests the docs don't cover: Qualcomm Hexagon INT4, and TensorRT INT8 calibration caches. All three were the right call, but all three came from the fixed term list, not from low similarity. Both unsupported questions scored above the similarity threshold (0.64 and 0.73 against 0.50), so without the term list they would have gone to the model. That list contains those questions' own terms, so it says little about how the notebook handles a gap in a question about a familiar API. There's also no human-rated accuracy score for this run, only retrieval labels and format checks.

### What I'd try next

The first test I'd add reuses the provider example: two adjacent sentences, and a check that the answer keeps the CPU exception. The export-and-deploy question has a different problem, because only one of its four evidence labels made it into the final context. The notebook already has an optional step that splits a question into parts and searches for each one. In the dense path, though, that step can only reorder passages the original question already found, so it can't add new evidence yet. That's where I'd start.

## Rough edges

- The `gap_flagged` column in the evaluation table is true even when the model reports no gaps, because it matches the standard explanatory text. Ignore it for now.
- Chunk size limits are soft. Long sentences or punctuation-heavy text can produce chunks well over the 350-token target, and the counts are a word-and-punctuation estimate, not the model's tokenizer.
- Question splitting only turns on through the global `cfg.use_query_decomposition` setting. Passing it for a single call doesn't override a disabled global.
- Dependencies aren't pinned, and the source pages can change between runs.

## Try it in Colab

Open the notebook in Google Colab and run the cells in order through the definition of `ask`. The default setup installs Ollama, starts it and downloads the model, so you don't need an API key. If you only want to explore retrieval, uncomment `cfg.llm_provider = "none"` in the **LLM choice & keys** cell before running the component initialisation cell. Once the documents are indexed, try this in a new cell and expand the sources under the answer:

```python
ask("Do I have to specify execution providers when creating an ONNX Runtime session?")
```

It's slow on CPU. In the saved run a generated answer took between about 1 and 10 minutes. The notebook suggests a GPU runtime for faster generation, but no GPU run is recorded here.

The evaluation cells write per-question scores, full answer traces (including why rejected drafts failed) and a blank human-review sheet to `/content/edgedocs/results`. Those files aren't in this repo, and Colab deletes them when the session ends, so download anything you want to keep.

## How this was made

This project was created with AI assistance. AI coding assistants helped write and debug the notebook and drafted this README. The figures here come from the run saved in the notebook. There are no automated tests, and the evaluation cells in the notebook are the only checks.
