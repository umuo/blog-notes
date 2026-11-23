# 退款知识库实验

示例使用合成退款资料与公开评测数据。Python 检索、抽取、Wiki、SQL 实验使用标准库。扫描实验使用 macOS 自带的 PDFKit 和 Vision。纯 Python 的向量、重排和回答器使用可检查的规则。

```bash
python3 experiments/rag-knowledge/retrieval_lab.py
python3 experiments/rag-knowledge/workflow_lab.py
python3 experiments/rag-knowledge/sql_lab.py
xcrun swiftc -framework PDFKit -framework Vision experiments/rag-knowledge/layout_lab.swift -o /tmp/refund-layout
/tmp/refund-layout experiments/rag-knowledge/scanned-dictionary.pdf
```

## 公开基准的复跑

`curl` 下载数据和官方评分器。DuckDB 只用于读取两份 Parquet（列式数据文件），ujson 供 HotpotQA 官方脚本读取 JSON。测试只运行 SciFact 的 10 个查询和 HotpotQA 的 5 道验证题。仓库只保留题目 ID、指标和输入摘要值。

```bash
data_dir=$(mktemp -d)
uv venv --python 3.13 "$data_dir/venv"
uv pip install --python "$data_dir/venv/bin/python" duckdb==1.4.4 ujson==5.12.0
curl -fLsS 'https://huggingface.co/api/datasets/BeIR/scifact/parquet/corpus/corpus/0.parquet' -o "$data_dir/corpus.parquet"
curl -fLsS 'https://huggingface.co/api/datasets/BeIR/scifact/parquet/queries/queries/0.parquet' -o "$data_dir/queries.parquet"
curl -fLsS 'https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/main/test.tsv' -o "$data_dir/test.tsv"
curl -fLsS 'https://datasets-server.huggingface.co/rows?dataset=hotpotqa%2Fhotpot_qa&config=distractor&split=validation&offset=0&length=5' -o "$data_dir/hotpot-rows.json"
curl -fLsS 'https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py' -o "$data_dir/hotpot_evaluate_v1.py"
"$data_dir/venv/bin/python" experiments/rag-knowledge/benchmark_lab.py "$data_dir"
```

如果公开数据或脚本更新，先与存档结果的 `inputs_sha256` 对照，再比较指标。
