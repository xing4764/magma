# MAGMA 4B 重编码 - 宿主机运行说明

## 需要复制的文件

```
C:\openclaw-magma\
├── magma_reencode_4b_gpu.py    ← 重编码脚本
├── data\
│   └── magma.db                ← 数据库（必须）
└── models\Qwen\
    └── Qwen3-Embedding-4B\     ← 4B模型（如果宿主机已有可跳过）
```

## 宿主机准备

```bash
# 1. 安装依赖
pip install sentence-transformers faiss-gpu numpy

# 2. 把文件放到脚本同目录下
#    - data/magma.db
#    - models/Qwen/Qwen3-Embedding-4B/ （或修改脚本里的 MODEL_PATH）

# 3. 运行
python magma_reencode_4b_gpu.py
```

## 预期输出

```
============================================================
MAGMA 4B Batch Re-encoding (GPU)
============================================================
[1/4] Loading Qwen3-Embedding-4B...
  Model loaded. Dimension: 2560
  Active nodes: 5932
  Already encoded: 754
  Need encoding: 5178

[2/4] Batch encoding 5178 nodes (batch=100)...
  [500/5178 10%] 150.0n/s, 3s elapsed, ETA 31s
  ...
  Encoding done: 5178 ok, 0 err, 35.2s

[3/4] Rebuilding FAISS index (dim=2560)...
  FAISS: 5932 vectors, dim=2560

[4/4] Verification...
  Nodes with embedding: 5932/5932

============================================================
DONE! Total: 40s
============================================================
```

## 完成后

把 `data/magma.db` 和 `data/faiss.index` 拷回服务器 `C:\openclaw-magma\data\` 即可。
