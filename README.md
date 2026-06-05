# MAGMA

MAGMA 鏄潰鍚?OpenClaw 澶?Agent 绯荤粺鐨勮法浼氳瘽銆佽法 Agent 璁板繂鏋舵瀯锛屽叏绉颁负 Multi-Graph Adaptive Memory Architecture銆?
瀹冭礋璐ｆ妸瀵硅瘽浜嬩欢銆佸疄浣撱€佸叧绯汇€佸悜閲忋€佸彫鍥炶褰曞拰鍙嶉鍐欏叆鏈湴 SQLite + FAISS 璁板繂灞傦紝骞堕€氳繃 `magma-recall` OpenClaw 鎻掍欢锛屽湪 Agent 鏋勫缓鎻愮ず璇嶅墠鑷姩娉ㄥ叆鐩稿叧璁板繂銆?
![MAGMA 澶氬浘璋辨櫤鑳戒綋璁板繂鏋舵瀯](docs/assets/magma-hero.jpg)

## 鍥捐В

### 1. 鏋舵瀯鍏ㄦ櫙锛氫笁灞傝В鑰︾殑璁板繂澶ц剳

![MAGMA 鏋舵瀯鍏ㄦ櫙](docs/assets/magma-architecture-overview.jpg)

### 2. 鎰忓浘鎰熺煡璺敱锛氳涔夈€佸叧閿瘝銆佹椂闂翠俊鍙疯瀺鍚?
![MAGMA 鎰忓浘鎰熺煡璺敱](docs/assets/magma-intent-router.jpg)

### 3. 鍩哄噯鏁堟灉锛氬鏉傛帹鐞嗗満鏅笅鐨勭ǔ瀹氬彫鍥?
![MAGMA 鍩哄噯鏁堟灉](docs/assets/magma-benchmark.jpg)

## 褰撳墠杩愯鎬?
- Embedding 妯″瀷锛氭湰鍦?`Qwen3-Embedding-0.6B`
- Embedding 缁村害锛?024
- 鎱㈣矾寰?LLM 鍚庣锛欴eepSeek V3 via OpenRouter
- 涓?API锛歚http://127.0.0.1:8904`
- MCP 鏈嶅姟锛氭寚鍚?8904 涓?API 鐨勮杽浠ｇ悊

LLM 鍚庣鍜?embedding 妯″瀷鏄袱濂椾笢瑗匡紝涓嶈兘娣风敤銆侱eepSeek V3 鐢ㄤ簬鎱㈣矾寰勫叧绯绘娊鍙栥€佸洜鏋滄帹鏂拰璁板繂宸╁浐锛沗Qwen3-Embedding-0.6B` 鐢ㄤ簬鏈湴璇箟鍚戦噺鍙洖銆傚巻鍙蹭笂鐨?MiniLM-L6-v2 / 384 缁村悜閲忓拰 bge-small-zh-v1.5 / 512 缁村悜閲忎笉鏄綋鍓嶈繍琛屾€併€?
## 鍙敤鎬ц竟鐣?
杩欎釜浠撳簱鍖呭惈 MAGMA 鐨勪唬鐮併€丱penClaw 鎻掍欢鍜岃繍缁磋剼鏈紝浣嗕笉鍖呭惈浠讳綍鏈湴杩愯鏁版嵁銆傞娆″惎鍔ㄤ細鍒涘缓涓€涓┖鏁版嵁搴擄紱瑕佺湅鍒扮湡瀹炲彫鍥炴晥鏋滐紝闇€瑕侀€氳繃 API銆丮CP 鎴?OpenClaw 鎻掍欢鎸佺画鍐欏叆璁板繂銆?
寮€绠卞嵆鐢細

- FastAPI API 鏈嶅姟
- SQLite 琛ㄧ粨鏋勮嚜鍔ㄥ垵濮嬪寲
- 鏈湴 embedding 缂栫爜
- 鎵嬪姩鍐欏叆銆佹煡璇€丮CP 浠ｇ悊
- doctor / ops / governance 杩愮淮鑴氭湰

闇€瑕佸閮ㄧ郴缁燂細

- OpenClaw 鑷姩鍙洖鍜岃嚜鍔ㄦ姄鍙栭渶瑕佸畨瑁?`openclaw-plugin-magma-recall`
- 鎱㈣矾寰?LLM 宸╁浐闇€瑕佷綘鑷繁閰嶇疆鍙敤鐨?LLM 鍚庣
- 鐢熶骇鏁版嵁銆丗AISS 绱㈠紩鍜屽浠戒笉浼氶殢浠撳簱鍙戝竷

## 鏍稿績鑳藉姏

- 瀵硅瘽鍓嶈嚜鍔ㄥ彫鍥炶蹇?- Agent 鍥炲鍚庤嚜鍔ㄥ啓鍏?L0 鍘熷璁板繂
- 閫氳繃 `source_agent_id` 鍜?`department` 璁板綍璺?Agent 鏉ユ簮
- 璇箟鍒嗘暟 + 涓枃鍏抽敭璇?+ 鐢熷懡鍛ㄦ湡鏉冮噸鐨勭粺涓€妫€绱?- 鍙洖鍙嶉闂幆鍜?importance 鍔ㄦ€佹洿鏂?- 杩愮淮閿氱偣锛岀‘淇濆叧閿郴缁熶簨瀹炵ǔ瀹氬彫鍥?- 绾㈤粍缁垮仴搴疯瘖鏂?- 淇濆畧璁板繂娌荤悊锛氶粯璁?dry-run锛宎pply 鍙仛杞不鐞?- MCP 宸ュ叿鍏煎锛屽苟缁熶竴璧?8904 涓婚摼璺?
## 鐩綍缁撴瀯

```text
magma/
  api/
    server.py              # FastAPI 鏈嶅姟
    mcp_server.py          # MCP 钖勪唬鐞?  graph/
    sqlite_store.py        # SQLite 鍥捐氨/璁板繂瀛樺偍
  vector/
    encoder.py             # sentence-transformers 缂栫爜鍣?  backup.py
  entities.py
  search.py

openclaw-plugin-magma-recall/
  index.js                 # OpenClaw hook 鎻掍欢
  openclaw.plugin.json
  package.json

scripts/
  magma_doctor.py          # 鍋ュ悍璇婃柇
  magma_ops.py             # 鐘舵€佹鏌ュ拰瀹夊叏淇妫€鏌?  magma_governance.py      # dry-run / 杞不鐞?  magma_recall_eval.py     # 鍙洖璐ㄩ噺璇勬祴
  migrate_source_agent.py  # 鏉ユ簮褰掑洜杩佺Щ
  seed_operational_anchors.py
  magma_cli.py

RUNBOOK.md
HANDOFF_OPENCLAW.md
璧锋簮.md
```

## 瀹夎

```powershell
cd C:\openclaw-magma
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

濡傛灉 Hugging Face 璁块棶鎱㈡垨涓嶅彲鐢紝鍙互鍦ㄩ娆″姞杞芥ā鍨嬪墠璁剧疆 `HF_ENDPOINT`銆?
鍙互鍙傝€?`.env.example` 閰嶇疆绔彛銆佹暟鎹簱璺緞銆乪mbedding 妯″瀷鍜屽悗鍙颁换鍔￠棿闅斻€?
## 鍚姩 API

```powershell
cd C:\openclaw-magma
python -m magma.api.server
```

榛樿鐩戝惉 `127.0.0.1:8904`銆傞娆″惎鍔ㄤ細鑷姩鍒涘缓 `data/magma.db`锛屼笉闇€瑕佹墜鍔ㄥ垵濮嬪寲鏁版嵁搴撱€?
鍚姩鍚庨獙璇侊細

```powershell
Invoke-RestMethod http://127.0.0.1:8904/api/v1/health
python scripts\magma_doctor.py --quick
```

鍐欏叆涓€鏉℃祴璇曡蹇嗭細

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8904/api/v1/nodes `
  -ContentType "application/json" `
  -Body '{"id":"demo:hello","label":"event","properties":{"content":"MAGMA demo memory","source":"demo","importance":0.5}}'
```

鏌ヨ娴嬭瘯璁板繂锛?
```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8904/api/v1/query `
  -ContentType "application/json" `
  -Body '{"query":"demo memory","top_k":3}'
```

## OpenClaw 闆嗘垚

OpenClaw 鎻掍欢浣嶄簬锛?
```text
openclaw-plugin-magma-recall/
```

绀轰緥閰嶇疆锛?
```json
{
  "plugins": {
    "entries": {
      "magma-recall": {
        "path": "C:\\openclaw-magma\\openclaw-plugin-magma-recall",
        "config": {
          "enabled": true,
          "apiBaseUrl": "http://127.0.0.1:8904",
          "topK": 6,
          "timeoutMs": 12000,
          "scoreThreshold": 0.35,
          "magmaCwd": "C:\\openclaw-magma",
          "embeddingModel": "C:\\openclaw-magma\\models\\Qwen\\Qwen3-Embedding-0___6B",
          "apiEnv": {
            "MAGMA_EMBEDDING_MODEL": "C:\\openclaw-magma\\models\\Qwen\\Qwen3-Embedding-0___6B"
          },
          "capture": {
            "enabled": true,
            "ttlDays": 180,
            "maxChars": 4000
          }
        },
        "hooks": {
          "allowConversationAccess": true
        }
      }
    }
  }
}
```

鎻掍欢娉ㄥ唽涓変釜 hook锛?
- `before_prompt_build`锛氬璇濆墠鑷姩鍙洖骞舵敞鍏ヨ蹇?- `agent_end`锛氬璇濈粨鏉熷悗鑷姩鎶撳彇 L0 璁板繂锛屽苟鍋氬急姝ｅ弽棣?- `before_message_write`锛氬啓鍏ュ巻鍙插墠鍓ョ娉ㄥ叆鐨勮蹇嗗潡锛屼繚鎸佸巻鍙插共鍑€

MCP 鍏ュ彛浠嶇劧鍏煎锛?
```powershell
python -m magma.api.mcp_server
```

鎵€鏈?MCP 宸ュ叿閮戒細浠ｇ悊鍒?`http://127.0.0.1:8904/api/v1/...`锛屼笉浼氬啀鍦?stdio 杩涚▼閲岀洿鎺ュ姞杞?SQLite 鍜?embedding 妯″瀷銆?
## 杩愮淮鍛戒护

```powershell
# 鍋ュ悍妫€鏌?python scripts\magma_doctor.py --json

# 绠€鐭姸鎬?python scripts\magma_ops.py status

# 瀹夊叏淇妫€鏌?python scripts\magma_ops.py repair

# 璁板繂娌荤悊 dry-run
python scripts\magma_governance.py --dry-run --json

# 杞不鐞?apply
python scripts\magma_governance.py --apply --json

# 鍙洖璐ㄩ噺璇勬祴
python scripts\magma_recall_eval.py

# Qwen3 embedding 鏃佽矾璇勪及锛屼笉淇敼鐢熶骇鏁版嵁
python scripts\qwen_embedding_probe.py --model .\models\Qwen\Qwen3-Embedding-0___6B

# Qwen3 embedding 鐢熶骇鍒囨崲鍓嶇殑鐙珛璇曠敤搴撴瀯寤?python scripts\qwen_embedding_trial.py --port 8905

# Qwen3 reranker 鏃佽矾璇勪及锛屼笉鎺ュ叆瀹炴椂鍙洖
python scripts\qwen_reranker_probe.py --candidate-k 20 --top-k 6
```

## 鍙洖璐ㄩ噺绛栫暐

MAGMA 鐨勫疄鏃跺彫鍥為摼璺粯璁や繚鎸佽交閲忥細

1. `Qwen3-Embedding-0.6B` 璐熻矗绗竴闃舵鍚戦噺鍙洖銆?2. 涓枃鍏抽敭璇嶃€佺敓鍛藉懆鏈熴€乻ource agent銆乮mportance 鍜?current-state 淇″彿鍋氳鍒欓噸鎺掋€?3. `ops_anchor`銆乣L1`銆乣current_state` 绛夐珮瀵嗗害璁板繂鍦ㄨ繍缁村拰鏋舵瀯绫婚棶棰樹腑浼樺厛銆?4. L0 鍘熷瀵硅瘽淇濈暀涓鸿瘉鎹紝浣嗕笉浼氳交鏄撳帇杩囬珮瀵嗗害璁板繂銆?
`Qwen3-Reranker-0.6B` 鐩墠鍙綔涓烘梺璺瘎浼板伐鍏蜂娇鐢ㄣ€傚疄娴?CPU 鐜涓?reranker 瀵?top20 鍊欓€夌簿鎺掍細杈惧埌鏁板崄绉掔骇锛屼笉閫傚悎鏀捐繘 `before_prompt_build` 瀹炴椂閾捐矾锛涘畠鏇撮€傚悎绂荤嚎璐ㄩ噺瀹¤銆佹參璺緞娌荤悊鍜屽悗缁?L1 鎻愮偧璇勪及銆?
## 鏁版嵁瀹夊叏

杩愯鏁版嵁涓嶄細鎻愪氦鍒?Git锛?
- `data/`
- `*.db`銆乣*.db-shm`銆乣*.db-wal`
- `*.index`
- `backups/`
- `models/`
- `logs/`
- `.env`

涓嶈鎻愪氦鏈湴璁板繂鏁版嵁搴撱€丗AISS 绱㈠紩銆丱penClaw 鍑嵁銆侀涔﹀瘑閽ャ€丱penRouter/OpenAI key 鎴?token 鏂囦欢銆?
## 璁稿彲璇?
MIT
