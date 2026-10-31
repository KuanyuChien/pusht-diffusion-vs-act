# Phase 3–5 報告：驗證流程、探索資料、開始訓練

**日期：** 2026-10-27
**結果：**
- Phase 3：完成。官方模型在我們的流程裡得到 59.4% 成功率（500 場），作為之後所有比較的基準。
- Phase 4：完成。產生 5 張圖，弄清楚資料集的內容和 action 的意義。
- Phase 5：smoke test 通過；200k 步的正式訓練（job 2260235）18:15 開始，預計約 3.5 小時完成。

今天遇到並解決了 **6 個問題**（problem log 第 3–8 條），其中兩個屬於「不會報錯、但會讓結果出錯」的靜默錯誤。

---

## 1. 摘要

| Phase | 目標 | 結果 |
|---|---|---|
| 3 | 用官方模型驗證 eval 流程 | ✅ 10 場 80%；500 場 **59.4%**（CI95 55.0–63.6），`avg_max_reward` 0.941。官方公布 65.4% / 0.955 |
| 4 | 了解資料集 | ✅ 206 個示範、中位數 122 步；action 是「目標點」；目標 T 位置固定 |
| 5 | 訓練自己的 Diffusion Policy | 🔄 smoke test 通過（訓練、存檔、eval、續跑）；正式訓練進行中 |

---

## 2. Phase 3：驗證 eval 流程

### 2.1 步驟

| # | 做了什麼 | 結果 |
|---|---|---|
| 1 | 在 LeRobot 原始碼找處理舊格式模型的工具 | 找到 `processor/migrate_policy_normalization.py` |
| 2 | 第一次轉換官方模型 | ❌ `Couldn't encode 84`（問題 5） |
| 3 | 寫包裝腳本 `pusht/tools/migrate_official.py` 修正型別 | ⚠️ 模型存好了，但產生說明卡的步驟被離線模式擋下 |
| 4 | 在登入節點、不設離線模式重跑 | ✅ `all keys matched`，轉換乾淨 |
| 5 | 確認 eval 時的影像尺寸 | ✅ 模型看到 96×96；384×384 只用於錄影片 |
| 6 | 寫 `pusht/jobs/eval_official.slurm`，跑 10 場 | ❌ `ConnectionResetError`（問題 6） |
| 7 | 加 `--eval.use_async_envs=false` 後重跑 | ✅ 8/10 成功 |
| 8 | 跑 500 場 | ✅ 59.4%，12 分鐘 |

### 2.2 結果

| | 成功率 | 95% 信賴區間 | `avg_max_reward` |
|---|---|---|---|
| 我們（500 場，job 2259854） | **59.4%** | 55.0–63.6% | 0.941 |
| 官方公布 | 65.4% | 約 61.2–69.5% | 0.955 |

**解讀：**
- **流程是正確的：** 如果正規化或影像尺寸出錯，成功率會掉到個位數。
- **約 6 個百分點的差距勉強達到統計顯著。** 可能原因：軟體版本、模型格式轉換、GPU 數值精度、隨機種子。目前都還沒驗證。
- **決定：** 以 59.4% 作為基準，因為之後的 DP 和 ACT 都用同一套流程評估。

結果檔：`pusht/results/official_dp_500ep/eval_info.json`

---

## 3. Phase 4：探索資料集

腳本：`pusht/analysis/explore_dataset.py`、`pusht/analysis/zoom_trajectory.py`
輸出：`pusht/results/phase4/`（`summary.json` + 5 張圖）

### 3.1 關鍵數字

| 項目 | 數值 |
|---|---|
| 示範數 | 206 episodes、25,650 frames、10 fps |
| episode 長度 | 49–246 步，中位數 122 步（12.2 秒）；eval 上限 300 步，足夠 |
| action 每一步移動 | 中位數 8 px，99% 在 40 px 以內，最大 137 px |
| action 領先 agent | 中位數 19 px，90% 在 46 px 以內 |

### 3.2 發現

1. **action 是「目標點」，不是速度。** agent 由 PD 控制器（`k_p=100`、`k_v=20`）拉向目標點，每個 action 執行 10 個 0.01 秒的物理小步。
2. **action 來自人類的滑鼠位置**，所以是連續的小步移動。規則上 action 可以是 0–512 的任何點，但資料裡從沒出現大跳躍。
3. **目標 T 的位置和角度固定**，只有灰色 T 和 agent 的起始位置不同。
4. **推法有很多種（多模態）。** 人會繞到 T 的不同側推，這是 DP 預期比 ACT 好的原因。
5. **資料集沒有成功標籤**：reward 最高 0.949，`next.success` 全部是 False。對模仿學習沒有影響，因為不使用 reward。
6. **讀軌跡圖：** 點越密 = 越慢（推 T）；點越疏、灰線越長 = 越快（繞路）；轉彎時藍線尖、橘線圓 = 慣性。

---

## 4. Phase 5：訓練 Diffusion Policy

### 4.1 準備

| # | 做了什麼 | 結果 |
|---|---|---|
| 1 | 讀 `TrainPipelineConfig` 的欄位 | `--eval_freq` 已改名為 `--env_eval_freq` |
| 2 | 讀官方的 `train_config.json` | batch 64、200k 步、每 25k 存檔和 eval、seed 100000、Adam lr 1e-4、cosine 加 500 步 warmup |
| 3 | **逐一比對官方 `config.json` 和 `DiffusionConfig()` 預設值** | 發現 6 個不同的值（問題 8） |
| 4 | 讀續跑和 output 資料夾的規則 | 用 `--resume=true --config_path=...`；資料夾已存在又沒加 resume 時會報錯，不會覆蓋 |
| 5 | 寫 `pusht/jobs/train_dp.slurm` | 有參數、會自動續跑、明確指定官方設定 |

### 4.2 Smoke test

| Job | 指令 | 結果 |
|---|---|---|
| 2260220 | `train_dp.slurm dp_smoke 1000 500 10` | ❌ 31 秒失敗：缺 `accelerate`（問題 7） |
| — | 補裝 `[training]` extra | ✅ `pip check` 沒有衝突 |
| 2260228 | 同上，從頭訓練 1,000 步 | ✅ 3 分鐘；存檔和 eval 各兩次 |
| 2260231 | `train_dp.slurm dp_smoke 2000 500 10` | ✅ 自動從第 1000 步續跑到 2000 步，資料順序也接上了 |

**量到的數字：**

| 項目 | 數值 |
|---|---|
| 速度 | 0.059 秒/步（約 1,080 樣本/秒）→ 200k 步約 3.3 小時 |
| GPU 記憶體 | 5 GB（A800 有 40 GB） |
| 每個 checkpoint | 3.0 GB → 8 個約 24 GB（放 scratch） |
| loss | 0.536（第 100 步）→ 0.045（第 2,000 步） |
| eval（2,000 步） | 成功率 0%，`avg_max_reward` 0.19 → 0.37：開始會推了，但還不準（初期的正常現象） |

### 4.3 正式訓練

| 項目 | 值 |
|---|---|
| 指令 | `sbatch jobs/train_dp.slurm dp_run1 200000 25000 50` |
| Job | 2260235，grn023，18:15 開始，時間上限 8 小時 |
| 預計完成 | 約 21:45–22:00 |
| 輸出 | `/scratch/general/vast/u1470293/pusht/outputs/train/dp_run1/` |
| 撰寫報告時 | 第 5,000 步左右，loss 0.042，速度和 smoke test 一致 |

---

## 5. 遇到的問題（problem log 第 3–8 條）

> 第 1–2 條（QOS、SSH）見 `phase1-2_report.md`。

### 問題 5：轉換官方模型 → `Exception: Couldn't encode 84`

| | |
|---|---|
| **現象** | 轉換腳本報錯，`config.json` 是 0 bytes |
| **原因** | LeRobot 轉換腳本的 bug：JSON 只有 list，`crop_shape=[84, 84]` 沒有轉成程式規定的 tuple，存檔時失敗 |
| **怎麼找到的** | traceback 指向存設定檔那一行 → `grep "84"` 找到 `crop_shape` → 讀腳本，發現它只幫 `dtype` 做了型別轉換 |
| **解法** | 包裝腳本 `pusht/tools/migrate_official.py`，用 monkey patching 先轉換型別。LeRobot 原始碼不動 |
| **附帶問題** | 最後產生說明卡時，`card.validate()` 要連網驗證 YAML，被 `HF_HUB_OFFLINE=1` 擋下 → 在登入節點拿掉離線模式重跑。另外確認了訓練存 checkpoint 時**不會**呼叫它（只有上傳到 Hub 時才會） |
| **學到** | 不修改別人的程式碼；遇到問題時順便檢查「別的地方會不會也有同樣的問題」 |

### 問題 6：`lerobot-eval` → `ConnectionResetError`

| | |
|---|---|
| **現象** | eval 在建立模擬器時失敗，最後一行像是網路錯誤 |
| **真正原因** | log 更前面的 `NamespaceNotFound: gym_pusht`。平行的子行程用 `forkserver` 全新啟動，沒有 import `gym_pusht` |
| **怎麼找到的** | `grep -n -i "error\|exception"` 搜尋整份 log → 讀 `envs/configs.py` 建立環境的程式碼 |
| **解法** | `--eval.use_async_envs=false`，和官方 eval 的設定一致 |
| **學到** | 最後一行錯誤不一定是原因；多行程的程式要找最早、最具體的錯誤 |

### 問題 7：第一次訓練 → `ImportError: 'accelerate' is required`

| | |
|---|---|
| **現象** | smoke test 31 秒就失敗 |
| **原因** | `lerobot-train` 需要 `[training]` extra（`accelerate` + `wandb`）。Phase 2 只檢查了 eval 需要的東西 |
| **解法** | `pip install -e ".[pusht,diffusion,training]"` |
| **學到** | smoke test 用最小的成本抓到安裝漏洞 |

### 問題 8：DP 預設設定 ≠ 官方設定（靜默錯誤）

| | |
|---|---|
| **現象** | 不會報錯。如果沒發現，訓練出來的模型會和官方差很多，而且不知道為什麼 |
| **原因** | LeRobot 後來把 DP 的預設值改成適合真實機器人的設定 |
| **怎麼找到的** | 寫一小段 Python，讀出官方 `config.json` 和 `DiffusionConfig()` 的所有欄位，只列出不同的 |
| **解法** | 在 `train_dp.slurm` 明確指定官方的值（見下表） |
| **學到** | 預設值不等於正確的值；要重現結果，就拿官方設定檔逐一比對 |

| 設定 | 官方 | 0.6.2 預設 |
|---|---|---|
| `crop_shape` | [84, 84] | None（不裁切） |
| `horizon` | 16 | 64 |
| `n_action_steps` | 8 | 32 |
| `pretrained_backbone_weights` | None | ImageNet（離線時也會下載失敗） |
| `use_group_norm` | True | False |
| `use_separate_rgb_encoder_per_camera` | False | True |

### 其他已確認、不是問題的事

| 事項 | 結論 |
|---|---|
| 環境設定寫 384×384 | 只是錄影片的解析度，模型看到的是 96×96 |
| 資料集沒有成功標籤 | 模仿學習不使用 reward |
| 官方模型 59.4% vs 65.4% | 流程正確；差距的原因待查，不影響專案主線 |
| 續跑時改 `--steps` 會重建 learning rate 排程 | 只發生在 smoke test；正式訓練不改總步數 |

---

## 6. 今天新增的檔案

```
~/projects/pusht/
├─ tools/migrate_official.py          修正版的模型轉換
├─ jobs/eval_official.slurm           評估官方模型：sbatch eval_official.slurm <場數>
├─ jobs/train_dp.slurm                訓練 DP：sbatch train_dp.slurm <run> <步數> <頻率> <eval 場數>
├─ analysis/explore_dataset.py        資料集統計 + 圖 1–4
├─ analysis/zoom_trajectory.py        單一 episode 放大圖（圖 5）
├─ results/official_dp_500ep/         官方模型 500 場結果
├─ results/phase4/                    summary.json + 5 張圖
└─ reports/phase3-5_report.md         本報告

/scratch/general/vast/u1470293/pusht/
├─ models/diffusion_pusht_migrated/   轉換後的官方模型
├─ outputs/eval/                      eval 結果和影片
└─ outputs/train/
   ├─ dp_smoke/                       smoke test 的 checkpoint（約 12 GB，可以刪除）
   └─ dp_run1/                        正式訓練（進行中）
```

環境新增的套件：`matplotlib`、`accelerate`、`wandb`（`[training]` extra）。

---

## 7. 下一步

1. **明天：** 確認 `dp_run1` 完成，看 8 個 checkpoint 的成功率。
2. **Phase 6：** 選最好的 checkpoint 跑 500 場 eval，畫「成功率 vs. 訓練步數」曲線，和基準 59.4% 比較。
3. **Phase 7（ACT）：** 先用同樣的方法比對 ACT 的官方設定和預設值，再做 smoke test、正式訓練。
