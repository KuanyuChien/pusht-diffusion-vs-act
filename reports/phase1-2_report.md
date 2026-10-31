# Phase 1–2 報告：CHPC 入門與環境建置

**期間：** 2026-10-21 至 2026-10-27
**結果：** Phase 1、2 完成。環境在 GPU 節點上以離線模式通過全部檢查，可以進入 Phase 3（評估官方模型）。

---

## 1. 摘要

| Phase | 目標 | 結果 |
|---|---|---|
| 1 | 學會在 CHPC 送 GPU job | ✅ 在 lab 的 A800 上跑過短 job、25 分鐘 job、互動檢查、取消 job |
| 2 | 建好 LeRobot 環境、下載資料和模型 | ✅ conda env、LeRobot 0.6.2、PushT 資料集、官方 Diffusion Policy 都準備好，GPU 檢查 job 通過 |

過程中解決了 **4 個問題**，另有 **1 個待處理的警告**（留到 Phase 3），見第 4 節。

---

## 2. Phase 1：CHPC 與 Slurm 入門

### 2.1 確認可用資源

用 `myallocation`、`sinfo`、`sacctmgr` 查出可用的資源：

| 項目 | 結果 |
|---|---|
| Cluster | `granite`（原計畫寫的是 notchpeak，已修正） |
| Account | `kmarino` |
| Lab 自有 partition | `kmarino-gpu-grn`，**必須搭配** `--qos=kmarino-gpu-grn` |
| Lab GPU 節點 | grn023：8× A800 40GB（通常有空位）；grn067：4× H200 NVL（常被組員佔滿） |
| 備用 | `granite-gpu-guest`（別的 lab 的閒置 GPU，可能被中斷） |

### 2.2 跑過的 job

| Job ID | 內容 | 結果 | 學到什麼 |
|---|---|---|---|
| 2210984 | `gpu_hello`：`nvidia-smi` 加上用 CUDA 程式加總 100 萬個數 | ✅ 約 20 秒就開始，在 A800 上成功 | 最基本的 `sbatch` 流程、log 檔 |
| 2211113 | `gpu_long`：讓 GPU 滿載 25 分鐘 | 跑了 10.5 分鐘後由我們手動 `scancel` | `squeue`、`srun --overlap nvidia-smi`（GPU 使用率 100%）、`scancel`、`sacct` |
| 2255442 | 從 GPU 節點連外網測試 | ✅ huggingface.co 和 pypi 都回 HTTP 200 | **compute node 有網路**（原計畫說沒有，已修正） |

### 2.3 學會的 GPU 使用方式

| 方式 | 適合 | 備註 |
|---|---|---|
| `sbatch` | 正式訓練、eval | 關掉電腦也會繼續跑 |
| `salloc` | 終端機 debug | 要記得 `exit` |
| Open OnDemand：VS Code Server | 在瀏覽器裡開一個跑在 GPU 節點上的 VS Code | 關掉分頁**不會**釋放 GPU，要按 Delete |
| SSH 或 VS Code Remote-SSH 加上 placeholder job | 用自己電腦的 VS Code 連到 GPU 節點 | 先送 `hold_gpu.slurm`，結束時要 `scancel` |

### 2.4 儲存空間與權限

| 位置 | 容量 | 誰看得到 | 用途 |
|---|---|---|---|
| `~/`（home） | 49.4 GiB，有備份 | 自己可讀寫；lab 成員和 CHPC 管理員可讀 | 程式碼、環境、最終成果 |
| `/scratch/general/vast/u1470293/` | 系統共 1.5 PB，**會清除很久沒用的檔案** | 所有 CHPC 使用者可讀 | 資料集、checkpoint |

原則：**scratch 是工作檯，home 是保險箱**。token 和密碼不要寫進一般檔案。

---

## 3. Phase 2：環境建置

### 3.1 做了什麼

| # | 步驟 | 結果 |
|---|---|---|
| 1 | 建立資料夾：`~/projects/pusht/{jobs,logs,results}`、scratch 上的 `pusht/` | ✅ |
| 2 | `git clone --depth 1` LeRobot | ✅ v0.6.2，commit `dd5313e`，27 MB |
| 3 | 讀 `pyproject.toml` | 發現需要 **Python ≥ 3.12**；Diffusion Policy 要加 `diffusion` extra |
| 4 | `conda create -p ~/envs/lerobot python=3.12` | ✅ Python 3.12.16 |
| 5 | `conda install ffmpeg -c conda-forge` | ✅ ffmpeg 5.1（有 `libsvtav1`） |
| 6 | `pip install -e ".[pusht,diffusion]"` | ✅ 沒有錯誤，環境 5.8 GB |
| 7 | 下載官方模型 `lerobot/diffusion_pusht` | ✅ 約 1 GB，262.7M 參數 |
| 8 | 下載資料集 `lerobot/pusht` | ✅ 206 episodes、25,650 frames、10 fps、7.4 MB |
| 9 | GPU 檢查 job `check_env` | 第一次 ❌（問題 4），修正後 ✅ |

### 3.2 安裝的版本

| 套件 | 版本 |
|---|---|
| Python | 3.12.16 |
| PyTorch | 2.11.0（CUDA 13.0；驅動支援 13.2） |
| LeRobot | 0.6.2（commit `dd5313e`） |
| diffusers | 0.39.0 |
| torchcodec | 0.11.1 |
| gym-pusht | 0.1.8 |
| ffmpeg | 5.1 |

### 3.3 最終檢查（job 2259582，grn023，58 秒）

| 檢查 | 結果 |
|---|---|
| ① GPU | A800 可用，50 次 4096×4096 矩陣乘法用了 0.5 秒 |
| ② 離線讀資料集、解碼影片 | 影像 (3, 96, 96)、state 和 action 都是 2 維 |
| ③ 無螢幕執行模擬器 | reset 加上 10 步正常 |
| ④ 官方模型載入 GPU | 262.7M 參數在 `cuda:0` 上 |

這個檢查可以重複使用：`sbatch ~/projects/pusht/jobs/check_env.slurm`

### 3.4 檔案位置

```
~/envs/lerobot/                          conda env
~/projects/
├─ CLAUDE.md                             專案計畫（持續更新）
├─ lerobot/                              LeRobot 原始碼
└─ pusht/
   ├─ SLURM_NOTES.md                     Slurm 筆記
   ├─ jobs/check_env.{py,slurm}          環境檢查
   ├─ logs/                              pip 安裝、check_env 的 log
   └─ reports/                           本報告
~/gpu_hello/                             Phase 1 練習的 job（含 hold_gpu.slurm）
/scratch/general/vast/u1470293/pusht/
├─ data/lerobot/pusht/                   資料集（訓練時用這個）
└─ hf_cache/                             官方模型
```

---

## 4. 遇到的問題

### 問題 1：`sbatch: Invalid qos specification`

| | |
|---|---|
| **現象** | 第一個 job 送出就被拒絕 |
| **原因** | Lab 的 owner partition 需要搭配對應的 QOS，預設的 QOS 不能用在這個 partition |
| **解法** | 加 `#SBATCH --qos=kmarino-gpu-grn`；送出前先用 `sbatch --test-only` 驗證 |
| **學到** | Slurm 的 account、partition、QOS 三者要搭配；`--test-only` 可以免費檢查設定 |

### 問題 2：`ssh grn023` → Permission denied

| | |
|---|---|
| **現象** | 想用 SSH 或 VS Code 直接連到 GPU 節點，被拒絕 |
| **原因** | CHPC 只允許 SSH 進「你有 job 正在跑」的節點，避免有人繞過 Slurm 偷用資源 |
| **解法** | 先送一個 placeholder job（`hold_gpu.slurm`）佔住 GPU，再連到它所在的節點 |
| **學到** | 共用叢集的資源一律經過排程系統分配；互動用完要記得釋放 |

### 問題 3：計畫寫 Python 3.10，但 LeRobot 0.6.2 要求 ≥ 3.12

| | |
|---|---|
| **現象** | 安裝前讀 `pyproject.toml` 時發現（沒有真的裝錯） |
| **原因** | 教學和計畫會過時；套件的 `requires-python` 才是準的 |
| **解法** | 用 Python 3.12 建環境；同時發現 Diffusion Policy 需要 `diffusion` extra |
| **學到** | **先讀規格再安裝**；記錄 commit 編號，確保之後能重現 |

### 問題 4：Job 裡 `HF_HUB_OFFLINE=1` → `OfflineModeIsEnabled`

| | |
|---|---|
| **現象** | `check_env` 第一次執行時，讀資料集失敗（job 2259560） |
| **原因** | 離線時，LeRobot 只會去 `root`（預設是 `HF_LEROBOT_HOME/<repo_id>`）找資料集。但它之前下載的副本放在自己的 snapshot 快取裡，所以找不到，轉而上網查版本，又被離線模式擋下 |
| **解法** | 在登入節點用 `LeRobotDataset(..., root=<固定位置>)` 下載到指定資料夾，之後讀取一律指定同一個 root（CLI 用 `--dataset.root`） |
| **怎麼找到的** | 從 traceback 最下面往上讀，再讀 LeRobot 原始碼（`dataset_metadata.py:120-151`）確認它的搜尋邏輯 |
| **學到** | 離線模式能逼出程式的隱藏網路依賴；**1 分鐘的檢查 job 抓到了原本會讓 7 小時訓練失敗的 bug** |

### 順帶處理的小事

| 事項 | 處理 |
|---|---|
| ffmpeg 裝到 5.1，不是文件說的 8.x | 實際解碼影片正常，不用換 |
| batch script 裡 `conda activate` 需要先初始化 | 加 `eval "$(conda shell.bash hook)"`；**不跑 `conda init`**，避免修改 `~/.bashrc` |
| `hf download` 舊名稱是 `huggingface-cli` | 新版改用 `hf` |
| `pkg_resources is deprecated` 警告 | 來自 pygame，和結果無關，可以忽略 |

### ⚠️ 待處理（Phase 3）：官方模型的正規化警告

載入 `lerobot/diffusion_pusht` 時出現：`Unexpected key(s) when loading model: normalize_inputs...`

- **原因：** 官方模型是用舊版 LeRobot 存的。舊版把正規化數值存在模型裡，新版改放在獨立的 processor 檔案。
- **風險：** 如果 eval 時拿不到正確的正規化數值，成功率可能異常地低。
- **計畫：** Phase 3 先確認 `lerobot-eval` 是否拿到正確的正規化數值（可能需要轉換腳本），之後才採信分數。

---

## 5. 對原計畫的修正

| 原計畫 | 實際情況 |
|---|---|
| 登入 notchpeak | granite（`granite1`） |
| job 沒寫 `--qos` | 必須寫 `--qos=kmarino-gpu-grn` |
| `--gres=gpu:1` | `--gres=gpu:a800:1`（比較快排到） |
| compute node 沒網路 | 有網路，但仍在登入節點下載，job 裡維持離線 |
| Python 3.10 | Python 3.12 |
| `pip install -e ".[pusht]"` | `pip install -e ".[pusht,diffusion]"` |
| `huggingface-cli download` | `hf download`；資料集要透過 LeRobot 加上 `root` 下載 |
| 訓練輸出放在 lerobot repo 裡 | 輸出放 scratch，最終成果複製回 home |

---

## 6. 下一步：Phase 3

1. 處理正規化警告，確認官方模型的正規化數值正確。
2. 用 `lerobot-eval` 跑官方模型 10 episodes，確認流程正常。
3. 跑 500 episodes，和官方約 64% 的成功率比較，作為之後所有實驗的基準。
