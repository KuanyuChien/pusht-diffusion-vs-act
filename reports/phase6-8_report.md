# Phase 6–8 報告：可靠評估、ACT 比較、Demo

**期間：** 2026-10-27 至 2026-10-29
**結果：**
- **Phase 6：** 用 500 場評估 DP 全部 8 個 checkpoint。200k 步的成功率 **67.0%**，重現了官方結果，也顯著高於基準 59.4%。
- **Phase 7：** 訓練並評估 ACT。200k 步的成功率 **31.6%**，DP 顯著較好（p ≈ 4 × 10⁻³¹）。
- **Phase 8：** 完成成果表、曲線圖、2×3 對照影片、英文 README，並把模型備份到 home。**推上 GitHub 待指導教授同意。**

這段期間處理了 **9 個問題或決策**，見第 5 節。

---

## 1. 最終結果

所有模型都用 500 場評估，而且是**同一組 500 個起始位置**（seed 1000）。

| 模型 | 成功率 | 95% 信賴區間 | `avg_max_reward` | 訓練時間 |
|---|---|---|---|---|
| 官方 DP（官方公布） | 65.4% | — | 0.955 | 約 7 小時（RTX 3090） |
| 官方 DP（我們的流程，基準） | 59.4% | 55.0–63.6 | 0.941 | — |
| **我的 DP（200k）** | **67.0%** | 62.8–71.0 | 0.946 | 3.5 小時（A800） |
| **我的 ACT（200k）** | **31.6%** | 27.7–35.8 | 0.645 | 1.5 小時（A800） |

**配對檢定（McNemar，同一組起始位置）：**

| 比較 | 只有前者成功 | 只有後者成功 | p 值 |
|---|---|---|---|
| 我的 DP 200k vs. 基準 | 119 | 81 | 0.009 |
| 我的 DP 200k vs. 我的 DP 100k | 129 | 73 | < 0.001 |
| 我的 DP 150k vs. 我的 DP 200k | 99 | 79 | 0.15（不顯著） |
| 我的 DP 200k vs. 我的 ACT 200k | 215 | 38 | ≈ 4 × 10⁻³¹ |

---

## 2. Phase 6：可靠評估 DP

### 2.1 做了什麼

| # | 步驟 | 結果 |
|---|---|---|
| 1 | 分析訓練 log（`analysis/training_log.py`） | 訓練健康：loss 0.536 → 0.004，沒有 NaN，設定和官方一致 |
| 2 | 寫通用的 eval script `jobs/eval_ckpt.slurm` | 可以評估任何 run 的任何 checkpoint；完成後自動把結果複製回 home |
| 3 | 500 場評估 100k 和 200k（job 2260931） | 55.8% 和 67.0% |
| 4 | 500 場評估其餘 6 個 checkpoint（job 2261184） | 完整曲線 |
| 5 | 畫「成功率 vs. 步數」，對照 50 場和 500 場（`analysis/checkpoint_curve.py`） | `results/dp_run1/success_vs_step.png` |

### 2.2 DP 的成功率 vs. 訓練步數

| 步數 | 25k | 50k | 75k | 100k | 125k | 150k | 175k | 200k |
|---|---|---|---|---|---|---|---|---|
| 500 場 | 20.2 | 45.0 | 58.6 | 55.8 | 60.8 | **71.0** | 67.6 | 67.0 |
| 50 場（訓練中） | 22 | 46 | 60 | **66** | 66 | 66 | 68 | 66 |

### 2.3 發現

1. **50 場的 eval 會誤導。**
   - 50 場顯示 100k 之後成功率就持平在約 66%，500 場卻顯示 100k 只有 55.8%，之後還持續進步。
   - 把 100k 的 500 場切成 10 組、每組 50 場，各組成功率從 48% 到 72% 都有。50 場的誤差約 ±13%。
2. **報告 200k，不報告 150k。**
   - 150k 最高（71.0%），但它是在同一組測試場次上挑出來的，數字會偏高（selection bias），而且和 200k 的差異不顯著（p = 0.15）。
   - 200k 是事先決定要報告的「訓練完的模型」。
3. **DP 減少的主要是「差一點點」的失敗。** 和基準相比，分數 ≥ 0.9 卻失敗的場次從 140 降到 109。

---

## 3. Phase 7：ACT

### 3.1 做了什麼

| # | 步驟 | 結果 |
|---|---|---|
| 1 | 在 Hugging Face 找官方的 ACT PushT 模型 | **不存在**：LeRobot 只發布了 PushT 的 DP、VQ-BeT 和 keypoint DP |
| 2 | 檢查 ACT 的預設值 | `chunk_size=100`、`n_action_steps=100`，是為 50 Hz 的 ALOHA 設計的。在 10 Hz 的 PushT 上等於閉著眼睛推 10 秒 |
| 3 | **Kuanyu 選擇「對齊時間尺度」** | `chunk_size=16`、`n_action_steps=8`（和 DP 一樣每 0.8 秒重新規劃），其他保留 ACT 預設，包括 ImageNet ResNet18 |
| 4 | 在登入節點下載 ImageNet 權重到 `$TORCH_HOME` | 45 MB，job 裡離線也能載入 |
| 5 | 寫 `jobs/train_act.slurm`，跑 smoke test（jobs 2261189、2261203） | 訓練、存檔、eval、續跑全部通過 |
| 6 | 正式訓練（job 2261213） | 1 小時 33 分完成 |
| 7 | 500 場評估全部 8 個 checkpoint（job 2261507） | 54 分 |
| 8 | 畫 DP vs. ACT 對照圖（`analysis/compare_policies.py`） | `results/dp_vs_act.png` |

### 3.2 ACT 和 DP 的比較

| | DP | ACT |
|---|---|---|
| 參數量 | 263M | 52M |
| 每步訓練時間 | 0.059 秒 | 0.026 秒 |
| GPU 記憶體 | 5 GB | 1.2 GB |
| 每個 checkpoint | 3.0 GB | 0.6 GB |
| loss 的意義 | 猜雜訊的誤差 | 動作的 L1 誤差加上 10 × KL（**和 DP 的 loss 不能互相比較**） |

| 步數 | 25k | 50k | 75k | 100k | 125k | 150k | 175k | 200k |
|---|---|---|---|---|---|---|---|---|
| DP（500 場） | 20.2 | 45.0 | 58.6 | 55.8 | 60.8 | 71.0 | 67.6 | 67.0 |
| ACT（500 場） | 18.0 | 22.2 | 20.0 | 26.2 | 26.6 | 29.0 | 30.4 | **31.6** |

### 3.3 失敗的方式不同

| | DP 200k | ACT 200k |
|---|---|---|
| 失敗場數 | 165 | 342 |
| 差一點點（分數 ≥ 0.9） | 109（66%） | 104（30%） |
| 明顯失敗（分數 < 0.8） | 53（32%） | **234（68%）** |
| 失敗場次的分數中位數 | 0.97 | **0.43** |

**解讀：** DP 的失敗大多是最後沒對準；ACT 的失敗大多是整個推的過程就出問題。這和 PushT 的多模態示範一致：同一個情況，從左邊或右邊推都對。DP 能選定一種推法；ACT 的回歸傾向把兩種平均起來。

---

## 4. Phase 8：Demo

| 產出 | 檔案 | 說明 |
|---|---|---|
| 模型備份 | `models/dp_run1_200k/`（1.0 GB）、`models/act_run1_200k/`（0.2 GB） | 從 scratch 複製回 home，`cmp` 確認逐位元組相同 |
| 對照影片 | `results/demo/dp_vs_act_grid.mp4` | 2×3，上排 DP、下排 ACT；每一欄是同一個起始位置（第 4、2、0 場）；2 倍速，15 秒 |
| 影片截圖 | `results/demo/grid_frame.png`、`first_frames.png` | 後者證明上下兩排的起始畫面相同 |
| 曲線圖 | `results/dp_vs_act.png`、`results/{dp,act}_run1/success_vs_step.png`、`training_curves.png` | |
| README | `README.md`（英文） | 結果、發現、設定、限制、重現步驟 |

**影片挑選：**

| 欄 | 場次 | DP | ACT | 意義 |
|---|---|---|---|---|
| 左 | 4 | 成功（12.8 秒） | 成功（14.6 秒） | 兩者都成功 |
| 中 | 2 | 成功（5.7 秒） | 失敗（0.10） | 典型的差異：T 一開始就在目標旁，ACT 卻把它推遠了 |
| 右 | 0 | 失敗（0.98） | 成功（27.7 秒） | 反例：DP 也會輸 |

---

## 5. 遇到的問題和決策

### 問題 9：失敗的 job 顯示「Completed」

| | |
|---|---|
| **現象** | 31 秒就失敗的 job 2260220，在 OnDemand 和 `sacct` 都顯示 COMPLETED，exit code 0:0 |
| **原因** | job script 最後一行是 `echo`。Slurm 只看 script **最後**的 exit code，而那是 `echo` 的 0，不是 `lerobot-train` 的 1 |
| **解法** | 主程式後立刻 `STATUS=$?`，結尾 `exit $STATUS`。之後所有 job script 都用這個寫法 |
| **學到** | Slurm 的狀態是 script 告訴它的；job 狀態只是參考，log 才是事實 |

### 問題 10：50 場的 eval 讓我做了錯誤的推論

| | |
|---|---|
| **現象** | 分析訓練 log 時，我根據 50 場的結果說「100k 就到頂了，只需要一半的訓練時間」 |
| **真相** | 500 場：100k 是 55.8%，200k 是 67.0%（p < 0.001），後半段的訓練有幫助 |
| **原因** | 50 場的誤差約 ±13%；訓練中的 eval 用 seed 100000，Phase 6 用 seed 1000，是不同的起始位置 |
| **處理** | 更正推論，把「50 場 vs. 500 場」畫進曲線圖 |
| **學到** | 訓練中的 eval 只用來監控，不能用來下結論或挑模型 |

### 問題 11：挑最好的 checkpoint 會讓數字偏高

| | |
|---|---|
| **情況** | 150k 是 71.0%，比 200k 的 67.0% 高 |
| **決策** | 報告 200k。用同一組場次評估 8 個 checkpoint 再挑最高的，最高分包含了運氣成分（selection bias）；而且兩者差異不顯著（p = 0.15） |
| **學到** | 要事先決定報告哪個模型；或用另一組資料來挑選（validation set 和 test set 要分開） |

### 問題 12：統計檢定的兩個 bug

| | |
|---|---|
| **12a 現象** | `ImportError: cannot import name 'comb'` |
| **原因** | 誤用了系統內建的 `python3`（3.6 版），而不是環境裡的 Python 3.12 |
| **12b 現象** | 99 比 79 卻得到 p = 1.000，還有 `divide by zero` 警告 |
| **原因** | 計數用了 numpy 的 `int64`，算 `2**178` 時溢位變成無限大 |
| **解法** | 先 `conda activate`；計數轉成 Python 內建的 `int`（沒有大小上限） |
| **學到** | 結果不合理時，先懷疑計算本身；在 CHPC 上 `python3` 不一定是你以為的那個版本 |

### 決策 13：ACT 沒有官方設定

| | |
|---|---|
| **情況** | LeRobot 沒有 ACT 的 PushT 模型；ACT 的預設 chunk 100 步在 PushT 上等於 10 秒不看畫面 |
| **選項** | ① 對齊時間尺度（chunk 16/8，保留 ImageNet 編碼器）② 也拿掉 ImageNet，完全對齊 DP ③ 完全用預設值 |
| **Kuanyu 的決定** | ①：比較的是「兩種方法」，最公平也最常見 |
| **代價** | ACT 的結果取決於這組設定，寫進 README 的「限制」 |

### 問題 14：ACT 需要 ImageNet 權重，但 job 是離線的

| | |
|---|---|
| **原因** | ACT 預設用 torchvision 的 ImageNet ResNet18，第一次使用要從網路下載 |
| **解法** | 在登入節點用 `TORCH_HOME=hf_cache/torch` 下載一次（45 MB），job 裡設定同樣的 `TORCH_HOME` |
| **學到** | 和 Phase 2 的資料集一樣：所有下載都在登入節點完成，job 裡保持離線 |

### 問題 15：ffmpeg 報錯 `Invalid file index 6`

| | |
|---|---|
| **原因** | 我的 bug：用 `len(inputs)` 當影片編號，但每支影片在 `inputs` 裡佔兩個元素（`-i`、路徑），所以編號變成 0、2、4… |
| **解法** | 改用 `len(filters)` |

### 問題 16：README 的數字是估計的，不是查證的

| | |
|---|---|
| **現象** | README 寫 ACT 的 `avg_max_reward` 是 0.62 |
| **原因** | 我從圖上估計，沒有查原始資料 |
| **解法** | 查 `eval_info.json` 後改成 0.645，同時修正 `CLAUDE.md` |
| **學到** | 報告裡的每個數字都要從原始資料查證 |

### 問題 17：推上 GitHub 前的準備（待處理）

| 檢查項目 | 結果 |
|---|---|
| 密碼、token | ✅ 沒有（唯一被標記的是錯誤訊息裡的一行原始碼，屬於誤判） |
| 大檔案 | ❌ `models/` 1.2 GB，超過 GitHub 單檔 100 MB 上限 → 要在第一次 commit 前用 `.gitignore` 排除 |
| 帳號和路徑 | ⚠️ 30 個檔案含有 `u1470293`、`kmarino`、CHPC 路徑 → 決定要保留或改寫 |
| GitHub 登入 | ❌ CHPC 上沒有 `gh`、沒有 SSH key |
| 公開前 | 先問指導教授（用的是 lab 的 GPU 額度；可能要加致謝） |

### 小問題

| 問題 | 處理 |
|---|---|
| 圖上標籤重疊（訓練曲線的「60%」、checkpoint 曲線的「71.0」） | 調整標籤位置後重畫；每張圖都會打開來檢查 |
| ACT smoke test 留下 2.4 GB 的 checkpoint | 確認後刪除 |

---

## 6. 新增的檔案

```
~/projects/pusht/
├─ README.md                              英文專案說明（GitHub 用）
├─ jobs/
│  ├─ eval_ckpt.slurm                     通用 eval：sbatch eval_ckpt.slurm <run> <場數> <步數...>
│  └─ train_act.slurm                     訓練 ACT
├─ analysis/
│  ├─ training_log.py                     解析訓練 log → CSV + 訓練曲線
│  ├─ regex_example.py                    regex 教學範例
│  ├─ checkpoint_curve.py                 成功率 vs. 步數（50 場 vs. 500 場）
│  ├─ compare_policies.py                 DP vs. ACT 對照圖
│  └─ make_video_grid.py                  2×3 對照影片
├─ models/                                200k 模型備份（不放進 git）
│  ├─ dp_run1_200k/
│  └─ act_run1_200k/
├─ results/
│  ├─ dp_run1/                            8 個 500 場結果、CSV、曲線
│  ├─ act_run1/                           同上
│  ├─ dp_vs_act.png
│  └─ demo/                               對照影片和截圖
└─ reports/phase6-8_report.md             本報告
```

job script 修正：`train_dp.slurm`、`eval_official.slurm` 改成回傳正確的 exit code。

---

## 7. 專案狀態和下一步

| Phase | 狀態 |
|---|---|
| 1–7 | ✅ |
| 8 | 🔄 推上 GitHub 待指導教授同意 |

**推上 GitHub 之前：**
1. 問指導教授：可以公開嗎？要放在哪個帳號或組織？要不要加致謝？
2. 決定帳號和路徑要保留還是改寫。
3. 建立 `.gitignore`（排除 `models/`、`logs/`）、`requirements.txt`、`LICENSE`。
4. 在 CHPC 設定 GitHub 的 SSH key，然後推上去。
5. （選擇性）把模型上傳到 Hugging Face Hub。

**長期保存：** scratch 上的 checkpoint 會被清除，但所有結果都已在 home。home 會在 CHPC 帳號停用後刪除，所以 GitHub 和 Mac 上的備份才是長期的保存方式。

**未來可以延伸的方向：**
- 每個方法用不同的 seed 多訓練幾次，量出「重新訓練」的變異。
- 調整 ACT 的設定（learning rate、`kl_weight`、chunk 大小），看差距能縮小多少。
- 換到真正的機械手臂模擬器：`gym-aloha`（ACT 原本的任務）或 LIBERO。
