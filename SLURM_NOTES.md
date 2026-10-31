# Slurm 筆記（granite / kmarino）

## 核心觀念

Slurm 是 GPU 節點的「餐廳帶位員」：先說你要什麼（GPU、CPU、記憶體、時間），有位子就帶你進去，沒有就排隊。**時間到就請你離開**：job 會被強制結束。

| 詞 | 意思 | 我的值 |
|---|---|---|
| Account | 用誰的額度 | `kmarino` |
| Partition | 去哪一區的機器 | `kmarino-gpu-grn` |
| QOS | 這區的使用規則（**一定要寫**） | `kmarino-gpu-grn` |
| GRES | 特殊硬體（GPU） | `gpu:a800:1` |
| Job ID | job 編號 | `sbatch` 會印出來 |

## 兩種用法

| | `sbatch`（批次） | `salloc`（互動） |
|---|---|---|
| 關掉電腦後 | 繼續跑 | 通常跟著結束 |
| 適合 | 正式訓練、eval | debug、試跑 |
| 輸出 | log 檔 | 螢幕 |

## Job script 範本

```bash
#!/bin/bash
#SBATCH --job-name=dp_pusht
#SBATCH --account=kmarino
#SBATCH --partition=kmarino-gpu-grn
#SBATCH --qos=kmarino-gpu-grn
#SBATCH --gres=gpu:a800:1            # 1 張 A800
#SBATCH --cpus-per-task=8            # CPU（給 dataloader）
#SBATCH --mem=32G                    # RAM，不是 GPU 記憶體
#SBATCH --time=12:00:00              # 超過就被砍
#SBATCH -o logs/%x-%j.out            # %x=job 名稱，%j=job ID

module load miniforge3/25.11.0
conda activate ~/envs/lerobot
python train.py
```

`#SBATCH` 行之間不能夾一般指令，遇到第一行一般指令之後的 `#SBATCH` 都不會被讀。

## 日常指令

```bash
sbatch job.slurm                 # 送出
sbatch --test-only job.slurm     # 預測何時、在哪開始，不送出
squeue --me                      # 我的 job
scancel <jobid>                  # 取消
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # 結束後的結果
tail -f logs/<name>-<jobid>.out                         # 即時看 log
srun --jobid=<jobid> --overlap nvidia-smi               # 看跑中 job 的 GPU
salloc -A kmarino -p kmarino-gpu-grn --qos=kmarino-gpu-grn --gres=gpu:a800:1 -c 4 --mem=16G -t 1:00:00
```

## Job 狀態

| 代號 | 意思 | 該做什麼 |
|---|---|---|
| `PD` | 排隊中 | 看 `squeue` 最右欄的原因 |
| `R` | 執行中 | 看 log |
| `CG` | 收尾中 | 等幾秒 |
| `CD` | 成功 | 看結果 |
| `F` | 程式出錯 | 看 log 最後幾行 |
| `TO` | 時間到被砍 | 加長 `--time` 或從 checkpoint 接著跑 |
| `OOM` | RAM 不夠 | 加大 `--mem` |
| `CA` | 被取消 | — |

排隊原因：`Resources` / `Priority` 就等；`QOSMaxGRESPerUser` 表示你同時要的 GPU 超過上限。

## 常見的坑

| 坑 | 避免方法 |
|---|---|
| 在登入節點跑訓練 | 一律用 `sbatch` / `salloc` |
| `--time` 太短 | 先 smoke test 量時間，再多抓 30–50% |
| `--time` 太長 | 排隊比較久，不要無腦設 7 天 |
| 搞混 `--mem` 和 GPU 記憶體 | GPU 記憶體固定（A800 = 40 GB）；`CUDA out of memory` 要調小 batch size |
| 忘記關互動 session | `exit` / `scancel` / OnDemand 按 Delete |
| log 資料夾不存在 | 先 `mkdir -p logs`，否則 job 直接失敗且不留 log |
| 相對路徑 | job 在「送出時所在的資料夾」執行，所以用絕對路徑 |

## 進階

```bash
# 訓練成功後自動跑 eval
TRAIN=$(sbatch --parsable train_dp.slurm)
sbatch --dependency=afterok:$TRAIN eval_dp.slurm
```

Job 裡可用的環境變數：`$SLURM_JOB_ID`、`$SLURM_CPUS_PER_TASK`（可設給 `num_workers`）、`$CUDA_VISIBLE_DEVICES`（Slurm 自動設好）。

## 送出前檢查

1. account、partition、qos、gres 都寫了嗎？
2. log 資料夾存在嗎？
3. 用 `sbatch --test-only` 確認會被接受。
4. 先跑 smoke test。
