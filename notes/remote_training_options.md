# Remote Training Options — TAAC 2026 UNI-REC

> 운영 전제 (CLAUDE.md §2.1): 로컬 M1 Pro 32GB는 sample-scale (≤ 1000 rows) sanity 전용. **모든 full-data 학습은 외부.** 본 문서는 그 "외부"를 4 tier로 분해해 각 tier의 사용처/비용/제약/구체 명령을 정리한다. `experiments/{id}/card.yaml`의 `compute_tier:` 필드는 본 문서의 코드와 1:1 대응한다.

---

## Tier 0 — No-train baselines (cost = 0)

학습 자체를 안 하는 옵션. P0 round-trip 검증 + 데드라인 직전의 안전 패치 용도.

### T0.1 — Class prior
- `infer.py`가 `MODEL_OUTPUT_PATH/prior.json` (또는 학습 통계 미존재 시 0.5) 을 읽어 모든 user_id에 동일 값 반환.
- **장점**: §13 contract 100% 준수, 결정적, 실패 risk 0.
- **용도**: P0 첫 제출, 채점 파이프라인 sanity check.
- **단점**: 점수 floor.

### T0.2 — Heuristic ranker (deterministic feature engineering)
- 학습 없는 점수: `score = w1·timestamp_recency + w2·domain_overlap_count + w3·has_aligned_pair`. weight는 EDA 통계로 직접 산출 (no gradient).
- 후보: 시퀀스 마지막 이벤트와 `item_id` 매칭 카운트, `<id, weight>` aligned pair sum, `label_time - timestamp` gap의 inverse.
- **장점**: 데이터 의미 정렬, T0.1보다 점수 상한 높음.
- **단점**: 가설 1개 = 코드 1개. 휴리스틱 업데이트 = 코드 변경.

### T0.3 — kNN / item-popularity baseline
- Test의 `item_id`별로 train에서 같은 item을 본 user들의 평균 conversion ratio.
- 학습 시간 0, 룩업 테이블만 패키지에 포함.
- **단점**: cold-start item 0%. fallback prior 필수.

**공통 게이트**: T0 제출은 모두 `submission/local_validate.py` G1–G6 통과 + `submission/README.md`에 "Tier=T0" 명시.

---

## Tier 1 — Local sample-scale (cost = 0)

M1 Pro에서 `demo_1000.parquet` 1000 rows로 §10.6 budget 안의 모델 학습.

### T1.1 — `.venv-arm64` LightGBM/XGBoost
- 시퀀스/dense feature는 mean-pool 또는 last-K aggregation으로 flatten → 표 형식.
- 학습: `lightgbm.LGBMClassifier` on M1 CPU. 1000 rows 학습 < 30s.
- **용도**: tabular feature 1차 가설 검증, ablation matrix 생성, 시퀀스 vs non-sequence 정보 분리 측정.
- **장점**: 결정적, MPS 의존 없음, scikit 친화적.
- **단점**: UNI-REC 통합 axis 검증 불가 (sequential axis는 pool 후 사라짐).

### T1.2 — `.venv-arm64` PyTorch MPS — minimal UNI-REC block
- §10.6 budget (≤ 2146 trainable params) 안에서:
  - per-domain seq encoder = mean-pool + 1-layer linear (no attention)
  - DCN-V2 cross L=1 on x0 (LayerNorm 필수)
  - bridge gating sigmoid(−2) init (InterFormer R1)
- 1000 rows × 8 epoch ≈ 5분 wall.
- **용도**: P0→P1 전환 전 sanity, 코드 경로 검증 (broadcast shape, autograd flow).
- **단점**: 점수 신뢰도 낮음 — sample-noise dominant. 추세 신호만.

### T1.3 — Cross-validated parameter sweep on sample
- T1.1 또는 T1.2를 5-fold × 5-seed = 25-run으로 반복.
- 보고: paired delta (control vs treatment) on fold-seed 매칭.
- **장점**: 1000-row noise floor 측정 + treatment 효과 separation.

**공통 게이트**: T1 실험은 `metrics.json`에 `n_rows: 1000` + `claim_scope: "sample-only, cannot generalize"` 명시. 본 데이터 결과 주장 금지.

---

## Tier 2 — Cloud spot/free (cost ≈ $0–$10/run)

짧은 ablation, single-seed 학습. iteration 사이클 빠름.

### T2.1 — Colab Free (T4 16GB)
- 12hr 세션 한계, 비활성 시 끊김.
- HF dataset 직접 마운트: `from datasets import load_dataset; load_dataset("TAAC2026/data_sample_1000")`.
- 학습 노트북은 `notebooks/train_*.ipynb`에 두고 `notes/remote_training_options.md`에 노트북 ID 기록.
- **단점**: 결정적 재현 어려움 (커널 차이), git pin 수동.
- **용도**: 빠른 hyperparameter scan, OneTrans/InterFormer 1-layer prototype.

### T2.2 — Colab Pro/Pro+ (~$10–50/mo)
- A100 40GB, 24hr 세션.
- T2.1 시나리오에서 학습 시간 5-10x 단축, **본데이터 도착 시 P1 sweep용**.
- 결제는 사용자 카드, 사전 합의 필수.

### T2.3 — Kaggle Notebooks (P100 16GB, 30hr/wk)
- 데이터 업로드 후 코드 컨테이너에서 학습.
- **장점**: 완전 결정적 reproducible 환경. 노트북 = config = code.
- **단점**: 외부 인터넷 옵션 켜야 HF 접근 가능 → 보안/정책 확인.
- **용도**: 1주 단위 단일 캠페인 실험에 적합.

### T2.4 — Modal.com (A10 ~$1.10/hr, A100 ~$3/hr)
- Python SDK로 함수 정의 → 클라우드 실행. cold start 5-10s.
- 학습 스크립트:
  ```python
  import modal
  app = modal.App("taac")
  image = modal.Image.debian_slim().pip_install("torch", "lightgbm", "pyarrow")
  @app.function(gpu="A10G", image=image, timeout=3600)
  def train(config_sha): ...
  ```
- **장점**: pay-per-second, 코드 git pin 자동, persistent volume 옵션 (`modal.Volume`).
- **용도**: T2 중 가장 reproducible. 새 가설 1개 = Modal job 1개.

### T2.5 — RunPod Spot (A40 ~$0.40/hr, A100 ~$1/hr)
- SSH 접속, persistent volume.
- spot price는 변동, 강제 종료 가능 → checkpoint마다 volume에 sync.

**공통 게이트**: T2 실험은 `metrics.json`에 `compute_tier: T2.X` + `cost_estimate_usd:` 필드 추가. seed × 3 repeat 의무 (variance 측정 가능해야 P1 통과).

---

## Tier 3 — Cloud reproducible at scale (cost ≈ $30–$300/campaign)

P1 통과 후 본데이터 학습, 최종 제출용.

### T3.1 — Lambda Labs On-demand (A100 80GB ~$1.99/hr, H100 ~$3/hr)
- 시간 청구, no preemption.
- HF dataset 미러를 자체 S3에 두고 컨테이너 시작 시 sync.
- **재현성**: docker image hash + git SHA + config sha256 → metrics.json.
- **용도**: 최종 OneTrans/InterFormer/HyFormer 통합 백본 학습.

### T3.2 — RunPod Secure Cloud (on-demand)
- T2.5의 비-spot 버전. preemption 없음.
- 가격 ~10-20% 상승하지만 24시간+ 캠페인에 안전.

### T3.3 — Tencent Cloud GPU
- 대회와 같은 vendor, 일부 region에서 무료 크레딧 발급.
- **장점**: 데이터 locality, 잠재적 platform 친화 (확인 안 됨).
- **단점**: 영문 문서 부족, 가입 절차.

### T3.4 — University HPC / Group cluster
- 사용자에게 접근 가능한 학내/그룹 자원 있는지 확인 필요. 무료 + 결정적이면 T3 1순위.

**공통 게이트**: T3 실험은 seed × 3 + 본데이터 split (label_time 기준) + OOF 10% 보존 + ablation matrix 의무. `reports/phase_reviews/P{n}_verdict.md` 작성 트리거.

---

## Decision Matrix — 어떤 tier를 언제 쓰나

| 단계 | 적합 tier | 이유 |
|---|---|---|
| P0 첫 제출 (round-trip 검증) | T0.1 | 모델 없이 contract만 검증 |
| P0 baseline 점수 floor 올리기 | T1.1 (LightGBM tabular) | M1에서 무료 + 결정적 |
| P0→P1 전환 sanity (UNI-REC 통합 블록 코드 검증) | T1.2 (MPS, ≤2146 params) | 코드 경로만 확인 |
| P1 ablation matrix (treatment vs control) | T2.4 (Modal) | per-job 비용 명확, seed×3 자동화 |
| P1 새 가설 1-layer prototype | T2.1 또는 T2.3 | 비용 0 |
| 본데이터 도착 후 P1 통과 시도 | T3.1 (Lambda A100) | 결정적, 24h+ run 가능 |
| 최종 제출 직전 best ckpt 학습 | T3.1 또는 T3.2 | 보증된 재현성 |

---

## 학습 ↔ 제출 분리 워크플로우 (CRITICAL)

§13에 따라 **infer.py만 채점 컨테이너에서 실행됨**. 즉:

1. **학습은 외부 (T1–T3 어딘가)** → ckpt + 메타파일을 로컬로 download.
2. ckpt를 `submission/ckpt/` 또는 `MODEL_OUTPUT_PATH`로 패키징.
3. `submission/local_validate.py`로 G1–G6 통과 확인.
4. `submission/prepare.sh`로 `submission/` 디렉토리를 zip → 플랫폼 업로드.

**절대 안 됨**:
- 학습 코드를 `infer.py`에 포함 (G6 위반 가능, 시간 초과).
- 외부 다운로드를 `infer.py`에서 시도 (§13.1, §14.1 G6 위반).
- `requirements.txt`에 명시되지 않은 dependency를 `infer.py`가 import.

**제출 패키지 표준 레이아웃**:
```
submission/
├── infer.py
├── ckpt/
│   ├── model.pt           # state_dict
│   ├── schema.json        # FeatureSchema dump
│   ├── train_config.json  # hparams
│   └── prior.json         # T0 fallback (class prior)
├── local_validate.py
└── README.md              # 제출 로그
```

---

## 비용 가드레일 (사전 합의 필수)

- **T2.4 / T2.5**: 1-job ≤ $5, 1-day ≤ $20 한도. 초과 시 사용자 confirm.
- **T3.1**: 1-run ≤ 6h × $2/hr = $12, 1-campaign ≤ $100. 초과 시 사용자 confirm.
- 모든 비용은 `experiments/{id}/metrics.json`의 `cost_estimate_usd:` 필드에 기록 + `experiments/INDEX.md`에 cumulative sum.

---

## 직접 검증되지 않은 가정 (ASSUMPTION FLAG)

- 채점 컨테이너의 GPU/CPU/메모리 한계는 가이드 본문에 미명시 — `infer.py`는 CPU-only fallback path를 항상 보유.
- 채점 컨테이너의 wall-clock budget도 미명시 — `infer.py`는 batch size를 작게 (128) 시작, 필요 시 늘리도록.
- HF dataset 라이선스 (CC-BY-NC-4.0)는 학습/평가 목적엔 OK, 상업화 금지. 대회 종료 후 사용 시 재확인.
- 본데이터 (full set)의 스키마가 demo_1000.parquet와 동일하다는 보장 없음 — 조직자 공지 시 §3 업데이트.
