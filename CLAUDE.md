# TAAC 2026 — UNI-REC Project Harness (tencent-cc2)

> 매 세션 자동 로드. **모델링 의견 금지** (그건 `hypotheses/`). 여기는 바뀌지 않는 규칙과 1차 데이터 팩트만.
> 상세 운영 문서는 `notes/refs/` 로 분리됨. **§19 Reference Routing** 참조.

---

## §0 UNI-REC North Star — 모든 의사결정의 상위 기준

**대회**: *Tencent UNI-REC Challenge* (TAAC 2026, algo.qq.com).
운영: HF `TAAC2026` org (Junwei Pan, Yuliang Yang, Chao Zhou).
**왜 "UNI-REC"인가** — *unified recommendation*: 두 계열을 단일 backbone에서 통합하는 것이 이 대회의 혁신축이자 승부처.

**두 축** — 모든 가설은 두 축을 **동시에** 다뤄야:

1. **Sequential axis** — 행동 시퀀스의 시간/순서/장기 관심.
   - 표준 참조: SASRec, BERT4Rec, DIN/DIEN, SIM/TWIN, HSTU.
   - 데이터 근거: 4개 도메인 시퀀스 (`domain_a/b/c/d_seq_*`), 도메인별 컬럼 수 a=9, b=14, c=12, d=10.
2. **Feature-interaction axis** — 스칼라/멀티밸류/dense feature의 explicit cross.
   - 표준 참조: DCN-V2, CAN, FwFM, FmFM, AutoDis.
   - 데이터 근거: user_int 46 (35 scalar + 11 list), item_int 14 (13 scalar + 1 list), user_dense 10 (모두 list<float>).

### Backbone 가이드 — 3 papers (papers/unified_backbones/)

| 논문 | 핵심 메커니즘 | 우리에게 주는 lever |
|---|---|---|
| **OneTrans** (Tencent, WWW 2026, arXiv:2510.26104) | S-token + NS-token single-stream transformer + **mixed causal mask** + pyramid pruning | NS-token 동등분할 chunking, mixed causal로 `label_time` leakage 차단 |
| **InterFormer** (Meta, CIKM 2025, arXiv:2411.09852) | 3-arch (Interaction × Sequence × Cross) **bidirectional bridges**, gate init σ(−2)≈0.12 | Bridge gating 패턴 — H015 DAMTB 정당화 |
| **PCVRHyFormer** (organizer baseline, `tencent-cc/competition/`) | Per-domain seq → query decoder → joint fusion → **RankMixer NS tokenizer** + dual optimizer (Adagrad sparse + AdamW dense) | RankMixer parameter-free chunking이 sample-scale에서 직접 transferable |

**하네스 규칙**:
- P1+ 모든 H의 `transfer.md`에 §⑤ UNI-REC alignment 블록 필수: Sequential × Interaction × backbone 통합 메커니즘.
- P1 진입 조건: 시퀀스 인코더와 explicit interaction cross가 **같은 블록에서 gradient 공유** (concat-late는 P0까지만).
- `hypotheses/INDEX.md`의 "UNI-REC axes" 컬럼: `seq` / `int` / `seq+int` / `--` 중 하나.

**Anti-pattern**: 단일축 우승 주장 / 3 논문 1:1 재현 / concat만으로 "통합" 주장.

---

## §1 Project Boundary — 위반 금지

**허용 경로**: `/Users/david/Desktop/assignments/DataEngine/tencent-cc2/**`, `/tmp/**`.

**금지**:
- 프로젝트 바깥 파일의 삭제/수정/이름변경/이동.
- 프로젝트 바깥으로의 `cp`/`mv` 금지 (읽기 전용 참조는 OK, 내용 요약만).
- `git`은 이 프로젝트 워킹트리 내부에서만. `..`의 `.git`이나 사이드 레포 건드리지 말 것.
- 환경 전역 변경(`brew install`, `pip install --user`, dotfile 수정) 금지.
- 위 금지를 우회하는 심볼릭 링크 생성 금지.

**Sibling repo `tencent-cc/`는 read-only 참조 자료**. `papers/`, `skills/`, `competition/` 에서 정보를 읽을 수 있지만, 그 쪽에 쓰기는 금지. 필요한 산출물은 본 프로젝트로 카피.

**근거**: 이 Mac에는 프로젝트 밖에 다른 사용자 작업물이 많다. 실수 한 번이 큰 손실.

---

## §2 Environment & Compute Reality

- OS: macOS 13.5.1, Apple **M1 Pro** (14-core, Metal 3), 32 GB RAM.
- Python: `.venv-arm64/` (arm64 native, MPS) 우선. 새 venv는 `uv venv --python 3.11`.
- 실행은 항상 풀패스: `.venv-arm64/bin/python ...`.
- 디스크 여유 상시 모니터: `du -sh data/ experiments/`.

**§2.1 — 학습은 로컬에서 안 한다 (CRITICAL)**: M1 Pro는 sample-scale (≤ 1000 rows) sanity check 전용. 모든 full-data 학습은 외부.
- 로컬 full-data 학습 차단 — `card.yaml`의 `compute_tier:` 명시 필수.
- 로컬 sample run은 §10.6 budget 준수.
- Cloud run은 deterministic seed + git SHA pin + config sha256 없으면 카드 미기록.
- 4-tier (T0/T1/T2/T3) 상세: `notes/remote_training_options.md`.

---

## §3 Frozen Data Facts (1차 출처: HF README + 직접 EDA만)

**1차 출처**: `https://huggingface.co/datasets/TAAC2026/data_sample_1000` (README, 2026-04-10 업데이트).
모델링 판단 금지. 데이터 레벨 팩트만.

- 파일: `demo_1000.parquet`, 40 MB, **flat column layout** (이전 nested struct 폐기, 2026-04-10).
- 1,000 rows × 120 cols. License: CC-BY-NC-4.0.
- **6 categories of columns**:
  | Category | Count | Type |
  |---|---|---|
  | ID & Label | 5 | int64 / int32 |
  | User Int Features | 46 | int64 또는 list<int64> (35 scalar + 11 array) |
  | User Dense Features | 10 | list<float> |
  | Item Int Features | 14 | int64 (13) + list<int64> (1) |
  | Domain Sequence Features | 45 | list<int64> (a=9, b=14, c=12, d=10) |
- ID & Label: `user_id`, `item_id`, `label_type` (int32), `label_time` (int64), `timestamp` (int64). 5개 모두 nullable=False.
- **Aligned `<id, weight>` 규약** (verified 2026-04-30, 출처: `competition/ns_groups.json` `_note_shared_fids` + `_note_user_dense`):
  - `user_int_feats_{fid}` 와 `user_dense_feats_{fid}` 가 같은 fid 공유 시 align 되어 동일 entity/signal jointly 기술.
  - **Verified shared (aligned) fids**: `{62, 63, 64, 65, 66, 89, 90, 91}` — 8 fids. ID 측 (user_int) 과 weight 측 (user_dense) 둘 다 존재.
  - **Dense-only fids** (user_int 측 매칭 없음, aligned 효과 미적용): `{61, 87}`.
  - **user_dense_feats flat layout**: 10 fids (`{61, 62, 63, 64, 65, 66, 87, 89, 90, 91}`) 의 multi-dim list 가 concat 되어 per-row **total_dim=918**. per-fid offset/dim 매핑은 `competition/dataset.py` 의 `_user_dense_plan` 참조.
  - aligned fids 의 user_ns_groups 위치: U2 (`[48, 49, 89, 90, 91]`) + U7 (`[3, 4, 55–59, 62–66]`) — 비-aligned fids 와 섞여 있음. group-aware binding 필요.
- Scalar user_int fids: `{1, 3, 4, 48–59, 82, 86, 92–109}` (35).
- Array user_int fids: `{15, 60, 62–66, 80, 89–91}` (11).
- Scalar item_int fids: `{5–10, 12–13, 16, 81, 83–85}` (13).
- Array item_int fids: `{11}` (1).
- Domain seq fids: `domain_a_seq_{38–46}` (9), `domain_b_seq_{67–79, 88}` (14), `domain_c_seq_{27–37, 47}` (12), `domain_d_seq_{17–26}` (10).

**검증되기 전엔 인용 금지**: 위 외 수치 (라벨 분포, 시퀀스 길이 통계, vocab overlap 등) 는 EDA로 직접 측정 후 `eda/out/*.json`에 기록하고 cite. tencent-cc/CLAUDE.md §3의 1년 전 스냅샷 수치는 재검증 후 사용.

---

## §4 Ground Rules

1. **수치 인용은 1차 출처 + 파일경로 + 필드명 명시 후에만.** 미검증은 "TBD".
2. **모델링 선택은 "팩트"가 아닌 "가설"**. 단정 금지. 전부 `hypotheses/HXXX/` + `status` 부여.
3. **Train/val split은 항상 `label_time` 기준** (`timestamp` 기준 split 금지 — exposure→feedback 누수).
4. **OOF 홀드아웃**: `user_id` 10%는 모든 phase 공통 고정 (학습 절대 금지). seed 42, `np.random.default_rng(42)`.
5. **재현성**: `metrics.json`에 `{seed, git_sha, config_sha256, python, venv, host, compute_tier}` 필수.
6. **Config hash 중복 탐지**: 새 실험 생성 시 `experiments/INDEX.md`의 `config_sha256` 중복 체크.
7. **파일 emoji 금지** (사용자 명시 요청 시 예외).
8. **Aligned `<id, weight>`는 항상 한 쌍으로 이동**. 한쪽만 쓰는 코드는 leakage-audit 미통과.
9. **데이터 사실 인용 chain**: §3 의 "후보" / "TBD" / "미검증" 라벨이 붙은 사실은 fact 인용 금지. `competition/ns_groups.json` (NS 그룹/aligned fids/dense schema) 또는 `eda/out/*.json` 에서 검증된 값으로 §3 본문 갱신 후 인용. 다른 docs (`hypotheses/HXXX/*.md`, `card.yaml` 등) 가 §3 의 미검증 사실을 fact 처럼 인용 시 차단.

---

## §5 Workflow Cycle (요약 — 상세는 `notes/refs/hypothesis_workflow.md`)

```
1. redefine-problem → problem.md
2. challenger       → challengers.md (≥2 반대 프레임)
3. literature-scout → papers/{cat}/*.md
4. method-transfer  → transfer.md (P1+ hook 강제)
5. hypothesize      → predictions.md (반증 가능)
6. experiment-card  → card.yaml + run
7. verify-claim + journal-update → verdict.md, progress.txt
```

step 1, 2를 skip하면 local-minima 박힘.

---

## §6 Phase Gates

| Phase | 진입 조건 | 산출 | 통과 기준 |
|---|---|---|---|
| **P0** | 하네스 셋업 + 1회 platform round-trip success | scalar-only LR/LGBM baseline + class-prior `infer.py` smoke submission | `predictions.json`이 검증 통과 + 채점 시스템 valid score 1회 |
| **P1** | P0 통과 | per-domain encoder + aligned-pair pooled + **UNI-REC unified block** (seq+interaction이 한 블록에서 gradient 공유) | OOF에서 P0 대비 ≥ 1.5 pt AUC + ablation matrix + 통합 블록이 seq-only/int-only 두 ablation 능가 |
| **P2** | P1 통과 + 본데이터 수령 | long-seq retrieval (TWIN/SIM/ETA) + HSTU trunk OR OneTrans pyramid pruning | P1 대비 ≥ 0.5 pt AUC 또는 p90 seq-len bin GAUC 개선 |
| **P3** | 멀티모달/SID 데이터 공개 시 | Semantic-ID generative head | HR@10 / NDCG@10 기준, 조직자 baseline 능가 |

Phase Gate 통과는 `reports/phase_reviews/P{n}_verdict.md`에 기록. 통과 전 다음 phase 자료 생성 금지.

**Submission 추가 조건**: P0 통과 = ≥1회 sanity 제출 Success. P1 진입 = `infer.py`가 P0 베이스라인으로 정상 동작. (상세 G1–G6: `notes/refs/submission_contract.md`)

---

## §7 Paths

```
tencent-cc2/
├── CLAUDE.md                    # 이 파일 (매 턴 자동 로드)
├── data/                        # 불변. demo_1000.parquet 등
├── eda/                         # 데이터 팩트만. out/*.json은 1차 소스
├── papers/                      # 카테고리별 장기기억 (refs/papers_routing.md)
├── hypotheses/                  # 모든 사고·실험
├── experiments/INDEX.md         # 전 실험 레지스트리 (config_sha256)
├── notes/
│   ├── remote_training_options.md   # T0–T3 4-tier 상세
│   └── refs/                        # CLAUDE.md ref 분리 문서 (§19)
├── reports/                     # phase_reviews/, EDA 리포트
├── skills/                      # 프로젝트 로컬 스킬
├── submission/                  # infer.py / local_validate.py / prepare.sh / README.md
├── progress.txt                 # 이터레이션 저널 (append only)
└── .claude/settings.local.json  # 권한 + hook
```

---

## §10 Anti-Bias Rules

1. 어떤 phase 시작 전에도 `challenger.md`로 **반대 프레임 ≥ 2** 먼저 나열.
2. EDA 리포트 아키텍처 제안은 "첫 가설"이며 `hypotheses/`로 이전. 정답처럼 따르지 말 것.
3. 3회 연속 같은 계열 실험에서 개선 없으면 **강제 challenger 재호출**.
4. `external_inspirations/`에서 최소 1개 아이디어를 P1 이후 모든 phase에 주입.
5. **LayerNorm on x_0 MANDATORY for any DCN-V2-style cross stack**. 새 cross 설계 시 LayerNorm 누락은 차단 사유.
6. **Sample-scale param budget ≤ N/10**: 1000-row sample에서 trainable params hard ≤ 200, soft ≤ 2146. Full-data 도착 전 deep embedding tables는 archival.
7. **Category rotation mandate**: 같은 `papers/{category}/`에서 2회 연속 실험 금지 (재진입 정당화 없으면). transfer.md에 `primary_category:` 필수.
8. **Continuous-scouting 의무**: 학습 안 돌더라도 `papers/{cat}/*.md` 신규 + `hypotheses/HXXX/{problem,transfer,predictions}.md` 스캐폴드는 계속 생성.
9. **OneTrans softmax-attention 트랩**: sample-scale에서 attention prob uniform collapse 위험. `attn_entropy_per_layer ≥ 0.95·log(N)` ⇒ abort, hard routing or n_experts ≤ 2.
10. **InterFormer bridge gating init = sigmoid(−2) ≈ 0.12** — 새 bridge/scalar gate는 near-off로 시작. 즉시 active 시작은 차단.

---

## §16 Known Anti-patterns

- **사이드 레포 침범**: tencent-cc/, ../albatrips/ 등 sibling 디렉토리에 쓰기 시도 금지 (§1).
- **demo_1000을 본데이터처럼 다루기**: 1000 rows는 ablation 신호용. claims는 본데이터 도착 후 재검증.
- **3 papers 1:1 재현**: transfer.md에 "what's not a clone"을 적시.
- **TAAC2025 metric 재사용 가정**: 2025는 HR@10/NDCG@10 + 변환 weight. 2026은 `notes/refs/submission_contract.md`에 따라 conversion probability + AUC-style 가능성. 공식 metric 명시 전엔 가정 금지.
- **algo.qq.com 미확인 항목 인용**: JS 렌더링이라 직접 fetch 불가. 사용자 paste 또는 후속 공식 문서 확보 시까지 submission_contract.md만 신뢰.

---

## §19 Reference Routing — 작업별 분기

> 본 CLAUDE.md는 매 턴 자동 로드. 아래 ref 문서는 **트리거 작업 시에만** 읽음 (lazy load).

| 작업 트리거 | 읽을 ref 문서 | 원래 §  |
|---|---|---|
| `submission/infer.py` / `predictions.json` / 제출 패키징 | `notes/refs/submission_contract.md` | §13 + §14 |
| `infer.py` / `dataset.py` / `make_schema.py` 작성·수정 | `notes/refs/inference_lessons.md` | §18 |
| 새 H 또는 새 E 작성 / `card.yaml` / `run.sh` 작성 | `notes/refs/hypothesis_workflow.md` | §5 상세 + §9 + §17 |
| literature-scout 호출 / `papers/` 신규 항목 | `notes/refs/papers_routing.md` | §8 |
| 스킬 호출 / 등록 / 수정 | `notes/refs/skills_index.md` | §12 |
| `.claude/settings.local.json` 수정 / hook 디버깅 | `notes/refs/hooks.md` | §11 |
| Cloud T2/T3 실험 실행 직전 / tier 결정 | `notes/remote_training_options.md` + `notes/refs/hypothesis_workflow.md` §17.8 | §15 + §17.8 |
| aligned fid / NS group / user_dense schema / per-fid offset/dim 작업 | `competition/ns_groups.json` (`_note_*` keys 우선) + `competition/dataset.py` (`_user_dense_plan`) | §3 검증 chain |
| 새 데이터 사실 측정 / `eda/out/*.json` 산출 | `competition/ns_groups.json` 우선 검토 후 누락 시 직접 EDA → `eda/out/{slug}.json` 기록 | §3 + §4.1 + §4.9 |

**규칙**:
- ref 문서 안 읽고 해당 작업 진입 시 차단 사유 (특히 `infer.py` 수정 → `inference_lessons.md` 미독은 H001~H005 같은 무한 fail 재발).
- ref 문서 자체 수정은 본 CLAUDE.md의 §19 routing 표도 같이 갱신.
- 새 ref 문서 추가 시 본 §19 표에 한 줄 추가 + 트리거 명확히.
