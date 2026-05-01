# H006 — Longer encoder for long-tail sequence domains (esp. D)

## What we're trying to explain

original_baseline 의 smoke envelope 은 `seq_max_lens = seq_a:64, seq_b:64, seq_c:128, seq_d:128`. CLAUDE.md §3 의 데이터 팩트:
- domain_d 의 시퀀스 길이 distribution 이 1100 events tail (조직자 baseline data 분포 기준).
- 즉 **D 도메인 user 의 평균/긴-꼬리 행동 시퀀스가 128 token 으로 잘려서 약 88% 의 행동 데이터가 모델에 전달 안 됨.**

PCVRHyFormer baseline 의 `seq_encoder_type='transformer'` 는 quadratic attention (`O(L²)`) → seq_max_lens 를 늘리면 GPU/wall 비용 폭증. 그래서 우리 envelope 에선 단순 truncation 으로 짤림.

LongerEncoder (organizer-supplied, `model.py:616 LongerEncoder`) 는 **top-K attention compression** 으로 동일 hidden size 에서 긴 시퀀스를 처리. 입력 길이는 자유롭지만 layer 마다 attention probability mass 가 큰 top-K (default 50) S-token 만 보존 → compute O(L log K) 정도.

본 H 는 `--seq_encoder_type longer` 한 줄로 4 도메인 모두의 시퀀스 인코더를 교체. D 도메인의 long-tail 데이터 손실이 최대 lift 영역. A/B/C 도 default 64–128 envelope 안에선 LongerEncoder 의 top-K compression 이 transformer 와 거의 동일 동작 (K=50 ≥ L=64 일 때 truncation 안 함) → A/B/C 영향 minimal.

## Why now

- **Anchor reset (2026-04-28)**: H001–H005 인프라 bug 영향으로 invalid → original_baseline 이 새 anchor. 이 anchor 위 첫 mutation 은 **데이터 속성 mismatch 직접 해결**이 가장 확률 높음 (loss/regularization 같은 marginal mutation 보다).
- **§17.4 카테고리 rotation**: H001 (unified_backbones) → H002 (unified_backbones) → H004 (unified_backbones) → H005 (loss_calibration). H006 = `long_seq_retrieval` 으로 추가 rotation 충족.
- **§17.2 one-mutation 깔끔**: `--seq_encoder_type longer` CLI flag 변경 only. 코드 수정 0 (LongerEncoder 클래스가 이미 organizer model.py 에 있음).
- **비용 cheap**: smoke 환경 wall 약간 증가 예상 (~3분 → ~5분, top-K compression overhead). T2 budget 안.
- **§10.4 external_inspirations 의무 미충족**: 본 H 의 카테고리는 `long_seq_retrieval` (SIM/ETA/TWIN 계열). 별도 H 로 carry-forward.

## Scope
- In:
  - Encoder swap: `transformer` → `longer` (organizer LongerEncoder, top-K=50 default).
  - 4 도메인 모두 적용 (CLI flag global). D 가 최대 lift 영역, A/B/C 거의 영향 없음.
  - 그 외 모든 config: original_baseline 과 byte-identical envelope (organizer leak-fix split, train_ratio=0.05, num_epochs=1, halved seq_max_lens, NS=5+2=7, num_queries=2, d_model=64, num_hyformer_blocks=2, seed=42, BCE).
  - §18 inference 인프라 룰 (batch_size=1024 생성자, universal handler, 진단 로그) — original_baseline 에서 그대로 inherit.
- Out:
  - `seq_top_k` 튜닝 (default 50 그대로, §17.2 one-mutation).
  - seq_max_lens 확장 (별도 H — top-K + 더 긴 input 같이 mutation 은 §17.2 위배).
  - Per-domain encoder type (D 만 longer, 나머지 transformer) — 별도 H, 코드 수정 필요.
  - Backbone 변경.

## UNI-REC axes
- **Sequential axis**: TransformerEncoder → LongerEncoder. 시퀀스 입력 길이 capacity 증가, attention compute 동일 (top-K compression).
- **Interaction axis**: 변경 없음 — RankMixerNSTokenizer + MultiSeqHyFormerBlock 그대로.
- **Bridging mechanism**: 변경 없음 — block fusion 그대로.
- **primary_category**: `long_seq_retrieval` (§17.4 rotation).
- **Innovation axis**: 데이터 속성 (D 도메인 long-tail) 직접 활용. 모델 capacity 가 아니라 "정보 보존" 측면 mutation.

## Success / Failure conditions
**§17.3 binary lift 임계 적용**:

- **Success**: Δ vs original_baseline-anchor val_AUC ≥ **+0.5 pt**. **+ 4 부수 게이트**:
  1. Train 1 epoch NaN-free 완주.
  2. `submission/local_validate.py` 5/5 PASS (G1–G6).
  3. `metrics.json` 에 `{seed, git_sha, config_sha256, host, best_val_AUC, best_oof_AUC, split_meta}` 모두 채워짐.
  4. **Platform AUC 측정 가능**: inference 시 `[infer] OK: torch path produced N predictions` 로그 + batch heartbeat 둘 다 보임 (§18.3 룰). 만약 silent fallback 신호 (heartbeat 없음 / 모든 prediction ≈ 0.124) 면 P4 fail.
- **Failure**:
  - Δ < +0.5 pt → **REFUTED**. long_seq_retrieval 카테고리 일시 archive (top-K 튜닝, per-domain encoder, seq_max_lens 확장 후보 들 sub-tree 보류).
  - 부수 게이트 1–4 중 1개라도 fail → 코드/계약 위반 또는 인프라 회귀 신호.

## Frozen facts referenced
- `experiments/original_baseline/` (anchor): val_AUC TBD (인프라 fix 후 첫 measurement 대기), envelope = leak-fix smoke.
- CLAUDE.md §3: domain_d 시퀀스 길이 1100 events tail (1년 전 snapshot, 본 H smoke 결과로 직접 검증 — D vs A/B/C 사이 lift 비대칭이면 가설 confirm).
- `competition/model.py:616 LongerEncoder` (organizer-supplied, 코드 변경 없음).
- §10.6 sample budget cap: anchor envelope 동일, 면제.
- §18 infra rules: original_baseline 패키지 그대로 적용.

## Inheritance from prior H (carry-forward)

Active prior H 없음 (anchor reset 후 H006 = 첫 mutation). 단 archive 에서 carry-forward:

- **H002 verdict F-1**: cross-domain mix 는 token/layer-level 이어야 의미 있음 → 본 H 는 cross-domain 안 건드림 (§17.2). 다음 mutation 후보로 retain.
- **H004 verdict F-1 (P3 PASS)**: OneTrans softmax routing sample-scale 작동 검증 → full-data 도착 후 OneTrans + LongerEncoder 조합 후보로 carry-forward.
- **H005 verdict F-1**: BCE 가 12% imbalance 영역에서 충분히 작동 → 본 H 도 BCE 유지 (focal 시도 retire).
- **인프라 saga (§18 신설)**: original_baseline 패키지가 §18 룰 모두 적용 → 본 H 가 그 패키지 기반 → §18 자동 inherit.
