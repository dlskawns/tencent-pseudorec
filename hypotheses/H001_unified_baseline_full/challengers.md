# H001 — Challenger Frames

## Frame A (default — what we're proposing)
Baseline anchor 생성: organizer PCVRHyFormer를 결함 A/B/C/D 패치만 적용한 채로 demo_1000.parquet에서 1회 학습, OOF AUC 측정, §13 submission round-trip 검증.
- **Why this could be wrong**: 1000 rows로 ~250k–500k 파라미터 모델은 §10.6 sample budget을 100x 초과. 학습 신호 = noise. AUC가 0.5 근처 나와도 그것이 모델 부적합인지 데이터 부족인지 분리 불가.

## Frame B (counter — anchor를 demo_1000에서 만들지 말고 T0 prior로 두자)
- **Claim**: demo_1000에서 학습한 결과는 신호가 아니다. anchor = class prior (0.124 flat) 로 두고, 모든 component-mutation 가설은 T2/T3 cloud full-data에서만 측정해야 한다.
- **Evidence for**: 1000 rows × 124 positives는 AUC 분산이 매우 큼. paired Δ ≥ +0.5 pt 임계치를 sample에서 만족시켜도 full-data에서 재현 안 될 가능성 농후 (이전 캠페인의 정확한 실패 모드).
- **Distinguishing experiment**: full-data가 도착하면 demo-trained anchor와 cloud-trained anchor의 OOF AUC를 비교. 만약 둘이 0.05 pt 이상 차이나면 demo-anchor를 폐기.

## Frame C (orthogonal — baseline + LightGBM tabular control 동시 학습)
- **Claim**: baseline anchor만으로는 "unified block의 가치"를 측정 불가. user_int + item_int + user_dense를 mean-pool로 평탄화한 LightGBM 표 모델을 control로 같이 학습해야, "sequence/interaction을 unified block으로 연결한 lift"의 lower bound가 잡힌다.
- **Cost vs A**: +5분 (LGBM은 1000행에서 즉시 학습). 결과: 두 control (T0 prior, LGBM tabular) + 두 unified block (E000) → 4-cell 비교 매트릭스.
- **Risk**: scope creep — H001을 "anchor 1개 측정"에서 "anchor 3개 측정"으로 확장. 다음 H로 미루는 게 깔끔.

## Decision
**Frame A 채택**. Frame B의 reservation을 honor: H001 verdict.md에 `claim_scope: "demo-only, generalizability TBD"` 명시. Full-data 도착 시 즉시 E000.full로 재실행, 두 결과의 paired Δ를 reports에 기록.

Frame C의 LGBM control은 H002로 분리 — H001 통과 후 즉시 착수.

## Re-entry justification
H001은 첫 가설이라 카테고리 rotation 면제 (CLAUDE.md §10.7). primary_category = `unified_backbones` (baseline 자체가 그 카테고리의 reference 코드).
