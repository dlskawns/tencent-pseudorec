# H004 — Challengers

> §10.1 + §17.4: 모든 H 시작 전 반대 프레임 ≥ 2. 본 H 는 `unified_backbones` 카테고리 3회 연속 (H001/H002/H004) → 재진입 정당화 의무.

## §재진입정당화 (§17.4)

H001 (PCVRHyFormer anchor) → H002 (PCVRHyFormer + bridge mutation, refuted) → H004 (OneTrans backbone replacement) 모두 `unified_backbones` 카테고리.

**왜 다른 카테고리로 안 가는가**:
- §0 north star 가 명시: 모든 가설은 두 축 동시 (seq + int) 충족. lgbm/longer_encoder/switch_loss 같은 측정도구는 §0 미충족 → 메인 큐 부적격.
- H002 verdict F-1: "PCVRHyFormer query-decoder 가 이미 cross-domain mix" → backbone 자체를 더 깊은 통합으로 가져가는 것이 다음 자연스러운 step. 카테고리 rotation 보다 통합 깊이 incremental refinement 가 더 정당.
- §0 의 backbone 후보 3 중 OneTrans 만이 token-level fusion 까지 가는 backbone. PCVRHyFormer 와 다른 통합 깊이 → 같은 카테고리지만 다른 hypothesis class.
- H001 verdict F-N (carry-forward): "결함 A/B/C/D 패치 인프라가 production-ready" → 본 H 가 그 인프라를 reuse 해 backbone 만 교체하면 marginal cost 로 anchor 다양성 확보 가능.
- H002 verdict F-3: "cross-domain 정보 흐름의 새 channel 가설 보류" → OneTrans mixed-causal mask 가 그 channel 의 다른 implementation. H002 의 single-axis bridge 가 실패한 motivation 을 layer-level 로 옮겨 재시도.

**카테고리 rotation 예외 정당화 강도**: §17.4 가 요구하는 "직전 verdict.md F-N 직접 인용" 충족 (H002 F-1, F-3 위 인용). H005/H006 은 H004 결과 도착 후 카테고리 rotation 의무 재가동.

## Frame 1 — "OneTrans paper claim 은 100M+ user scale 한정"

OneTrans 논문 (arXiv:2510.26104) 의 +0.4–1.2 pt AUC lift 주장은 Tencent internal 100M user CTR data 기준. 본 대회는 **train_ratio=0.05 × demo_1000 = 47k rows** smoke + **full data 도착 시점 미정**. paper claim 의 scale-dependent 효과가 사라질 가능성 농후.

**구체적 risk**:
- Mixed causal mask 의 advantage 는 long-context (L_S ≥ 100, L_NS ≥ 50) 에서 발현. 우리 smoke 모드의 seq_max_lens=64–128 은 paper 의 1024+ 와 비교 불가.
- NS-token bidirectional attention 은 NS-token 수 ≥ 30 정도여야 의미 — RankMixer 의 user_ns_tokens=5 + item_ns_tokens=2 = 7개로는 attention pattern 단순화.
- §10.9 softmax entropy abort: 토큰 수가 적을 때 attention prob 이 uniform collapse 위험 정확히 본 H 의 약점.

**mitigation**:
- Anchor 자격 조건 (val_AUC ≥ 0.7) 만 충족하면 OK — paper 의 lift claim 은 본 H 에서 검증 안 함. 미래 H 들이 OneTrans-anchor 위에서 mutation 시 paper claim 의 scale 의존성을 직접 측정.
- `attn_entropy_per_layer` 로깅 + abort threshold (§10.9 룰). entropy 너무 높으면 verdict.md 에 plus-flag.

**falsifier**: smoke 결과 attn entropy 모든 layer 에서 ≥ 0.95·log(N) → abort, OneTrans backbone 전체 retire (cloud full-data 도 시도 중단).

## Frame 2 — "PCVRHyFormer 가 organizer 제공 + 검증된 인프라; OneTrans 는 from-scratch reimpl 리스크"

PCVRHyFormer 는 `competition/model.py` 1714줄 organizer-supplied. 사용자가 cloud 환경에서 한 번 학습 성공 (E_baseline_organizer val=0.8251) → infrastructure 검증 완료. OneTrans 는 paper 만 있고 reference code 없음 → 우리가 from-scratch 로 짜야 함.

**구체적 risk**:
- 1500+ 줄 model.py 신규 작성 = 한 턴에 정확히 짜기 어려움 + 디버깅 비용.
- mixed causal mask 의 정의 (S→S causal, NS→S bidirectional up to candidate, NS→NS full) 를 attention mask matrix 로 정확히 옮기는 게 paper 본문에서 ambiguous 한 부분 있음 — "up to the candidate position" 의 candidate position 정의가 시퀀스 안에 있는지 밖에 있는지.
- Dual optimizer (Adagrad sparse + AdamW dense) 의 sparse param 분류가 PCVRHyFormer 와 OneTrans 에서 다를 수 있음 — sparse re-init 패턴 (H002 로그의 "Re-initialized 96 high-cardinality Embeddings") 의 OneTrans 호환성 미검증.

**mitigation**:
- 본 H 의 model.py 를 PCVRHyFormer baseline 의 부분 교체 로 빌드 — `MultiSeqHyFormerBlock` 만 OneTrans 블록으로 교체, 그 외 (`PCVRParquetDataset`, `RankMixerNSTokenizer`, classifier head, dual optimizer wiring) 그대로 재사용. 신규 코드 ~400–600줄 예상.
- mixed causal mask 의 ambiguous 점은 본 H 의 `transfer.md §④ what we modify` 에 명시 + paper 정의 중 가장 보수적인 (NS→S 가 candidate timestamp 미만 S 토큰만) 해석 채택.
- Smoke 가 passed 만 검증하고 paper claim 의 full lift 는 검증 안 함 — 코드 정확성 sanity check 만.

**falsifier**: smoke 학습 NaN/OOM 또는 attention output dimension mismatch 로 abort → mixed causal mask 또는 token taxonomy 정의 오류 신호. 디버깅 후 1회 더 시도, 그래도 fail 시 OneTrans 설계 자체를 polished H 로 분리하고 다른 backbone (예: HSTU) 후보로 카테고리 내 새 anchor 시도.

## Frame 3 — "anchor 2개 운영 = experiment overhead 배가"

PCVRHyFormer-anchor 와 OneTrans-anchor 가 공존하면 미래 H 마다 두 anchor 위에서 mutation 시도하거나 (배가 비용) 또는 한 anchor 만 골라야 함 (선택 편향). §17.1 "Baseline-first, end-to-end on cloud" 정신은 anchor 1개를 강하게 검증하라는 쪽.

**구체적 risk**:
- H005, H006, H007 등이 모두 OneTrans-anchor 위에서만 mutation → PCVRHyFormer-anchor 가 사실상 죽은 자산이 됨.
- 또는 모든 mutation 을 양쪽 anchor 에서 다 돌리면 cloud cost cap (§17.6 per-campaign $100) 초과 위험.
- 두 anchor 의 상대 우열을 verdict.md 에서 어떻게 비교할지 룰 없음 — 후속 mutation 의 paired Δ 측정 baseline 이 분기.

**mitigation**:
- 본 H 결과 도착 후 즉시 anchor 선택 룰 명시: max(0.8251, X) 위에서 만 mutation, 떨어지는 anchor 는 archive. 두 값이 noise 수준 이하 (|0.8251−X| < 0.005) 이면 OneTrans-anchor 우선 (token-level fusion 이 §0 north star 더 강한 충족).
- §17.6 budget cap 모니터링: 본 H smoke + full = ≤ $5 예상. campaign 전체 $100 cap 안에서 anchor 선택 후 mutation 큐 짤 여유 충분.
- experiment overhead 가 진짜 문제면 anchor 선택 후 PCVRHyFormer 자체 retire — H001 verdict 에 "anchor superseded by H004" carry-forward.

**falsifier**: 본 H smoke val_AUC 가 0.7 미만 → anchor 자격 미달 → PCVRHyFormer-anchor 단독 유지, OneTrans 방향 retire. anchor overhead 문제 발생 안 함.

---

§10.1 충족: 반대 프레임 3개 (paper claim scale dependency, reimpl risk, anchor overhead). 모두 `falsifier:` + `mitigation:` 명시. 본 H 의 success path 가 위 3 risk 모두에서 살아남으려면 smoke 단계에서 (a) NaN-free, (b) val_AUC ≥ 0.7, (c) attn entropy 적정, (d) submission round-trip 통과 모두 만족.
