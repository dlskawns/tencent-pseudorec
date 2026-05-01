# H006 — Challengers

> §10.1 — 모든 H 시작 전 반대 프레임 ≥ 2.

## Frame 1 — "LongerEncoder 의 top-K compression 이 D 도메인 long-tail 을 의미 있게 보존 못할 수도"

LongerEncoder 는 layer 마다 top-K=50 S-token 만 attention probability 기준으로 보존. D 도메인 1100-event tail 중 **유의미한 50 events** 가 정말 50 안에 들어오는지 paper 기반 보장 부족. 만약 D 의 1100 events 안에서 (a) recent 50 만 의미 있다 → 단순 truncation 과 동일 → lift 0, (b) random hash 같은 수동적 선택 → noise → REFUTED, (c) 진짜 informative top-K → 그때 lift.

**구체적 risk**:
- top-K=50 default 가 D 의 1100 tail 에 적합한지 paper-검증 부재. SIM/TWIN 같은 long-seq retrieval method 들은 candidate-aware retrieval (target attention 기반) 인데 LongerEncoder 는 self-attention probability mass 만 사용 → less targeted.
- 1100 events / 50 top-K = retention ratio 4.5%. 만약 user 의 정말 informative 패턴이 sparse 하지 않고 dense 하게 분포돼 있으면 LongerEncoder 가 cherry-pick 못 함.
- A/B/C 도메인은 seq_max_lens=64–128 이라 top-K=50 이 input 보다 작거나 비슷 → A/B/C 영향 없음 가정 검증 필요.

**mitigation**:
- §17.3 binary 임계 +0.5 pt 가 falsifier — 미달 → REFUTED + long_seq_retrieval 카테고리 retire.
- top-K tuning 별도 H 로 carry-forward (50 이 부족하면 100, 200 시도).
- D-only encoder swap 별도 H — A/B/C 성능 영향 분리.

**falsifier**: smoke val_AUC < anchor + 0.5 pt → REFUTED. domain-별 ablation (A/B/C 만 vs A/B/C/D 모두 vs D 만 longer) 은 sub-H 로 carry-forward.

## Frame 2 — "Anchor measurement 자체가 아직 invalid 상태일 수도"

original_baseline 은 새 anchor 인데 **실제 platform AUC 측정 아직 1번도 통과 안 함**. H001–H005 invalid 였던 인프라 saga 후 §18 룰 적용했지만 platform 환경에서 검증된 건 inference 가 1번 (heuristic fallback) 끝. **anchor 의 platform AUC 가 0.5 (chance) 인 상태에서 H006 mutation 의 lift 측정은 의미 없음**.

**구체적 risk**:
- original_baseline 의 platform AUC 측정 전에 H006 시작 → H006 결과가 anchor 대비 우월/열등 인지 정의 못 함.
- §18 inference 룰 적용된 패키지로도 다른 cloud-side issue (schema mismatch, ckpt sidecar 누락 등) 만날 수 있음.
- val_AUC 가 platform AUC 와 correlate 하는지 검증 필요. row-group split 잠재 leakage 이슈 carry-forward (label_time split 으로 mitigate 했지만 100% 검증 X).

**mitigation**:
- **순서 강제**: H006 launch 전에 original_baseline platform AUC 1회 측정 의무. 측정 통과 (heartbeat + OK 로그) 면 anchor 등록, H006 진행. 미통과 면 anchor 자체 디버깅 우선.
- 본 H 의 verdict.md 는 anchor val_AUC + OOF_AUC + platform AUC 3개 모두 표기 후 paired 비교.

**falsifier**: anchor measurement 가 1주 안에 통과 못하면 본 H 도 보류 + 인프라 디버깅 모드 복귀.

## Frame 3 — "Single mutation 의 lift 가 noise floor 안에 묻힘 (H001–H005 학습)"

H001–H005 결과 분석 중 사용자가 짚은 패턴: 모든 architectural mutation 의 Δ 가 noise 수준 (±0.01pt 안). Smoke envelope (train_ratio=0.05, num_epochs=1) 자체의 detection floor 한계. 본 H 도 같은 envelope → 같은 noise floor 안에 묻힐 가능성.

**구체적 risk**:
- 1-epoch training 으로 LongerEncoder 의 top-K 라우팅이 학습할 시간 부족. organizer-tuned PCVRHyFormer 가 같은 envelope 에서 빠르게 converge → mutation 의 lift detection 이 unfair.
- True effect size 가 작아도 (e.g., +0.2pt) noise floor (±0.5pt) 안에 묻혀 REFUTED 처리.

**mitigation**:
- H001–H005 archive 의 carry-forward: smoke envelope 한계 인지. 본 H REFUTED 시에도 retry on extended envelope (train_ratio=0.3, num_epochs=3) 한 번 더 시도 후보로 carry-forward.
- predictions.md 의 negative-result interpretation 에 noise vs true-fail 구별 룰 명시.
- D 도메인 ablation: D 만 LongerEncoder 적용 (A/B/C 는 transformer 그대로) 시 lift 가 더 잘 드러날 수 있음 — 코드 수정 필요한 별도 H.

**falsifier**: smoke 결과 noise floor (±0.5pt) 안 → 보조 진단 (D 도메인 attention pattern, top-K hit rate) 으로 mutation 자체 작동 여부 분리.

---

§10.1 충족: 반대 프레임 3개 (top-K compression 적합성, anchor measurement validity, smoke noise floor). 모두 falsifier + mitigation 명시.

§17.4 카테고리 rotation: 이전 5 H 중 3개가 unified_backbones, 1개 loss_calibration. **H006 = 첫 long_seq_retrieval** → rotation 추가 충족, 정당화 불필요.

§10.4 external_inspirations 의무: 본 H 는 long_seq_retrieval 카테고리 (paper 영역 = SIM/ETA/TWIN/HSTU). external_inspirations 별개 카테고리 → 본 H 미충족, H007+ 자료로 carry-forward.
