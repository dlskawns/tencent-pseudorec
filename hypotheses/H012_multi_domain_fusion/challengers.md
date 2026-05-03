# H012 — Challenger Frames

> CLAUDE.md §10.1, §10.7. primary_category = `multi_domain_fusion` (신규
> first-touch — FREE). 직전 2 H (H010 target_attention, H011
> feature_engineering) 와 다른 카테고리 → 재진입 정당화 불필요.

## Frame A (default — what we're proposing)

- **Claim**: 4 도메인 (a/b/c/d) 의 vocab disjoint (Jaccard ≤ 0.10) + length
  distribution 차이 + frac_empty 차이 (a=0.5%, d=8%) 가 큰 상태에서 단일
  fusion block 이 sub-optimal. MMoE/PLE expert routing (4 experts, per-domain
  specialization) 으로 ≥ +0.001pt platform AUC.
- **Mechanism**: NS xattn 출력 (B, N_NS=7, D) → expert routing layer (4
  experts, gate per NS-token 또는 per domain-aware aggregation) → enriched
  NS tokens (B, 7, D) → DCN-V2 fusion (그대로). NS-side enrichment, anchor
  입력 byte-identical (H010 F-1 안전 stacking).
- **Why this could be wrong**:
  1. H010 의 selective routing (entropy 0.81 = ~2 tokens / 384 attend) 가
     이미 도메인 routing 을 implicit 학습 중 → explicit MMoE 가 redundant.
  2. sample-scale (1000 rows × 30%) 에서 4 expert × per-domain params 는
     §10.6 budget 위험 — 학습 데이터 부족 시 expert collapse 또는 uniform
     gate.
  3. cohort drift 이 hard ceiling 이면 (H011 F-5) routing 변경으로도 platform
     일반화 한계 못 깸.

## Frame B (counter-frame — what would make A redundant)

- **Claim**: H010 NS xattn 이 이미 implicit 도메인 routing 학습 중. selective
  routing (entropy 0.81) 자체가 도메인 specific tokens 에 집중하는 신호.
  explicit MMoE 가 같은 일을 다른 form 으로 푸는 redundant work.
- **What evidence would prove this?**:
  - H012 결과 Δ < +0.001pt (noise) + expert utilization 측정 시 1-2 expert
    dominant (collapse) 또는 H010 attention pattern 과 high correlation.
  - 또는 H010 attention map 직접 측정 (sub-H) 시 도메인별 명확한 specialization
    이미 학습됨.
- **Distinguishing experiment**: H012 의 expert utilization (per-expert
  weighted activation 분포) 측정. 균등 routing (entropy ≈ log(4)=1.39)
  이면 explicit routing 효과 없음 (B 가 맞음). 도메인-specific routing
  (entropy ≪ 1.39, 1-2 expert dominant) 이면 H010 이 이미 dominant 학습
  중 (역설적으로 B 가 맞음). 균등 사이 (entropy ~ 0.7-1.2) + Δ > 0 이면 A.

## Frame C (orthogonal frame — different axis attack)

- **Claim**: 더 큰 lever 는 long-seq retrieval (P2 phase) — current truncate
  64-128 이 §3.5 의 p90 ≫ 100 정보 대량 손실. SIM/TWIN/HSTU 의 retrieval
  + compression 으로 더 많은 history 활용.
- **Mechanism**: target-aware retrieval (top-K relevant history) + truncate
  대신 trunk + retrieval block.
- **Cost vs A**: P2 phase 진입 자체가 P1 통과 후. 현재 P1 진입 검증 중.
  H012 = multi_domain_fusion 이 P1 sequence-side 마지막 mutation 후보.

## Decision

**A 선택** 이유:
1. **카테고리 신규** (§10.7 FREE) — direct rotation benefit.
2. **데이터 mandate** — Jaccard 0.7~10% (§3.5 갱신) 가 expert specialization
   의 정량 근거. 다른 H 가 이 data property 를 활용하는 mechanism 없음.
3. **H010 F-3 carry-forward** — selective routing 이 dominant signal 학습 중.
   explicit specialization 으로 보완 가설 검증 가치.
4. **위치 안전** — NS xattn 출력 위 stacking, anchor 입력 byte-identical
   (H010 F-1 안전 패턴, H011 F-1 cohort drift 회피).
5. **cost-effective** — minimum viable MMoE 4 experts 는 ≤ 1K params 추가
   (§10.6 budget 안).

**B/C 미루기 조건 (즉, B/C 로 돌아갈 trigger)**:
- A 결과 Δ < +0.001pt + expert utilization collapse (1-2 expert dominant) →
  Frame B 채택. H013 = NS xattn sub-H 또는 cohort 처리 (H011 F-5).
- A 결과 Δ ≥ +0.005pt (strong) → Frame A 강한 confirm. H013 = expert
  routing variants (PLE progressive, gate hyperparams).
- P2 phase 진입 가능 시 (P1 통과 + 본데이터) → Frame C (long-seq retrieval).

## (조건부) Re-entry justification

해당 없음. `multi_domain_fusion` 신규 카테고리 first-touch. §10.7 / §17.4
재진입 정당화 의무 미발동.
