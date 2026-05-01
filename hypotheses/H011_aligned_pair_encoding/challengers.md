# H011 — Challenger Frames

> CLAUDE.md §10.1, §10.7. 본 H 의 primary_category = `feature_engineering`
> (신규, FREE first-touch). 직전 2 H (H009 hybrid, H010 target_attention)
> 와 다른 axis → 재진입 정당화 불필요.

## Frame A (default — what we're proposing)

- **Claim**: aligned `<id, weight>` pair (verified shared fids `{62, 63,
  64, 65, 66, 89, 90, 91}`, 8 fids, 출처 `competition/ns_groups.json`) 의
  explicit binding 을 input embedding lookup stage 에 추가하면 baseline 의
  implicit binding 부담 제거 + downstream (NS tokenizer / NS xattn / DCN-V2)
  모두 enrichment propagate → §17.3 binary +0.5pt 또는 minimum measurable
  lift.
- **Mechanism**: aligned fid k 마다 `weighted_emb[k] = E_id(user_int[k]) *
  user_dense[k]` (element-wise broadcast, dim=D). 또는 추가 form: gating
  `weighted_emb[k] = E_id(user_int[k]) * sigmoid(user_dense[k])`. minimum
  viable form 부터 측정.
- **Why this could be wrong**:
  1. Sample-scale (1000 rows) 또는 extended (30% × 10ep) 에서 baseline 이
     이미 implicit binding 학습 충분 → explicit binding 의 marginal lift
     noise 안에 묻힘.
  2. dense feature 의 scale 분포가 ID embedding 과 곱하기 전 normalize
     필요 — 미적용 시 gradient instability.
  3. `n_k` (user_int array length) 와 `dim_k` (user_dense slice dim)
     mismatch 가능성 — P0 audit 첫 검증 항목. mismatch 시 binding semantics
     불명 → mechanism 재설계 또는 retract.

## Frame B (counter-frame — what would make A redundant)

- **Claim**: baseline 이 이미 implicit binding 학습. dense feature 가 이미
  `user_dense_proj` 로 1 NS token 으로 들어가고 NS-token level cross-attention
  (H010) 에서 user_int NS tokens 와 dense NS token 이 자유롭게 mix 됨 →
  effective binding 은 NS-token level 에서 일어남. input-stage explicit
  binding 은 redundant.
- **What evidence would prove this?**:
  - H011 결과 Δ vs anchor < +0.001pt (noise) + attention map 분석 시
    user_int NS 가 dense NS 에 이미 attention 집중하는 패턴.
  - 또는 baseline ablation (H010 minus user_dense_proj) 결과 lift 없으면
    dense 자체가 정보 거의 0.
- **Distinguishing experiment**: H011 의 weighted embedding 적용 시 NS xattn
  attention entropy 변화 측정. 만약 변화 없거나 더 sparse → A 가 맞고 input
  stage 가 효과적. 만약 더 uniform 또는 변화 미세 → B 가 맞고 NS-level 이
  이미 binding 처리 중.

## Frame C (orthogonal frame — different axis attack)

- **Claim**: 현재 더 큰 lever 는 multi-domain fusion (MMoE/PLE) — 4 도메인
  (a/b/c/d) 시퀀스가 다른 분포 (예: 도메인별 vocab 다름, 길이 다름) 인데
  현재 single fusion 으로 처리. expert routing 으로 분리하면 도메인 특이
  패턴 학습 가능.
- **Mechanism**: NS xattn (H010) 출력 → MMoE gate (4 도메인별) → DCN-V2
  fusion. 현재 H010 entropy 0.81 (highly selective routing) 이 이미 도메인
  routing 학습 중 신호 → MMoE 가 같은 문제 다른 form 일 가능성.
- **Cost vs A**: MMoE 는 expert weights 로 params 추가 (~10-30K), 통합
  위치도 NS xattn 출력 직후 = anchor 입력 변경 → 위치 충돌 위험 (H009 같은
  패턴) 회피 설계 필요. A 가 stage 가 다르고 (input) 구조적 충돌 위험 0.

## Decision

**A 선택** 이유:
1. **데이터 mandate** (§4.8) — 룰이 명시 요구. baseline 이 룰 위반 상태일
   가능성 높음 (코드 audit 결과 분리 처리). audit 자체로도 가치.
2. **Stage 가 다름 (input embedding lookup) → interference 위험 0** — H010
   anchor 의 NS xattn 출력 + DCN-V2 입력 텐서 byte-identical. H010 F-1 안전
   stacking 패턴 자연 적용.
3. **Cost 효율** — params 추가 ~0 (lookup 후 element-wise multiplication,
   parameter-free). H010 (16K) / H008 (DCN-V2 layers) 보다 훨씬 가벼움.
4. **Rotation** — `feature_engineering` 신규 카테고리 first-touch (§10.7
   FREE). target_attention 직전 2회 차단 회피.

**B/C 미루기 조건 (즉, B/C 로 돌아갈 trigger)**:
- A 결과 Δ < +0.001pt (noise) + NS xattn entropy 변화 미세 → B 가 맞음 (NS-level
  binding 충분). H012 = MMoE (Frame C 전환).
- A 결과 Δ ≥ +0.001pt + NS xattn entropy 변화 큼 (더 sparse 또는 더 uniform) →
  A 가 맞고 input-stage 가 작동. H012 = aligned pair 의 multi-form 또는
  item-side aligned pair sub-H.
- P0 audit 단계에서 `n_k != dim_k` (array length / slice dim mismatch)
  검출 시 binding semantics 재설계 또는 retract.

## (조건부) Re-entry justification

해당 없음. `feature_engineering` 은 신규 카테고리 first-touch. §10.7 / §17.4
재진입 정당화 의무 미발동.
