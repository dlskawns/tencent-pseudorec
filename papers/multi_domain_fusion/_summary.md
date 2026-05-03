# Multi-Domain Fusion — Summary

> **Multi-task / multi-domain learning 의 expert routing 카테고리.**
> 단일 fusion block 으로 다른 도메인 (또는 task) 의 분포 차이를 강제 균일
> 처리하지 않고, expert tower + gate 로 specialization 학습. sequence-side
> fusion 강화 family.

## Takeaway (3–5 lines)

- 신규 카테고리 first-touch (2026-05-01). 직전 5 H (H006–H011) 가 sequence /
  interaction post-encoder mutation + input-stage mutation 만 시도 — explicit
  multi-domain routing 미측정.
- 주요 family: **MMoE** (gate per task, shared experts), **PLE** (progressive
  task-specific separation), **STAR** (star topology trunk + domain layer),
  **MiNet** (domain-specific base towers), **Switch Transformer** (sparse
  top-1 routing).
- **TAAC 2026 데이터 motivation**: 4 도메인 (a/b/c/d) Jaccard overlap ≤ 0.10
  (`eda/out/domain_facts.json`) → 거의 disjoint vocab. length p50 a=577 /
  d=1035 차이 큼. frac_empty a=0.5% / d=8% 차이. **explicit specialization
  자연 motivation**.
- **Sample-scale viability**: minimum viable form (4 experts, ffn_hidden=128)
  ~33K params 추가 (anchor envelope 면제 인정 후 budget 안). H012 = 첫
  적용.
- **Anti-pattern**: sample-scale 에서 expert collapse (1-2 expert dominant
  routing) 위험. §10.9 룰 적용 — gate routing entropy threshold 0.5 ×
  log(N) 미만 시 abort.

## Entries

- `mmoe_ma2018.md` — Ma et al. KDD 2018, arXiv:1810.10739. **gate per task
  + shared experts**. Multi-task learning canonical form.
- `ple_tang2020.md` — Tang et al. RecSys 2020. **shared + task-specific
  expert progressive separation**. Cross-task interference 감소.

## Carry-forward rules (post-H012 시작)

- **Rule MD-1**: minimum viable form 우선 — single layer MMoE (no progressive),
  4 experts (= 도메인 수), ffn_hidden ≤ 2 × d_model.
- **Rule MD-2**: gate softmax routing 에 §10.9 entropy 룰 적용 — collapse
  (entropy < 0.5 × log(N)) 시 abort.
- **Rule MD-3**: NS-token level routing (post NS xattn 출력 위 stacking).
  anchor 입력 byte-identical (H010 F-1 안전 패턴, H011 F-1 cohort drift
  회피).
- **Rule MD-4**: sample-scale 에서 sparse top-K routing (예: Switch
  Transformer top-1) 회피 — uniform top-1 로 collapse 위험. soft routing
  (softmax all) 우선.

## Candidate sources to fetch later

- STAR (Sheng et al. CIKM 2021) — multi-domain CTR star topology.
- MiNet (Ouyang et al. KDD 2020) — multi-domain CTR with domain-specific
  base towers + cross-domain transfer.
- Switch Transformer (Fedus et al. JMLR 2022) — sparse top-1 routing scaled.
  단 sample-scale 에선 collapse 위험.
- M3oE (Zhang et al. SIGIR 2024) — task-domain joint MoE.
