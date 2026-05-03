# H012 — Method Transfer

## ① Source

- **MMoE** (Multi-gate Mixture-of-Experts, Ma et al. KDD 2018,
  arXiv:1810.10739). 다중 task / 도메인 학습에서 expert tower + per-task
  gate 로 specialization. Google production 시스템 검증.
- **PLE** (Progressive Layered Extraction, Tang et al. RecSys 2020). MMoE
  변형 — task-specific expert 와 shared expert 를 progressive 하게 분리.
  cross-task interference 감소.
- **STAR** (Star Topology Adaptive Recommender, Sheng et al. CIKM 2021).
  multi-domain CTR 의 trunk + domain-specific layer 분리.
- 카테고리 family (`multi_domain_fusion`): MMoE / PLE / STAR / MiNet.
  multi-task 또는 multi-domain learning 의 expert routing 패턴. 신규 카테고리
  first-touch.

## ② Original mechanism

**MMoE** (1단락 재서술):

MMoE 는 N 개의 expert tower (각각 small MLP) + K 개의 task 마다 별도 gate
network. gate 는 task 의 input 을 받아 N expert 에 대한 softmax routing
weight 를 출력. task t 의 출력 = `Σ_i gate_t(x)_i × expert_i(x)`. expert
는 task 간 공유, gate 만 task-specific. specialization 은 학습으로 자동
emergent — task-similarity 가 높으면 같은 expert 활용, 다르면 다른 expert.

본 H 는 **task → domain** 매핑으로 적용:
- N_experts = 4 (= 도메인 수 a/b/c/d).
- gate per NS-token (또는 per domain-fixed routing).
- expert specialization → 도메인별 vocab disjoint (Jaccard ≤ 0.10) 활용.

**PLE** (1단락 재서술):

PLE 는 MMoE 의 cross-task interference 문제 해결. expert 를 shared 와
task-specific 으로 분리, progressive 하게 layer 마다 task-specific expert
점차 분리. 본 H 는 minimum viable form 으로 single-layer MMoE 우선
시도, PLE progressive 는 sub-H 후보.

## ③ What we adopt

- **Mechanism class**: MMoE 4-expert routing on NS-token 출력. minimum
  viable form (single layer, no progressive separation).
- **신규 module `MultiDomainMoEBlock`** (~50 lines):
  ```python
  class MultiDomainMoEBlock(nn.Module):
      def __init__(self, d_model: int, num_experts: int = 4, ffn_hidden: int = 256):
          # 4 experts (domain-aligned)
          self.experts = nn.ModuleList([
              nn.Sequential(nn.Linear(d_model, ffn_hidden), nn.SiLU(),
                            nn.Linear(ffn_hidden, d_model))
              for _ in range(num_experts)
          ])
          # gate per NS-token (input → routing weights)
          self.gate = nn.Linear(d_model, num_experts)

      def forward(self, ns_tokens):  # (B, N_NS, D)
          gate_logits = self.gate(ns_tokens)  # (B, N_NS, num_experts)
          gate_weights = F.softmax(gate_logits, dim=-1)
          expert_outputs = torch.stack([e(ns_tokens) for e in self.experts], dim=-2)
          # expert_outputs: (B, N_NS, num_experts, D)
          weighted = (expert_outputs * gate_weights.unsqueeze(-1)).sum(dim=-2)
          return weighted + ns_tokens  # residual
  ```
- **통합 위치**: PCVRHyFormer 의 NS xattn 출력 직후, DCN-V2 fusion 전.
  NS-token (B, 7, D) 입력 → MultiDomainMoEBlock → enriched NS-token (B, 7, D).
  shape 보존, 모든 downstream 입력 텐서 byte-identical.
- **CLI flags**: `--use_multi_domain_moe`, `--num_experts 4`, `--moe_ffn_hidden 256`.
- **PCVRHyFormer constructor**: `use_multi_domain_moe: bool = False`,
  `num_experts: int = 4`, `moe_ffn_hidden: int = 256`.
- **§10.9 attn entropy active 적용** — gate softmax routing 가 sample-scale
  collapse 위험 (uniform 또는 1-expert dominant) → expert utilization
  entropy 측정, threshold = 0.5 × log(num_experts) = 0.5 × log(4) ≈ 0.69
  미만 시 collapse 경고.

## ④ What we modify (NOT a clone)

- **Multi-task → Multi-domain**: MMoE 원본은 N tasks 가 다른 loss 갖는 multi-
  task setting. 우리는 single-task (post-click conversion) + 4 domains.
  gate 는 task 가 아닌 NS-token 의 input 으로 결정 (input-conditioned routing).
- **Single layer**: PLE 의 progressive separation 없음 — sub-H.
- **No domain-specific output head**: MMoE 원본은 task 별 output head.
  우리는 NS-token 표현 enrichment 만, output head 는 기존 (DCN-V2 + clsfier).
- **gate input = NS-token (not raw user feature)**: NS-token 은 H010 NS
  xattn 출력 (도메인 cross-attended). gate 는 이 enriched 표현 위에서
  routing — 도메인 정보 implicit.
- **§17.2 one-mutation**: MultiDomainMoEBlock 한 클래스 추가 + flag 분기.
  다른 모든 component (NS tokenizer, NS xattn, DCN-V2 fusion, query decoder)
  byte-identical.

## ⑤ UNI-REC alignment (HARD)

- **Sequential reference**: per-domain encoder + NS xattn (H010) — 도메인별
  S-token 표현 + NS-tokens cross-attend. MMoE 가 NS-tokens 위에서 expert
  specialization 추가 → sequence-side fusion 의 새 layer.
- **Interaction reference**: DCN-V2 (H008 anchor) 그대로. enriched NS-tokens
  → DCN-V2 input → polynomial cross 처리. interaction axis 보존.
- **Bridging mechanism**: per-domain encoder → S-tokens → NS xattn (cross-
  domain alignment, attended NS) → MultiDomainMoEBlock (per-domain
  specialization, divergent NS) → DCN-V2 (interaction cross). MMoE 가
  sequence axis 의 specialization layer → §0 P1 ("seq + interaction 같은
  block gradient 공유") 강한 form (sequence specialization 가 DCN-V2 입력
  통해 interaction layer 와 gradient 공유).
- **primary_category**: `multi_domain_fusion` (신규).
- **Innovation axis**: 4 도메인 vocab disjoint (Jaccard ≤ 0.10) + length 차이
  + frac_empty 차이 데이터 사실에 기반한 explicit specialization. H010 의
  implicit selective routing 보완. 1:1 복제 아닌 부분 = single-task setting,
  NS-token level routing (not user-feature level), residual + share.

## ⑥ Sample-scale viability (Rule UB-1, §10.6)

- 추가 trainable params:
  - 4 experts × (Linear(64→256) + Linear(256→64)) = 4 × (64×256 + 256 + 256×64 + 64) = 4 × 32,896 = **131,584 params**.
  - Gate Linear(64→4) = 64×4 + 4 = **260 params**.
  - **Total: ~131,844 params** (gate + 4 experts).
- §10.6 cap **soft 위반** — anchor 의 ~198M 대비 미미하지만 H011 의 0
  추가 보다 큼.
  - 단 anchor envelope (H010) 면제 적용 가능: anchor 의 §10.6 cap 면제
    인정 후 추가 mutation 에 대해선 별도 budget. minimum viable form 으로
    cap 안에 넣으려면 ffn_hidden=128 (= 2 × d_model) 로 줄여 ~32K params 가능.
  - **결정**: ffn_hidden=128 default 채택 (~33K params 추가). card.yaml 에
    명시.
- Sample-scale (extended 30% × 10ep ≈ 51M sample steps): 33K params 학습
  충분.
- §10.9 attn entropy: gate softmax 의 routing entropy. threshold 0.5 ×
  log(4) ≈ 0.69. collapse 모니터.

## ⑦ Carry-forward rules to honor

- **§10.5 LayerNorm on x₀**: MMoE block 의 input/output 에 LayerNorm 추가
  (residual block 표준). expert FFN 자체는 SiLU + Linear (no LN inside).
- **§10.6 sample budget cap**: ffn_hidden=128 으로 ~33K params (cap 안).
- **§10.7 카테고리 rotation**: `multi_domain_fusion` 신규 first-touch.
- **§10.9 OneTrans softmax-attention entropy**: gate routing 에 적용.
  threshold 0.69 미만 시 abort.
- **§10.10 InterFormer bridge gating σ(−2)**: MMoE block 자체는 bridge
  아니지만 residual gating 추가 시 적용 (sub-H).
- **§17.2 one-mutation**: MultiDomainMoEBlock 한 클래스 + flag 분기.
- **§17.3 binary success**: Δ ≥ +0.001pt (relaxed measurable, sample-scale)
  또는 strong +0.005pt.
- **§17.4 카테고리 rotation 재진입 정당화**: 미발동 (FREE first-touch).
- **§17.5 sample-scale = code-path verification only**.
- **§17.6 cost cap**: ~3-3.5h extended, T2 안. 누적 ~24시간 (H006~H012).
  cap 압박 — fp16/batch=512 검토.
- **§17.7 falsification-first**: predictions.md 에 expert collapse / Frame
  B / C 분기 명시.
- **§17.8 cloud handoff discipline**: training_request.md + flat upload.
- **§18 inference 인프라 룰**: H010 패키지 inherit + H012 cfg keys read-back.
- **H010 F-1**: NS-only enrichment safe pattern → H012 도 NS-token 출력
  단계 stacking, anchor 입력 byte-identical.
- **H010 F-3**: H010 selective routing 보완 가설.
- **H011 F-1**: input-stage modify 위험 회피 — NS-token level (post-encoder)
  stacking.
- **H011 F-5**: cohort drift hard ceiling 가능성 — H012 결과 OOF-platform
  gap 모니터.
