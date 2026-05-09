# H042 — transfer.md (UNI-REC alignment)

## §⑤ UNI-REC alignment
- **Sequential axis**: H019 의 TWIN GSU+ESU + per-domain encoder 그대로 carry.
- **Interaction axis**: NS→S xattn + DCN-V2 stack 그대로 carry.
- **Loss-objective axis (NEW)**: BCE main + MSE aux (H038) + KLD prior
  matching (H042 신규) — 셋 다 같은 backbone output 에서 gradient 공유.
  output prob distribution 이 sequential + interaction 통합 backbone 의
  최종 output 이므로 KLD 가 통합 mechanism 에 distribution-level constraint
  주입.

## Why this mechanism for THIS data (CLAUDE.md §0.5)
- Signal: §3.4 class prior 12.4% — 14 H 모두 prior 정보 explicit constraint
  안 줌. F-G ceiling 의 일부 원인일 가능성 (model 이 calibration 못 함).
- Diagnose: H038 aux MSE (간접 supervision) +0.0012 OOF lift = output-level
  signal axis 작동 confirm. KLD = MSE 보다 *직접* prediction distribution
  attack.
- Mechanism design: per-sample Bernoulli KLD to fixed prior. 사용자 §0.5
  의 근무표 shift 분포 KLD 균등화 패턴 의 binary 적용.
- Validation chain: T0 sanity 4 test PASS (math correctness + grad flow +
  λ=0 byte-identity).

## What's NOT a clone
- 외부 paper 의 calibration head (Platt scaling / Temperature scaling) 와
  다름 — post-hoc inference time 이 아닌 *training time* loss term.
- Label smoothing 과 비슷한 효과지만 form 다름 — label smoothing 은 target
  label 을 smooth, KLD 는 *prediction* 을 prior 쪽으로 regularize. 우리는
  fixed prior 에 매칭, label 자체 smooth 안 함.
- 외부 paper transplant 0. Pereyra et al. 2017 confidence penalty 의
  Bernoulli 적용은 reference 일 뿐, mechanism 의 정당성은 §3.4 class prior
  data signal 에서 옴.

## Carry-forward
- §17.2 single mutation: KLD term 추가 (loss-axis), model graph
  byte-identical to H038.
- §17.4 rotation: output_distribution_supervision axis sub-H (H038 first
  → H042 sub-H), RE_ENTRY_JUSTIFIED — 같은 axis 의 더 깊은 form.
- §10.6 sample budget: trainable params +0.
- §0.5 data-signal-driven: §3.4 class prior 직접 attack.
