# H004 — OneTrans backbone anchor

## What we're trying to explain
PCVRHyFormer baseline (H001 anchor, val_AUC=0.8251) 의 통합 메커니즘은 query-decoder 후 RankMixerBlock 에서 token fusion 한 번. H002 가 그 위에서 한 단계 위 (per-domain seq encoder 출력 직후) 에 bridge 를 inject 했지만 query-decoder 가 이미 cross-domain mix 를 수행해 redundant → refuted. 본 H 는 통합 위치를 한 단계 더 안쪽 (per-domain encoder 자체 폐기) 으로 가져가 **모든 layer 에서 sequence × non-sequence 가 token-level 로 attention 공유** 하는 OneTrans single-stream backbone 으로 PCVRHyFormer 를 통째로 교체한다. 본 H 는 **최적화 가설이 아닌 새 anchor 생성** — 미래 H 들은 PCVRHyFormer-anchor (E_baseline_organizer, val=0.8251) 와 OneTrans-anchor (본 H 결과) 중 더 높은 쪽 위에서 mutation.

## Why now
- H002 결과: **per-domain encoder 사이의 sub-block bridge 로는 §0 두 축 통합 강화 marginal** (F-1, F-3). 통합 깊이를 한 layer 단위로 끌어내려야 함.
- §0 north star 의 backbone 후보 3 중 OneTrans (arXiv:2510.26104, Tencent WWW 2026) 만이 두 축 통합을 token granularity 까지 가져감 — InterFormer 는 arch-level bridge, PCVRHyFormer 는 block-level fusion. UNI-REC challenge 의 organizer (Tencent) 가 직접 발표한 backbone 이라 task fit 가능성 높음.
- H001 anchor 1개로 P1 게이트 들어가는 것은 backbone 다양성 부족 — `unified_backbones` 카테고리 안에서 backbone 선택 자체가 변수가 되어야 § 17.7 falsification-first 의 negative-result 해석 가능.
- 비용: H002 와 같은 T2.4 smoke 가 ~3분 소요됨을 확인. OneTrans full implementation 은 PCVRHyFormer 와 비슷한 throughput 예상 → smoke 한 번 + full 한 번 = 약 1시간 cloud time, T2 budget cap 안.

## Scope
- In:
  - PCVRHyFormer 의 `MultiSeqHyFormerBlock` + per-domain `TransformerEncoder` + per-domain `MultiSeqQueryGenerator` 를 OneTrans single-stream block 으로 교체.
  - **Mixed causal attention mask**: S→S causal, NS→S full bidirectional up to candidate, NS→NS full self.
  - **Token taxonomy**: S-tokens = 4 도메인 시퀀스 이벤트 (도메인 식별 embedding 추가), NS-tokens = RankMixerNSTokenizer chunking 그대로 재사용.
  - **Candidate token**: item embedding + bucket id, attends to all S/NS up to `timestamp`. 최종 head 입력.
  - 결함 A/B/C/D 패치 (H001 과 동일): label_time-aware split, 10% user OOF, path defaults, auto schema.json.
  - 결함 E 패치: `submission/infer.py` 가 OneTrans ckpt 로드 가능하도록 model class registry 추가.
  - `attn_entropy_per_layer` diagnostic 로깅 (§10.9 룰).
- Out:
  - **Pyramid pruning**: 본 H 미적용. paper 의 compute optimization 이라 anchor 단계엔 불필요. 별도 H (anchor 결정 후) 로 mutation.
  - Loss / focal / lr / schedule 튜닝 (§17.2 — anchor 는 zero-mutation).
  - Sequence merging strategy mutation (per-domain S-token vs single merged stream): 본 H 는 per-domain S-token + 도메인 embedding 으로 고정. merged 변형은 별도 H.
  - Pyramid pruning, longer-tail compression (D 도메인 1100 events), token routing 등 OneTrans 의 보조 메커니즘.

## UNI-REC axes
- **Sequential**: per-domain S-token causal self-attention (S→S sub-mask). SASRec/HSTU 계열 representation 보존.
- **Interaction**: NS-token bidirectional attention to all S/NS (NS→S, NS→NS sub-masks). DCN-V2/CAN 계열의 explicit cross 를 attention 으로 재구현 — token-level fusion.
- **Bridging mechanism**: 한 transformer block 안에서 S/NS 가 같은 attention layer 의 Q/K/V 를 공유 → **layer 마다 두 축 gradient 공유**. PCVRHyFormer 의 block-level fusion 대비 layer-level fusion. §0 P1 정의 ("seq + interaction 한 블록 gradient 공유") 보다 강한 조건 충족 — layer 단위.
- **primary_category**: `unified_backbones` (재진입 — H001/H002 와 같은 카테고리. challengers.md §재진입정당화 + H002 verdict.md F-3 carry-forward 인용).
- **Innovation axis**: PCVRHyFormer (block fusion) → OneTrans (layer/token fusion). 통합 깊이를 한 단계 안으로.

## Success / Failure conditions
**§17.2 anchor exemption**: 본 H 는 "한 component 클래스 교체" 가 아니라 backbone 통째 replacement. 따라서 §17.3 binary lift 임계 (Δ ≥ +0.5pt) 미적용. 대신 **anchor 자격 조건**:

- **Success (anchor 자격, 4 게이트)**:
  1. Train.py 1 epoch NaN-free 완주.
  2. val_AUC ≥ 0.7 (random 0.5 보다 충분히 높아 모델이 학습됨을 입증; 사용자 cloud 환경 organizer baseline 0.8251 대비 너무 떨어지지 않을 것 — soft target ≥ 0.80).
  3. `submission/infer.py` 가 OneTrans ckpt 로드 + `local_validate.py` 5/5 PASS.
  4. `metrics.json` 에 `{seed, git_sha, config_sha256, host, compute_tier, attn_entropy_per_layer}` 모두 채워짐.
- **Failure**: 위 4 게이트 중 하나라도 미달 → REFUTED. 결함 분류 + carry-forward.
- **Tie-breaker (anchor 비교)**: PCVRHyFormer-anchor val_AUC=0.8251 vs OneTrans-anchor val_AUC=X. 미래 H 들은 max(0.8251, X) 위에서 paired Δ 측정. **본 H 자체는 X > 0.8251 을 요구하지 않음** — 두 anchor 가 공존하는 것이 §17.7 falsification-first 정신.

## Frozen facts referenced
- H001 verdict (E_baseline_organizer): val_AUC=0.8251, organizer row-group split, train_ratio=0.05, num_epochs=1, halved seq_max_lens.
- H002 verdict F-1, F-3: PCVRHyFormer 내부에서 cross-domain bridge marginal → token-level fusion 으로 한 단계 깊이.
- `papers/unified_backbones/onetrans_tencent.md`: paper claims, mixed causal mask 정의, pyramid pruning 미적용 정당화.
- §10.9 OneTrans softmax-attention entropy abort: `attn_entropy_per_layer ≥ 0.95·log(N) ⇒ abort`. 본 H 가 그 룰의 첫 적용.
- §10.6 sample budget cap: anchor 면제 (H001 과 동일).

## Inheritance from prior H
- H001 의 결함 A/B/C/D/E 패치 인프라 (dataset.py, train.py path defaults, make_schema.py, infer.py prior fallback) 그대로 재사용. 본 H 는 **model.py 의 backbone 부분만** 교체.
- H002 의 mixed-causal motivation (§17.8 cloud handoff discipline + run.sh local-fallback 제거 + Taiji flat upload) 운영 패턴 그대로 재사용.
- H002 verdict.md F-3: cross-domain 정보 흐름이 PCVRHyFormer query-decoder 후엔 redundant → 본 H 가 그것을 layer-level 로 끌어와 의미 변경.
