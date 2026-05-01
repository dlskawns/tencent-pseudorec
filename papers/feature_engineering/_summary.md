# Feature Engineering — Summary

> **Input-stage feature representation 카테고리.** sparse embedding lookup
> 과 dense numerical feature 를 결합하거나, ID 와 weight 를 binding 하거나,
> 분포 변환 (discretization, normalization) 을 적용하는 input-stage mutation
> family. interaction layer (DCN-V2, FmFM 등) 와 sequence layer (DIN, HSTU
> 등) 의 **upstream** 단계.

## Takeaway (3–5 lines)

- 신규 카테고리 first-touch (2026-04-30). 직전 5 H (H006–H010) 가 sequence/
  interaction axis post-encoder mutation 만 시도 — input stage 는 미측정.
- 주요 family: DLRM (sparse-dense input fusion), FwFM (field-weighted FM),
  DIN (attention-weighted history), Wide & Deep (cross-tower input),
  AutoDis (dense feature discretization).
- §0 UNI-REC north star 의 **upstream lever**: input embedding 강화는 모든
  downstream block (sequence + interaction + fusion) 에 gradient 공유 propagate.
- TAAC 2026 데이터 mandate (CLAUDE.md §3 / §4.8): aligned `<id, weight>`
  pair `{61–66, 89–91}` 가 데이터 자체 mandate — 미시행 시 leakage-audit
  미통과.
- Sample-scale viability: 대부분 mechanism 은 parameter-free 또는 ≤ 1K params.
  §10.6 budget 부담 없음.

## Entries

- `dlrm_naumov2019.md` — Naumov et al. 2019, arXiv:1906.00091. Deep Learning
  Recommendation Model. **sparse embedding × dense feature input fusion** 의
  canonical form. interaction layer 입력 단계에서 element-wise / dot-product
  결합.
- `fwfm_pan2018.md` — Pan et al. WWW 2018, arXiv:1806.03514. Field-weighted
  Factorization Machine. **field-pair-specific scalar weight** 를 cross 항에
  곱함. weight semantics 의 정수.

## Carry-forward rules (post-H011 시작)

- **Rule FE-1**: input-stage mutation 은 anchor 의 downstream 텐서 byte-identical
  유지 권장 (interference 위험 0). H010 F-1 carry-forward.
- **Rule FE-2**: dense feature scale handling 의무 — raw [0, ∞) weight 가
  embedding 과 곱해질 때 LayerNorm 또는 sigmoid 게이팅 sub-form 비교 필수.
- **Rule FE-3**: aligned pair 가설은 **데이터 사실** 검증 후만 (eda/out/aligned_fids.json
  산출). 가정 미검증 시 매핑 잘못 → negative control 역할.
- **Rule FE-4**: parameter-free form 우선 (DLRM raw multiply). 학습 가능
  parameter 추가 (FwFM scalar weights) 는 sub-H.

## Candidate sources to fetch later

- DIN paper (Zhou et al. KDD 2018, arXiv:1706.06978) — attention-weighted
  history. soft alignment 의 canonical form.
- AutoDis (Guo et al. CIKM 2021) — dense feature automatic discretization.
  scale handling sub-H 후보.
- DLRM-v2 또는 후속 — Meta 의 production-scale dense-sparse fusion 진화.
- TencentRec / FuxiCTR family — Tencent 내부 production CTR 시스템 architecture.
