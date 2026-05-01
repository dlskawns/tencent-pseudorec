# H002 — Challenger Frames

## Frame A (default — 우리가 제안)
InterFormer R1: 4 도메인 seq encoder 사이에 low-rank (rank=4) gated bridges 추가. 매 `MultiSeqHyFormerBlock` 안에서 per-domain encoder 출력을 mask-aware pool → 다른 도메인으로 broadcast update. Gate σ(-2)≈0.12 init (near-off, 학습으로 grow).
- Why this could be wrong: 5% data (94k rows) 는 12,312 새 param 학습에 부족. gate가 0 부근에서 학습 안 되고 그대로 머물 가능성 → effectively no-op.

## Frame B (counter — bridge가 redundant)
- Claim: HyFormer 의 RankMixerBlock 이 이미 query token + NS token 을 mix 하므로 cross-domain 정보가 그 단계에서 흐름. Bridge 추가는 redundant overhead.
- 반증 evidence: paper 의 controlled ablation (bridge ON vs OFF) 에서 +0.3–0.8 pt 의 일관된 lift. paper data 가 우리 data와 다를 가능성 있지만, bridge mechanism 자체는 architecture-level이라 어느정도 transfer 됨.
- Distinguishing experiment: bridge ON vs OFF 의 paired Δ. 미미하면 (<+0.3pt) Frame B 채택, bridge 방향 retire.

## Frame C (orthogonal — OneTrans single-stream 이 더 효과적)
- Claim: 도메인간 정보 공유는 InterFormer bridge 보다 OneTrans mixed causal mask (모든 도메인 토큰을 한 attention 안에서 처리) 가 더 자연스러움.
- Cost vs A: OneTrans는 변경 ~150줄 (block 재구현). InterFormer bridge는 ~80줄.
- Risk vs A: OneTrans는 attention 비용 4x (4 도메인 토큰 통합). 우리 production data 에서 메모리 + 속도 추가 부담.
- 미루는 조건: H002 (InterFormer) 통과 → H003 (OneTrans mixed causal) 자연 진행. H002 refuted → OneTrans 가 single-stream 으로 더 큰 변경이라 별 의미 없을 수 있음.

## Decision
**Frame A 채택**. Frame B/C 는 H002 결과로 자동 검증.

## Re-entry justification (§10.7, §17.4)
직전 H001 = `unified_backbones`. 본 H002 도 `unified_backbones`. 같은 카테고리 2회 연속.

**정당화**:
1. **사용자 explicit 요청**: 3 papers (OneTrans/InterFormer/HyFormer) 기반 가설을 명시 요청. papers/unified_backbones/ 가 그 home.
2. **H001 verdict carry-forward**: anchor 만 측정 (zero mutation). 본격 mutation 은 H002–H006 에서 시작 — 모두 같은 카테고리 (3 papers) 로 자연 진행.
3. **§10.4 external_inspirations 의무는 다음 차례 (H006 또는 별도 시점)**: 본 H들이 unified_backbones 에서 끝난 뒤에 inject.
4. **다른 카테고리 rotation 보존**: H006 후 long_seq_retrieval / multi_domain_fusion / loss_calibration 등으로 진행 예정.

§17.4 의 강제 차단 사유는 "재진입 정당화 부재" — 위 4 항목이 정당화 충족.
