# HXXX — Method Transfer (P1+ 필수)

> CLAUDE.md §11 hook이 P1 이후 본 파일 존재를 PreToolUse에서 강제.

## ① Source
(어느 paper/카테고리에서 가져왔나. `papers/{category}/{slug}.md` 링크.)

## ② Original mechanism
(원 논문의 메커니즘 1단락 — copy-paste 금지, 우리 언어로 재서술.)

## ③ What we adopt
(우리가 그대로 가져오는 component. 1줄 bullet.)

## ④ What we modify (NOT a clone)
(원 메커니즘에서 의도적으로 바꾼 것. 우리 데이터/제약/§10 규칙에 맞추기 위한 차이점. 이 섹션이 비어 있으면 "1:1 재현"이고 §10 anti-pattern.)

## ⑤ UNI-REC alignment (HARD)
- **Sequential reference**: (어느 seq backbone 표준 — SASRec/DIN/HSTU 등)
- **Interaction reference**: (어느 cross 표준 — DCN-V2/CAN/FwFM 등)
- **Bridging mechanism**: (둘이 같은 블록에서 gradient를 어떻게 공유하나)
- **primary_category**: `papers/{category}/` 단일값
- **Innovation axis**: 우리가 1:1 복제 아닌 부분의 핵심 한 줄

## ⑥ Sample-scale viability (Rule UB-1, §10.6)
- 예상 trainable params: ~XXX (≤ 200 hard / ≤ 2146 soft)
- 만약 violate한다면, full-data archival인 이유와 sample-scale 대체안

## ⑦ Carry-forward rules to honor
- §10.5 LayerNorm on x0
- §10.10 InterFormer gate init σ(−2)
- §10.9 OneTrans softmax-attention entropy ≥ 0.95·log(N) abort
- (해당 시) 직전 H verdict.md F-N
