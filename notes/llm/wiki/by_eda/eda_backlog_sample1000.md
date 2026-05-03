---
title: "EDA Backlog for Sample-1000"
type: "overview"
status: "draft"
created_at: "2026-05-02"
updated_at: "2026-05-02"
sources:
  - path: "CLAUDE.md"
    kind: "file"
  - path: "eda/out/domain_facts.json"
    kind: "file"
  - path: "eda/out/dense_dim_breakdown.json"
    kind: "file"
  - path: "eda/out/aligned_audit.json"
    kind: "file"
confidence: "medium"
promotion_state: "not-promoted"
---

# EDA Backlog for Sample-1000

이 문서는 sample-scale(1000 rows)에서 **오프라인 신호 신뢰성 확보**를 위한 우선순위 EDA 백로그다.
모델링 결론이 아니라, 실험 진입 전 점검 항목을 정리한다.

## P0 — Immediate (must pass before comparing models)

### 1) Temporal integrity / leakage audit
- Output: `eda/out/time_integrity.json`
- Compute:
  - `% rows where any feature_ts > label_time`
  - `label_time - max(domain_ts)` distribution (min, p1, p50, p95, p99, max)
  - split boundary violations under `label_time` split rule
- Why: split/feature window가 깨지면 오프라인 비교 자체가 무효가 됨.
- Done criteria:
  - leakage 의심 row 비율 보고 완료
  - split violation 0 또는 허용 사유 문서화

### 2) Label-event gap closure
- Output: `eda/out/label_event_gap.json`
- Compute:
  - `label_minus_event_seconds` summary: count/min/p1/p50/p95/max
  - `% negative`, `% zero`, by-`label_type` breakdown
- Why: 시간축 정합성 및 유효 lookback/cutoff 규칙 정의에 필요.
- Done criteria:
  - 음수/0초 비율과 분포를 명시
  - 향후 split/feature cutoff 정책에 반영 포인트 기록

### 3) Offline objective mismatch risk
- Output: `reports/eda/metric_rank_agreement.md`
- Compute:
  - 동일 예측에 대해 AUC / PR-AUC / logloss / weighted-logloss 점수
  - metric 간 모델 순위 상관 (Spearman/Kendall)
- Why: 대회 metric 불확실 구간에서 잘못된 오프라인 의사결정 방지.
- Done criteria:
  - 단일 metric 사용 가능 여부를 rank-agreement로 판단

## P1 — Soon (run after P0)

### 4) Calibration + imbalance reliability
- Output: `eda/out/calibration_stats.json`
- Compute:
  - Brier, ECE(adaptive + fixed bins), calibration slope/intercept
  - AUC/PR-AUC/bootstrap CI
- Why: sample-scale 착시 개선(노이즈 lift) 차단.
- Done criteria:
  - 주요 metric에 CI 포함
  - CI overlap 기반 채택/보류 기준 정의

### 5) Cold-start slice map
- Output: `eda/out/cold_start_slices.json`
- Compute:
  - domain history length bins (0, 1-5, 6-20, >20)
  - unseen user/item rate (time split 기준)
  - slice별 positive rate + AUC(가능 시 CI)
- Why: target-in-history가 매우 낮은 조건에서 실제 취약 구간 식별.
- Done criteria:
  - 취약 slice와 안정 slice 구분
  - 이후 실험 보고서에 고정 섹션으로 연결

## P2 — Optional / defer-to-full-data

### 6) Positional semantics probe for array fids (89/90/91)
- Output: `eda/out/positional_semantics_probe.json`
- Why: slot 의미 해석은 sample 1000에서 과해석 리스크가 큼.
- Done criteria:
  - sample 기반 가설은 `hypotheses/`로만 기록
  - full-data 재검증 TODO 포함

## Agent-facing usage note

- 본 문서는 `notes/llm/wiki` 레이어(비권위)이며, 실험 주장 근거로 직접 사용 금지.
- 실험 gate 반영 시 `notes/refs/` 또는 `hypotheses/HXXX`로 승격(promotion) 필요.
