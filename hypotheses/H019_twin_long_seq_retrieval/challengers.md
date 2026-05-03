# H019 — Challengers (≥ 2 reverse frames)

> §10.3 + §17.4 — H019 = paradigm shift first entry. rotation 정당화
> trivial (미경험 카테고리 first-touch). 단 paradigm shift 자체의
> reverse frame 명시.

---

## §17.4 rotation 정당화 (auto-justified)

**카테고리**: `retrieval_long_seq` — 미경험 first-touch.
**§10.7 audit**: 다른 미경험 카테고리들 (`backbone_replacement`,
`debiasing`) 와 동일 priority. **TWIN 우선 이유**:
1. §3.5 quantitative motivation 가장 강함 (p90 ≫ 100).
2. L4 (truncate) retire 의 retrieval branch open — diagnosis 의 직접
   continuation.
3. T3 cost ~$15 (vs OneTrans full $30+) — cost cap 친화.
4. Tencent paper (대회 organizer) — UNI-REC 의 sequence axis 강화 직접
   alignment.

---

## Frame A — "Retrieval 도 cohort hard ceiling 못 풂"

**가설**: cohort drift 가 platform 일반화의 진짜 hard ceiling. retrieval
form (top-K) 도 결국 user-level history 의 일부만 보는 것 — cohort
distribution 자체는 변경 못 함. dense (H014) 와 retrieval 의 차이는
mechanism 효율성이지 ceiling break 와 무관.

**근거**:
- L2 cohort drift 가설 4 H 누적 (H011/H012/H013/H015) 모두 OOF+/Platform−
  pattern. mechanism 위치/종류 무관.
- Recent Findings F-D: H012 (MoE multi-domain) ≈ H015 (recency loss) =
  같은 ceiling 영역. 두 mechanism class 무관 같은 결과 → ceiling 자체가
  mechanism 과 무관한 외부 source (cohort drift).
- TWIN 도 결국 user history 의 일부만 attend → cohort 분포 변경 0.

**Falsification 조건**: H019 Δ vs H010 corrected ≥ +0.005pt → Frame A REFUTED
(retrieval 이 ceiling 깨면 cohort drift 가설 자체 약화).

**Frame A confirmed (Δ ≤ +0.001pt) 시 carry-forward**: paradigm shift
다른 form (HSTU trunk / OneTrans) 도 cohort handling 못 풀 가능성 높음 →
P3 phase (multi-modal 데이터) 대기 또는 cohort 자체 modeling
(distribution learning H — H020 후보 재선정).

---

## Frame B — "TWIN 구현 detail 의 sample-scale viability 위험"

**가설**: TWIN paper 는 100M+ user / 1B+ event 환경 검증. demo_1000 sample
+ 1000-row scale 에서 GSU/ESU 의 retrieval signal 자체가 noise 보다 작을
가능성. top-K=64 가 너무 큼 (entire seq L≤128 환경에서 retrieval 의미
없음).

**근거**:
- §17.5 sample-scale = code-path verification only. TWIN 의 "retrieval
  lift" 는 paper 환경 가정.
- §10.6 trainable params budget ≤ 200 (sample-scale). TWIN GSU candidate
  embedding lookup table 추가 시 budget 초과 위험.
- 현 envelope seq_max_lens [64, 128] → top-K=64 는 사실상 truncate-128 과
  거의 동일 (no retrieval).

**Falsification 조건**: H019 sample-scale sanity 가 GSU lookup OOM 또는
top-K 충분치 못한 경우 → §17.5 code-path P1 fail.

**Frame B confirmed 시 carry-forward**: H019 sub-form = top-K=32 + seq
expand to 256 (envelope 변경 동시) 또는 GSU 단순화 (hard search only).
또는 **full-data 도착 까지 H019 launch 보류**.

---

## Frame C — "Paradigm shift 의 비용/risk ratio 부적절"

**가설**: T3 ~$15/job + 누적 ~46h + paradigm shift 1 회 시도 후 negative
면 다음 시도 비용 어려움. cost-effective 하지 않음. 차라리 **현재
champion (H010 corrected 0.837806) 로 lock + ensemble / multi-seed
robust 측정** 으로 가는 게 ROI 높음.

**근거**:
- §17.6 cost cap (T3 per-campaign ≤ $100). 현재 ~46h * Taiji 가격 unknown
  + H018 추가 + H019 T3 → cap 도달 임박.
- 4 H 누적 REFUTED 패턴 → paradigm shift 도 같은 패턴 가능성 (Frame A).
- 대안: 현 H010 ensemble (seed 3-5 paired bootstrap) 로 robust 측정 +
  uncertainty 정량화.

**Falsification 조건**: H019 PASS strong (Δ ≥ +0.005pt) → Frame C REFUTED
(paradigm shift cost 정당화).

**Frame C confirmed (REFUTED 누적) 시 carry-forward**: H020 = ensemble /
multi-seed measurement H (mechanism mutation 0, infra mutation only).
cost cap 친화. paradigm shift 시도 multi-modal 데이터 도착 까지 보류.

---

## Counter-argument 종합 (왜 그래도 H019 진행)

1. **Falsification value 자체 큼**: H019 REFUTED → cohort drift hard
   ceiling 가설 강한 confirm + paradigm shift 종료 결정 가능. PASS 시
   anchor 갱신.
2. **Cost-effective relative**: TWIN GSU minimum viable (T3 $15) <
   HSTU/OneTrans full ($30+) — paradigm shift class 안 가장 cheap.
3. **§3.5 quantitative motivation 강함**: p90 1393~2215 vs envelope 64-128
   = 95%+ 손실. Frame A 가 진실 라도 retrieval 시도 없이 ceiling 단정 무리.
4. **rotation mandatory**: §17.4 + §10.7 strict reading 시 H019 같은
   카테고리 (temporal_cohort 5번째) 금지. retrieval 또는 backbone 또는
   debiasing 강제.
