# Authority Order & Promotion Gate

이 문서는 `notes/llm/*` 콘텐츠의 권한 범위를 정의한다.

## 1) Authority order (high -> low)

1. **Tier A (authoritative):**
   - `data/` (immutable)
   - `eda/out/*.json` (측정 산출)
   - `hypotheses/*`, `experiments/*`, `progress.txt` (공식 실험 기록)
2. **Tier B (governed references):**
   - `notes/refs/*`
   - `papers/*`
3. **Tier C (LLM overlay, non-authoritative):**
   - `notes/llm/wiki/*`
4. **Tier D (raw intake cache):**
   - `notes/llm/raw/*`

원칙:
- Tier C/D는 탐색/정리 목적이며, 단독으로는 판정 근거가 아니다.
- 충돌 시 항상 상위 tier를 우선한다.

## 2) Promotion gate (C/D -> B/A)

`notes/llm/*` 정보를 공식 근거로 사용하려면 아래를 모두 충족해야 한다.

1. **Source traceability**
   - 출처 경로/URL/버전/날짜가 명시되어야 함
2. **Verification**
   - 데이터 팩트는 `data/`, `eda/out/*.json`, `competition/ns_groups.json` 등 1차 근거로 재검증
3. **Placement**
   - 성격에 맞는 경로로 승격:
     - 문헌/개념: `papers/*`
     - 운영 규칙: `notes/refs/*`
     - 가설/판정 근거: `hypotheses/*`, `experiments/*`
4. **Journaling**
   - `progress.txt`에 승격 사실(무엇을, 왜, 어디로)을 append

## 3) Prohibitions

- `notes/llm/raw/*`를 직접 `card.yaml`/`verdict.md`의 단일 근거로 인용 금지
- `data/` 수정 금지 (immutable)
- `progress.txt` rewrite 금지 (append-only)

## 4) Commit hygiene

- 권장: wiki overlay 변경 커밋과 실험 결과/코드 커밋 분리
- 대규모 자동 생성 diff는 작은 묶음으로 분할
