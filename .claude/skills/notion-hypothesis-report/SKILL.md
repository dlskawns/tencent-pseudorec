---
name: notion-hypothesis-report
description: Use when a hypothesis result arrives and you need to publish a clean Notion page. Trigger phrases: "H### 결과 노션에 정리", "노션에 올려줘", "가설 결과 페이지 만들어줘". Idempotently creates-or-updates a Notion page titled with the hypothesis ID/name and writes a fixed readable template with methodology, metrics, verdict, and follow-ups. Never commit API keys.
---

# notion-hypothesis-report

> 목적: 특정 가설(H###)의 방법론/결과를 노션 페이지에 표준 양식으로 게시.
> 페이지 제목: 반드시 해당 가설명 (예: `H011_aligned_pair_encoding`).
> 중복 방지 기준: `title == hypothesis_id`.

## When to invoke

**Auto (description 매칭):**
- "H### 결과 노션에 정리해줘"
- "가설 결과를 노션 페이지로 만들어줘"
- "방법론/결과 노션 업로드"

**Explicit:**
- `/notion-hypothesis-report H###`

## Required inputs

1. `hypothesis_id` (예: `H011_aligned_pair_encoding`)
2. source artifacts (가능한 범위):
   - `hypotheses/H###/problem.md`
   - `hypotheses/H###/transfer.md`
   - `hypotheses/H###/predictions.md`
   - `hypotheses/H###/verdict.md`
   - `experiments/E###/training_result.md`, `metrics.json` (있으면)
3. Notion destination:
   - `NOTION_PARENT_PAGE_ID` 또는 `NOTION_DATABASE_ID`

## Preflight checks (MANDATORY)

1. `NOTION_API_KEY` 존재 확인.
2. `NOTION_PARENT_PAGE_ID`와 `NOTION_DATABASE_ID`는 정확히 하나만 설정.
3. `hypothesis_id` 형식 확인 (`H\d{3}` prefix 권장).
4. source artifact 누락 시 빈칸 대신 `N/A` 명시.

## Security (MANDATORY)

- API key는 절대 레포 파일에 저장/커밋하지 말 것.
- 제공받은 키는 세션 실행에만 사용하고, 영구 저장 금지.
- 권장: 로컬 shell env로만 주입.
- 권한 최소화: Notion integration은 대상 page/database에만 공유.

```bash
read -s NOTION_API_KEY
export NOTION_API_KEY
export NOTION_PARENT_PAGE_ID="<notion-page-id>"
# 또는
export NOTION_DATABASE_ID="<notion-database-id>"
```

> `set -x` 상태에서 실행 금지 (토큰 노출 위험).

## Output format (Notion page template)

페이지 제목 = `{hypothesis_id}`

본문은 아래 순서를 유지:

1. **Summary**
   - one-paragraph: 핵심 결과, verdict, 다음 액션

2. **Hypothesis Card**
   - Hypothesis: H###
   - Status: supported / refuted / invalid / draft
   - Related Experiment(s): E###
   - Date (KST)

3. **Methodology**
   - Problem framing (1~3 bullets)
   - Transfer rationale (paper/pattern)
   - Implementation delta (baseline 대비 변경점)

4. **Result Metrics**
   - 표(또는 bullet 표준):
     - Platform AUC
     - OOF AUC
     - Δ vs control/anchor
     - Train wall time
     - Inference wall time

5. **Decision (Binary)**
   - PASS / REFUTED / INVALID
   - 임계치 기준과 실제 값 비교

6. **Findings (Carry-forward)**
   - F-1, F-2 ... (각 1~2줄)

7. **Risks / Caveats**
   - 데이터/seed/infra 한계

8. **Next Actions**
   - 다음 H 후보 1~2개
   - 필요한 추가 검증

9. **Source Links (repo paths)**
   - `hypotheses/H###/...`
   - `experiments/E###/...`

## Notion API execution guide

## Idempotent create-or-update flow (MANDATORY)

1. title이 `{hypothesis_id}` 인 기존 페이지 조회.
2. 기존 페이지가 있으면 page_id에 block append/update.
3. 기존 페이지가 없으면 새 page create.

검색 예시:

```bash
curl --fail-with-body -sS -X POST "https://api.notion.com/v1/search" \
  -H "Authorization: Bearer ${NOTION_API_KEY}" \
  -H "Notion-Version: 2022-06-28" \
  -H "Content-Type: application/json" \
  --data '{"query":"H011_aligned_pair_encoding","filter":{"value":"page","property":"object"}}'
```

### A) Parent Page 아래에 일반 페이지 생성

```bash
curl --fail-with-body -sS -X POST "https://api.notion.com/v1/pages" \
  -H "Authorization: Bearer ${NOTION_API_KEY}" \
  -H "Notion-Version: 2022-06-28" \
  -H "Content-Type: application/json" \
  --data @payload.json
```

`payload.json` 골격:

```json
{
  "parent": { "page_id": "${NOTION_PARENT_PAGE_ID}" },
  "properties": {
    "title": {
      "title": [{ "type": "text", "text": { "content": "H011_aligned_pair_encoding" } }]
    }
  },
  "children": []
}
```

### B) Database에 row(page) 생성

```json
{
  "parent": { "database_id": "${NOTION_DATABASE_ID}" },
  "properties": {
    "Name": {
      "title": [{ "type": "text", "text": { "content": "H011_aligned_pair_encoding" } }]
    }
  },
  "children": []
}
```

> DB 속성명(`Name`)은 워크스페이스 스키마에 맞게 조정.

DB 스키마 확인 예시:

```bash
curl --fail-with-body -sS -X GET "https://api.notion.com/v1/databases/${NOTION_DATABASE_ID}" \
  -H "Authorization: Bearer ${NOTION_API_KEY}" \
  -H "Notion-Version: 2022-06-28"
```

## Writing rules

1. 수치는 원문 artifact에 있는 값만 사용 (추정/환각 금지).
2. `PASS/REFUTED/INVALID`는 predictions/verdict 기준과 일치.
3. 한국어로 쓰되, metric label은 영문 병기 허용.
4. 긴 로그 원문을 통째로 붙이지 말고 핵심만 요약.
5. 페이지 하단에 `Updated at (KST)` 추가.
6. 본문이 길면 heading/section 단위로 block 분할 append.

## Ready-to-fill markdown body (for conversion into Notion blocks)

```md
# {HYPOTHESIS_ID}

## Summary
{핵심 요약 3~5문장}

## Hypothesis Card
- Hypothesis: {HYPOTHESIS_ID}
- Status: {supported|refuted|invalid|draft}
- Related Experiment(s): {E###, ...}
- Date (KST): {YYYY-MM-DD HH:mm}

## Methodology
- Problem framing: {bullet}
- Transfer rationale: {bullet}
- Implementation delta vs baseline: {bullet}

## Result Metrics
- Platform AUC: {value}
- OOF AUC: {value}
- Δ vs control/anchor: {value}
- Train wall time: {value}
- Inference wall time: {value}

## Decision (Binary)
- Verdict: {PASS|REFUTED|INVALID}
- Why: {threshold 대비 근거}

## Findings (Carry-forward)
- F-1: {내용}
- F-2: {내용}

## Risks / Caveats
- {내용}

## Next Actions
- {다음 H 후보/실험}

## Source Links
- hypotheses/{HYPOTHESIS_ID}/problem.md
- hypotheses/{HYPOTHESIS_ID}/transfer.md
- hypotheses/{HYPOTHESIS_ID}/predictions.md
- hypotheses/{HYPOTHESIS_ID}/verdict.md
- experiments/{EXPERIMENT_ID}/training_result.md
- experiments/{EXPERIMENT_ID}/metrics.json

Updated at (KST): {YYYY-MM-DD HH:mm}
```

## Anti-patterns

- API key를 파일에 하드코딩/커밋
- verdict와 모순되는 문구("promising") 삽입
- 가설 제목이 아닌 임의 제목 사용
- artifact 경로 누락 (재현성 저하)
