# LLM Wiki Overlay (notes/llm)

이 디렉토리는 기존 TAAC 하네스 위에 올리는 **비권위(non-authoritative) 지식 레이어**다.

- `raw/`: 수집한 원문/노트 스냅샷 (읽기 전용 원칙)
- `wiki/`: LLM이 유지하는 합성/요약/연결 페이지
- `authority.md`: 근거 권한 우선순위 + 승격(promotion) 규칙

중요:
- `notes/llm/*` 단독으로는 실험 주장 근거로 사용할 수 없다.
- 실험/판정 근거로 쓰려면 `authority.md`의 promotion gate를 통과해
  `papers/`, `notes/refs/`, `hypotheses/*`로 승격해야 한다.

---

## Quick start

1. 새 소스 추가: `notes/llm/raw/` 아래에 파일 배치
2. 요약/합성 작성: `notes/llm/wiki/` 아래 페이지 생성/수정
3. `notes/llm/wiki/index.md` 갱신
4. `notes/llm/wiki/log.md`에 append-only 기록
5. 실험에 반영 필요 시 promotion gate 수행
