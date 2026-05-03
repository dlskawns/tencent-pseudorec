---
title: "LLM Wiki 운영 개요"
type: "overview"
status: "draft"
created_at: "2026-05-01"
updated_at: "2026-05-01"
sources:
  - path: "notes/llm/authority.md"
    kind: "file"
confidence: "high"
promotion_state: "not-promoted"
---

# LLM Wiki 운영 개요

## 목적

- 탐색/문헌/아이디어를 누적형 지식으로 정리
- 기존 실험 하네스(`hypotheses/experiments/progress`)를 대체하지 않고 보조

## 범위

- 작성/유지 대상: `notes/llm/wiki/*`
- 원문 intake 대상: `notes/llm/raw/*`

## 비범위

- 공식 판정/실험 근거의 단독 출처 역할
- immutable 데이터(`data/`) 수정

## 운영 루프

1. ingest: raw 소스 추가 + wiki 페이지 업데이트
2. query: 질문 기반 합성 페이지 생성/갱신
3. lint: 모순/고아 링크/낡은 주장 점검
4. promote: 가치 있는 내용만 상위 tier로 승격
