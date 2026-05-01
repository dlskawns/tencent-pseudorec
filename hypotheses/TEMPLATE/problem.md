# HXXX — Problem Statement

## What we're trying to explain
(현재 관찰된 현상 또는 우리가 풀려는 sub-problem 한 단락. 모델링 의견 X, 데이터 사실 + 측정 가능한 gap만.)

## Why now
(왜 이 가설을 지금 다루는가. 직전 H의 carry-forward, 데이터 신호, phase gate 요구사항.)

## Scope
- In: (다룰 변수/feature/메커니즘)
- Out: (의도적으로 미루는 것, 그리고 이유)

## UNI-REC axes
- Sequential: (seq encoder 어떻게 다루나)
- Interaction: (feature cross 어떻게 다루나)
- Bridging mechanism: (두 axis가 한 블록에서 gradient 공유하나? P1+ 필수)

## Success / Failure conditions
- Success: (정량적 metric + 임계치 + OOF에서 재현)
- Failure: (반증 조건. predictions.md와 일치)

## Frozen facts referenced
- CLAUDE.md §3.{x.y}
- eda/out/*.json 항목

## Inheritance from prior H
(직전 H의 verdict.md F-N carry-forward 중 본 H가 상속하는 것)
