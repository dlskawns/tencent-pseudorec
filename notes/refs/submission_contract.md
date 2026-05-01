# Submission Contract & Workflow (former §13 + §14)

> **트리거**: `submission/infer.py`, `submission/local_validate.py`, `submission/prepare.sh`, `predictions.json`, 제출 패키징 작업 시 본 파일 먼저 읽기.
> 출처: 사용자 paste 2026-04-26 (대회 측 가이드 본문). 운영 가정은 별도 출처 확보 전엔 §13에 기록 금지.

대회는 **offline inference script 제출형**. 우리 코드가 플랫폼 컨테이너 안에서 실행/채점됨. 외부 API / 인터넷 / 원격 서비스 금지.

---

## §13.1 — 실행 환경 제약

- 코드는 로컬에서 돌지 않고 플랫폼 컨테이너에서 실행.
- 사용 가능: 로컬 파일, 환경 변수.
- 사용 금지: 외부 API, 인터넷 접속, 원격 서비스.

## §13.2 — 환경 변수 (경로 해결의 유일한 채널)

| Env | 의미 | 가이드상 필수 여부 |
|---|---|---|
| `EVAL_DATA_PATH` | test data 디렉토리 (read) | 필수 |
| `EVAL_RESULT_PATH` | 결과 저장 디렉토리 (write) | 필수 |
| `MODEL_OUTPUT_PATH` | 모델 출력 디렉토리 | 옵션 |
| `USER_CACHE_PATH` | 캐시 디렉토리 | 옵션 |

**경로 하드코딩 금지** — 위 env 외 절대경로 리터럴 금지.

## §13.3 — `infer.py` 진입점 계약 (HARD GATE)

- 파일명: `infer.py` 고정.
- `def main()` 정의 필수.
- `main()`은 **인자 0개** (argparse 금지). 직접 실행 가능.

```python
def main():
    ...

if __name__ == "__main__":
    main()
```

## §13.4 — 출력 위치 (HARD GATE)

- 파일명: `predictions.json` 고정.
- 위치: `${EVAL_RESULT_PATH}/predictions.json`.
- `EVAL_RESULT_PATH` **밖으로 쓰기 금지**.

## §13.5 — `predictions.json` 포맷 (HARD GATE)

```json
{
  "predictions": {
    "user_id_1": 0.8732,
    "user_id_2": 0.1245
  }
}
```

- key: user_id (string).
- value: float ∈ [0, 1] — 예측 conversion probability.
- test set의 **모든 user_id가 정확히 한 번** 포함되어야 함 (누락/추가 → fail).

> 중복 user_id 처리 (한 유저 multi-row) 정책은 가이드에 명시 안 됨 — 우리 운영 결정으로 `infer.py`의 `DUP_USER_POLICY` 상수에 명시 (현재 `mean`).

## §13.6 — 성능 / 결정성

- 컴퓨트 자원 제한 — 전체 데이터 일괄 로드 금지, 배치 추론.
- 출력은 **결정적** — 같은 입력은 같은 출력.

## §13.7 — Common failure cases (가이드 본문, 자동 fail 트리거)

1. 경로 하드코딩 (env 무시).
2. `main()` 누락.
3. `main()`이 인자 받음.
4. JSON 포맷 위반.
5. test user_id 누락 또는 추가.
6. 외부 API / 인터넷 사용.
7. `EVAL_RESULT_PATH` 밖 출력.

---

## §14 Submission Workflow Discipline

### §14.1 — 제출 전 게이트

| Gate | 검증 |
|---|---|
| G1 — 시그니처 | `def main()` 인자 0개. `submission/infer.py`에 argparse import 0건. |
| G2 — env-only 경로 | `os.environ[...]` 외 절대경로 리터럴 0건. |
| G3 — JSON 포맷 + 위치 | `submission/local_validate.py` 통과: top-key=`predictions`, value ∈ [0,1] float, 파일이 `${EVAL_RESULT_PATH}/predictions.json`에만 작성. |
| G4 — user_id 커버리지 | EVAL_DATA_PATH의 모든 user_id 정확히 한 번 (G3 검증기에 통합). |
| G5 — Determinism | seed 고정 + `model.eval()` + `torch.no_grad()` + dropout off. 같은 입력 두 번 → bit-identical JSON. |
| G6 — No-internet | `submission/infer.py`에 `requests`, `urllib`, `httpx`, `socket.connect` 등 외부-호출 import 0건. |

### §14.2 — `submission/README.md` 기록 의무

매 제출마다 1블록:
- 제출 시각 (UTC)
- git_sha
- local_validate 통과 증거
- 점수 (Success) / 실패 사유 (Failed)

### §14.3 — Phase Gate와의 관계

- **P0 통과 조건에 추가**: ≥ 1회 sanity 제출 Success — 플랫폼 round-trip 검증.
- **P1 진입 조건에 추가**: `submission/infer.py`가 P0 베이스라인 모델로 정상 동작.
- 제출 실패는 무조건 `progress.txt` 이터레이션 블록에 기록. 원인을 §13.7 항목으로 매핑.
