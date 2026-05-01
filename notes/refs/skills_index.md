# Skills Index (former §12)

> **트리거**: 스킬 호출 / 스킬 등록·수정 / `skills/` 디렉토리 작업 시 본 파일 먼저 읽기.

| Skill | Trigger | Output |
|---|---|---|
| `redefine-problem` | 새 H / phase 시작 | `HXXX/problem.md` |
| `challenger` | 새 H | `HXXX/challengers.md` |
| `literature-scout` | 새 개념 필요 시 | `papers/{cat}/*.md` + `_summary.md` |
| `method-transfer` | P1+ 모든 실험 전 | `HXXX/transfer.md` (§⑤ UNI-REC alignment 포함) |
| `hypothesize` | problem 후 | `HXXX/predictions.md` |
| `experiment-card` | 실험 전 | `HXXX/experiments/EYYY/card.yaml` |
| `leakage-audit` | split 코드 작성 시 | `notes.md`에 통과 증빙 |
| `ablation-matrix` | phase 말미 | `HXXX/ablations.md` |
| `reproducibility` | `metrics.json` 작성 시 | 메타 필드 |
| `verify-claim` | 모든 수치 인용 | 1차소스 명시 |
| `journal-update` | 이터레이션 종료 | `progress.txt` append |
| `submission-gate` | 제출 직전 | `local_validate.py` G1–G6 통과 |
| `cloud-handoff` | T2/T3 실험 실행 직전 | `experiments/{id}/training_request.md` + `{id}_package.tar.gz` |
| `cloud-intake` | 사용자 결과 paste 시 | `training_result.md` + INDEX/verdict/progress 갱신 |

스킬 본문은 `tencent-cc/skills/` 14종을 참조. 본 프로젝트가 본격화하면 카피 또는 새로 작성.
