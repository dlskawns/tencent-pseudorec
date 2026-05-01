# Hooks (former §11)

> **트리거**: `.claude/settings.local.json` 수정 / hook 활성화·비활성화 / hook 디버깅 시 본 파일 먼저 읽기.

- **method-transfer hook (P1 이후 ACTIVE)**: `PreToolUse(Bash command matches "run.sh" or train script)` → `hypotheses/{active}/transfer.md` 존재 확인. 없으면 차단.
- **submission gate hook**: `submission/infer.py` 또는 `predictions.json`을 만지는 명령 전에 `submission/local_validate.py` 통과 증빙 요구.
- 활성/해제는 `.claude/settings.local.json` + `hypotheses/INDEX.md`의 `active_phase` 필드.

## §17 PreToolUse hook 후보 (P1+ 활성화 — hypothesis_workflow.md §17 참조)

- `experiments/{id}/card.yaml` 의 `compute_tier:` 누락 시 차단.
- `card.yaml` 의 `claim_scope:` 누락 시 차단.
- 같은 `primary_category` 가 직전 2개 H에서 등장 + `challengers.md`에 "재진입 정당화" 섹션 부재 시 차단.
- T2/T3 실험에서 `training_request.md` 누락 시 차단 (cloud-handoff skill 미통과).
