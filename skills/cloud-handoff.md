# Skill: cloud-handoff

> 학습은 사용자가 클라우드 플랫폼에서 직접 돌린다. 본 skill은 **실행 직전** 사용자가 받을 self-contained 패킷을 만든다.

## When to invoke
- 새 실험 (`experiments/{exp_id}/card.yaml`)이 `compute_tier: T2.X` 또는 `T3.X` 일 때.
- 사용자가 "E??? 클라우드에서 돌리자" 또는 "training request 만들어줘" 라고 할 때.
- 직접 학습을 실행하지 않는다 — 사용자만 실행.

## Inputs
- `experiments/{exp_id}/card.yaml` (config + falsification thresholds).
- `experiments/{exp_id}/run.sh` (1줄 entrypoint).
- `competition/` (patched 코드).
- `data/schema.json` (없으면 `make_schema.py` 호출 후 생성).
- `hypotheses/{H}/{transfer,predictions}.md` (claim 인용용).

## Outputs (모두 `experiments/{exp_id}/`)
1. **`training_request.md`** — 사용자가 읽을 hand-off packet:
   - TL;DR (claim 1줄, hypothesis ID, tier, expected wall, cost cap)
   - Prerequisites (GPU spec, deps with pinned versions)
   - Upload list (정확히 무엇을 플랫폼에 올릴지)
   - Commands (copy-paste numbered, env var 포함)
   - Expected wall time + cost estimate
   - Artifacts to download back
   - Falsification thresholds (card.yaml에서 복사)
   - Pre-run sanity check 1줄 명령
   - Where to fill `training_result.md`
2. **`{exp_id}_package.tar.gz`** — self-contained 업로드 번들 (build_cloud_package.py로 생성).
3. **`training_result.md`** — placeholder (사용자가 클라우드에서 결과 받은 후 채울 양식).

## Workflow
1. **Pre-flight check**:
   - `card.yaml`의 `compute_tier`, `claim_scope`, `falsification` 필드 확인. 누락 시 차단.
   - `run.sh`가 zero-arg 가능한지 확인 (env-only).
   - `competition/{model.py, dataset.py, trainer.py, train.py, utils.py}` 존재 확인.
2. **Compute config_sha256** from card.yaml `config:` block. card.yaml에 박음.
3. **Build package**:
   ```bash
   .venv/bin/python skills/_helpers/build_cloud_package.py \
       --exp-id {exp_id} \
       --include-data {true|false}   # demo는 true (40MB), full data는 false
   ```
   결과: `experiments/{exp_id}/{exp_id}_package.tar.gz`.
4. **Write training_request.md** from `experiments/_TEMPLATE/training_request.md` substituted with concrete values.
5. **Copy training_result.md template** as placeholder for user.
6. **Print summary to user**:
   - request packet 경로 (1줄)
   - 패키지 크기 (1줄)
   - 핵심 commands (3-5줄)
   - "결과는 training_result.md에 paste 후 cloud-intake skill 호출"

## Anti-patterns
- **직접 클라우드 호출 금지**: Modal SDK, Colab API 등 자동 실행 금지. 사용자가 손으로 돌린다.
- **로컬 absolute path 노출 금지**: 패키지 안 path는 모두 relative 또는 `$PWD` 기준.
- **venv 포함 금지**: `.venv-arm64/`, `.git/`, `__pycache__/`, `.DS_Store` 모두 제외.
- **민감정보 누출 금지**: data/ 안에 평문 user_id 외 secret 없는지 자동 grep.
- **`--exp-id` 기준 디렉토리 외부 파일 포함 금지**: build_cloud_package.py가 강제.

## Sanity check before declaring done
- 패키지를 `/tmp/test_extract/`에 풀어봤을 때 `bash experiments/{exp_id}/run.sh --num_epochs 1` 가 1 step이라도 통과하는지 (CPU/MPS 로컬 dry-run).
- 미통과 시 패키지 폐기, 원인 보고.
