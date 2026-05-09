# H029 — Method Transfer

## Source

- **Keskar et al. 2017** "On Large-Batch Training" — large batch
  generalization gap. small batch + small lr 가 sharp minima 회피.
- **Goyal et al. 2017** "Linear Scaling Rule" — batch K 배 → lr K 배 (H013
  검증, REFUTED).
- **Hoffer et al. 2017** "Train longer, generalize better" — small batch +
  longer training schedules.

## Mechanism

H010 byte-identical EXCEPT `--batch_size 256` (vs prior 2048/1024) + `--lr
1e-4` (unchanged but explicit at small batch). 1 launch.

## §17.2 EXEMPT (measurement H, no mechanism mutation)

regime 변경 = optimization regime 만. mechanism stack 그대로.
`measurement` re-entry (methodology framework).

## §⑤ UNI-REC alignment

mechanism / sequential / interaction 변경 없음. optimization-axis 만.
