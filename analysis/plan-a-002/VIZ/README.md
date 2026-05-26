# plan-a-003 / KR008 — 잔차 보정 시각화

KR008 (Kalman-CV 잔차 GRU + 입력 yaw 회전 + 반사/노이즈 augment) 의 OOF 예측
10000 샘플(uncalibrated)을 기준으로 "잔차 보정이 실제로 어떻게 이루어졌는가"를 시각화.

데이터: `../results_kr008.npz` (oof_residual=world frame, oof_pred, y, kalman_main,
per_sample_hit). theta(yaw)는 raw train X 에서 `yaw_from_last_step` 으로 복원.
생성 스크립트: `$CLAUDE_JOB_DIR/viz_kr008.py`.

## 구조

최종 예측 = `kalman_main + inverse_rotate(model_out, theta)`.
`model_out` = yaw frame 잔차 = head 의 `tanh × 0.02` → **축별 ±2cm hard-cap** (대각 한계 2.83cm).

## 핵심 수치

| 항목 | 값 |
|---|---|
| kalman-alone hit_1cm | 0.5964 |
| corrected hit_1cm | 0.6671 (+0.0707) |
| miss→hit | 928 |
| hit→miss | 221 (net **+707**) |
| 잔차 크기 (3D) | mean 0.41cm / median 0.31cm / **max 2.51cm** < 한계 2.83cm |

## 그림

### `kr008_residual_correction_overview.png` (2×2)
- **(a)** yaw frame 잔차 산점도 — GRU 가 적용한 보정. 전부 ±2cm tanh 박스 내부 (캡 시각 확인).
- **(b)** 보정 전 오차 vs 후 오차 — 대각선 아래 = 보정이 도움. 좌하단 1cm×1cm = hit.
- **(c)** 잔차 크기 분포 — 대부분 sub-cm, 관측 max 2.51cm 가 이론 한계 2.83cm 미달.
- **(d)** 오차 ECDF (보정 전/후) — 곡선 좌측 이동 = 개선. 1cm 선 교차폭이 hit 증가분.

### `kr008_example_corrections.png` (3×4)
잔차 보정이 결과를 바꾼 **miss→hit 12개 예시** (잔차 큰 순). □kalman → △corrected (파란 화살표),
★truth, 점선 = 1cm hit 반경. kalman 이 원 밖 → 보정 후 원 안 진입.

## 해석

보정량 자체는 sub-cm 가 대부분이지만 1cm 임계 근처에서 net +707 hit 를 만든다.
±2cm 캡 때문에 kalman 오차가 큰 샘플(예: 5~11cm)은 보정해도 hit 불가 — 캡이 천장.
이 plan(반사/노이즈 augment)은 OOF-neutral (KR003 0.6667 ≈ KR008 0.6671) 로,
보정 메커니즘 자체는 KR003 carry 이며 augment 의 효과는 LB(사용자 gated)에서 판정.
