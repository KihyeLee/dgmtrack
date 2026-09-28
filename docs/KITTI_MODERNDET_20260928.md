# KITTI 현대 검출기 결과 (자동 생성 — 손으로 고치지 않는다)

생성: `tools/summarize_kitti_moderndet.py`. 사전 등록: `results/ablation_20260922/PREREGISTRATION.md` "KITTI 현대 검출기 (2026-09-28)".

검출은 OpenPCDet 공개 체크포인트를 KITTI tracking val 11시퀀스(3,904 스캔)에 직접 추론해 만들었다
(`tools/kitti_tracking_infer.py`). 두 검출기 모두 공개 설정 그대로(`SCORE_THRESH 0.1`, NMS 0.1).
추적기 인자는 nuScenes에서 정한 값을 바꾸지 않았다(`--vel_max_hits 2`, `--feedback 0`, `--seed 12345`).

## score_thr 0.1, 2 Hz (주 분석)

| 검출기 | KITTI Car AP | 구성 | 클래스 | sAMOTA | AMOTA | AMOTP | MOTA | IDS | FRAG | recall | precision |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Voxel R-CNN (Car) | 84.54 | 속도 없음 | car | 0.9330 | 0.4605 | 0.8095 | 0.7828 | 53 | 73 | 0.8901 | 0.9489 |
| Voxel R-CNN (Car) | 84.54 | 속도 없음 | pedestrian |  |  |  |  |  |  |  |  |
| Voxel R-CNN (Car) | 84.54 | 속도 없음 | cyclist |  |  |  |  |  |  |  |  |
| Voxel R-CNN (Car) | 84.54 | 정책 A | car | 0.9215 | 0.4735 | 0.7943 | 0.8334 | 29 | 49 | 0.9173 | 0.9536 |
| Voxel R-CNN (Car) | 84.54 | 정책 A | pedestrian |  |  |  |  |  |  |  |  |
| Voxel R-CNN (Car) | 84.54 | 정책 A | cyclist |  |  |  |  |  |  |  |  |
| Voxel R-CNN (Car) | 84.54 | **차이(정책 A − 없음)** | car | -0.0115 | +0.0131 | -0.0152 | +0.0506 | -24 | -24 | +0.0271 | +0.0047 |
| PV-RCNN | 83.61 | 속도 없음 | car | 0.9010 | 0.4332 | 0.8054 | 0.7404 | 77 | 91 | 0.8729 | 0.9377 |
| PV-RCNN | 83.61 | 속도 없음 | pedestrian | 0.8360 | 0.3909 | 0.6378 | 0.6745 | 8 | 27 | 0.8593 | 0.8297 |
| PV-RCNN | 83.61 | 속도 없음 | cyclist | 0.7380 | 0.3265 | 0.6779 | 0.6421 | 16 | 21 | 0.8175 | 0.8784 |
| PV-RCNN | 83.61 | 정책 A | car | 0.9032 | 0.4549 | 0.7904 | 0.7881 | 30 | 44 | 0.8778 | 0.9497 |
| PV-RCNN | 83.61 | 정책 A | pedestrian | 0.8292 | 0.3868 | 0.6394 | 0.6668 | 24 | 49 | 0.8800 | 0.8176 |
| PV-RCNN | 83.61 | 정책 A | cyclist | 0.7374 | 0.3263 | 0.6783 | 0.6421 | 16 | 21 | 0.8175 | 0.8784 |
| PV-RCNN | 83.61 | **차이(정책 A − 없음)** | car | +0.0023 | +0.0217 | -0.0150 | +0.0477 | -47 | -47 | +0.0049 | +0.0119 |
| PointRCNN | 78.70 | 속도 없음 | car | 0.7746 | 0.3427 | 0.6630 | 0.6816 | 79 | 114 | 0.7950 | 0.9609 |
| PointRCNN | 78.70 | 속도 없음 | pedestrian | 0.7317 | 0.2875 | 0.5153 | 0.6546 | 36 | 120 | 0.7520 | 0.9079 |
| PointRCNN | 78.70 | 속도 없음 | cyclist | 0.6121 | 0.2410 | 0.5215 | 0.6199 | 8 | 11 | 0.7091 | 0.9286 |
| PointRCNN | 78.70 | 정책 A | car | 0.7892 | 0.3559 | 0.6699 | 0.7281 | 51 | 86 | 0.8175 | 0.9653 |
| PointRCNN | 78.70 | 정책 A | pedestrian | 0.7039 | 0.2857 | 0.5009 | 0.6531 | 39 | 119 | 0.7515 | 0.9084 |
| PointRCNN | 78.70 | 정책 A | cyclist | 0.6124 | 0.2411 | 0.5199 | 0.6162 | 8 | 11 | 0.7055 | 0.9282 |
| PointRCNN | 78.70 | **차이(정책 A − 없음)** | car | +0.0146 | +0.0131 | +0.0068 | +0.0465 | -28 | -28 | +0.0226 | +0.0044 |

## score_thr 0.30, 2 Hz (보조 운영점)

| 검출기 | KITTI Car AP | 구성 | 클래스 | sAMOTA | AMOTA | AMOTP | MOTA | IDS | FRAG | recall | precision |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PV-RCNN | 83.61 | 속도 없음 | car | 0.9154 | 0.4407 | 0.7908 | 0.7693 | 34 | 49 | 0.8458 | 0.9660 |
| PV-RCNN | 83.61 | 속도 없음 | pedestrian | 0.8346 | 0.3955 | 0.6413 | 0.6857 | 34 | 68 | 0.9026 | 0.8221 |
| PV-RCNN | 83.61 | 속도 없음 | cyclist | 0.6991 | 0.3045 | 0.6593 | 0.6384 | 16 | 21 | 0.8394 | 0.8582 |
| PV-RCNN | 83.61 | 정책 A | car | 0.9053 | 0.4535 | 0.7762 | 0.8122 | 14 | 29 | 0.8729 | 0.9670 |
| PV-RCNN | 83.61 | 정책 A | pedestrian | 0.8238 | 0.3916 | 0.6410 | 0.6806 | 26 | 64 | 0.9016 | 0.8160 |
| PV-RCNN | 83.61 | 정책 A | cyclist | 0.7092 | 0.3104 | 0.6612 | 0.6494 | 14 | 19 | 0.8394 | 0.8614 |
| PV-RCNN | 83.61 | **차이(정책 A − 없음)** | car | -0.0101 | +0.0128 | -0.0146 | +0.0430 | -20 | -20 | +0.0271 | +0.0010 |
| PointRCNN | 78.70 | 속도 없음 | car | 0.7889 | 0.3409 | 0.6852 | 0.6739 | 97 | 137 | 0.7982 | 0.9611 |
| PointRCNN | 78.70 | 속도 없음 | pedestrian | 0.7277 | 0.2887 | 0.5205 | 0.6648 | 33 | 114 | 0.7283 | 0.9431 |
| PointRCNN | 78.70 | 속도 없음 | cyclist | 0.6053 | 0.2408 | 0.5176 | 0.6162 | 8 | 11 | 0.7080 | 0.9238 |
| PointRCNN | 78.70 | 정책 A | car | 0.7863 | 0.3541 | 0.6711 | 0.7275 | 33 | 74 | 0.8093 | 0.9639 |
| PointRCNN | 78.70 | 정책 A | pedestrian | 0.7252 | 0.2877 | 0.5197 | 0.6679 | 33 | 119 | 0.7515 | 0.9214 |
| PointRCNN | 78.70 | 정책 A | cyclist | 0.6089 | 0.2436 | 0.5187 | 0.6236 | 6 | 9 | 0.7080 | 0.9238 |
| PointRCNN | 78.70 | **차이(정책 A − 없음)** | car | -0.0026 | +0.0132 | -0.0141 | +0.0536 | -64 | -63 | +0.0111 | +0.0027 |

## score_thr 0.1, 10 Hz (보조)

| 검출기 | KITTI Car AP | 구성 | 클래스 | sAMOTA | AMOTA | AMOTP | MOTA | IDS | FRAG | recall | precision |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Voxel R-CNN (Car) | 84.54 | 속도 없음 | car | 0.9540 | 0.4771 | 0.8181 | 0.8680 | 61 | 142 | 0.9387 | 0.9548 |
| Voxel R-CNN (Car) | 84.54 | 속도 없음 | pedestrian |  |  |  |  |  |  |  |  |
| Voxel R-CNN (Car) | 84.54 | 속도 없음 | cyclist |  |  |  |  |  |  |  |  |
| Voxel R-CNN (Car) | 84.54 | 정책 A | car | 0.9578 | 0.4808 | 0.8179 | 0.8787 | 29 | 104 | 0.9404 | 0.9586 |
| Voxel R-CNN (Car) | 84.54 | 정책 A | pedestrian |  |  |  |  |  |  |  |  |
| Voxel R-CNN (Car) | 84.54 | 정책 A | cyclist |  |  |  |  |  |  |  |  |
| Voxel R-CNN (Car) | 84.54 | **차이(정책 A − 없음)** | car | +0.0038 | +0.0038 | -0.0001 | +0.0107 | -32 | -38 | +0.0018 | +0.0038 |
| PV-RCNN | 83.61 | 속도 없음 | car | 0.9456 | 0.4714 | 0.8169 | 0.8573 | 61 | 129 | 0.9226 | 0.9576 |
| PV-RCNN | 83.61 | 속도 없음 | pedestrian | 0.8493 | 0.4042 | 0.6383 | 0.7232 | 18 | 99 | 0.8552 | 0.8718 |
| PV-RCNN | 83.61 | 속도 없음 | cyclist | 0.7713 | 0.3376 | 0.6935 | 0.6306 | 0 | 24 | 0.8107 | 0.8227 |
| PV-RCNN | 83.61 | 정책 A | car | 0.9462 | 0.4725 | 0.8176 | 0.8565 | 51 | 116 | 0.9307 | 0.9489 |
| PV-RCNN | 83.61 | 정책 A | pedestrian | 0.8355 | 0.4010 | 0.6404 | 0.7128 | 17 | 101 | 0.8674 | 0.8540 |
| PV-RCNN | 83.61 | 정책 A | cyclist | 0.7664 | 0.3509 | 0.6874 | 0.7033 | 0 | 38 | 0.8867 | 0.8320 |
| PV-RCNN | 83.61 | **차이(정책 A − 없음)** | car | +0.0006 | +0.0011 | +0.0007 | -0.0007 | -10 | -13 | +0.0081 | -0.0087 |
| PointRCNN | 78.70 | 속도 없음 | car | 0.7881 | 0.3519 | 0.6682 | 0.7506 | 74 | 223 | 0.8297 | 0.9570 |
| PointRCNN | 78.70 | 속도 없음 | pedestrian | 0.7307 | 0.2930 | 0.5299 | 0.6839 | 37 | 413 | 0.7496 | 0.9279 |
| PointRCNN | 78.70 | 속도 없음 | cyclist | 0.6514 | 0.2464 | 0.5617 | 0.6387 | 0 | 21 | 0.6930 | 0.9340 |
| PointRCNN | 78.70 | 정책 A | car | 0.7890 | 0.3529 | 0.6691 | 0.7538 | 45 | 193 | 0.8304 | 0.9562 |
| PointRCNN | 78.70 | 정책 A | pedestrian | 0.7424 | 0.2938 | 0.5315 | 0.6838 | 40 | 413 | 0.7489 | 0.9287 |
| PointRCNN | 78.70 | 정책 A | cyclist | 0.6502 | 0.2458 | 0.5620 | 0.6447 | 0 | 21 | 0.6966 | 0.9371 |
| PointRCNN | 78.70 | **차이(정책 A − 없음)** | car | +0.0009 | +0.0011 | +0.0009 | +0.0032 | -29 | -30 | +0.0007 | -0.0009 |

## 사전 등록된 판정 (주 분석: 2 Hz, Car, sAMOTA, score_thr 0.1)

이 기준은 **PV-RCNN과 PointRCNN의 사전 등록**("KITTI 현대 검출기")에만 적용된다.
Voxel R-CNN은 공개 `SCORE_THRESH`가 0.3이어서 별도 사전 등록("KITTI 세 번째 검출기")의
기준(최저 2구간 제외 차이 ≥ +0.010 **및** IDS 감소)을 쓴다 — 아래 별도 항목.

- **PV-RCNN** (Car AP 83.61): Car sAMOTA 0.9010 → 0.9032, 차이 +0.0023 → **(b) 부분 재현**
- **PointRCNN** (Car AP 78.70): Car sAMOTA 0.7746 → 0.7892, 차이 +0.0146 → **(a) 재현**

- 공개 sAMOTA로는 검출기 품질이 오르면 이득이 줄어든다: PointRCNN +0.0146 → PV-RCNN +0.0023.
  **그 축소는 최저 재현율 구간 한 곳의 artifact다** — `tools/kitti_samota_threshold_diag.py`와 `results/kitti/SAMOTA_THRESHOLD_DIAG.json`을 함께 읽는다.

## 사전 등록된 판정 — Voxel R-CNN (Car) (2 Hz, Car, 공개 thr 0.30)

- 최저 2구간 제외 sAMOTA 차이 +0.0142, IDS 53 → 29 (-24) → **(a) 검출기 무관 확정**
- 공개 sAMOTA 차이는 -0.0115다. 세 검출기 모두 음수이며 최저 구간 artifact가 지배한다 — `results/kitti/SAMOTA_THRESHOLD_DIAG.json`.

## 상한 (정답 속도, 2 Hz)

- Voxel R-CNN (Car): Car sAMOTA 상한 차이 +0.0194
- PV-RCNN: Car sAMOTA 상한 차이 +0.0316; 정책이 그 중 7%를 얻는다

## 장면 단위 짝지은 부트스트랩 (중심거리 매칭, 참고용 — 판정에 쓰지 않는다)

| 쌍 | 지표 | 차이 | 95% 구간 |
|---|---|---:|---|
| k5_none_voxelrcnn_vs_k5_polA_voxelrcnn | amota | +0.0657 | [+0.0275, +0.1069] |
| k5_none_voxelrcnn_vs_k5_polA_voxelrcnn | amotp | -0.1001 | [-0.1974, -0.0466] |
| k5_none_voxelrcnn_vs_k5_polA_voxelrcnn | ids | -5.0000 | [-15.0000, +0.0000] |
| k5_none_voxelrcnn_vs_k5_polA_voxelrcnn | mota | +0.0553 | [+0.0088, +0.1472] |
| k1_none_voxelrcnn_vs_k1_polA_voxelrcnn | amota | +0.0115 | [+0.0037, +0.0209] |
| k1_none_voxelrcnn_vs_k1_polA_voxelrcnn | amotp | -0.0010 | [-0.0042, -0.0000] |
| k1_none_voxelrcnn_vs_k1_polA_voxelrcnn | ids | -6.0000 | [-29.0250, +13.0000] |
| k1_none_voxelrcnn_vs_k1_polA_voxelrcnn | mota | +0.0111 | [-0.0056, +0.0216] |
| k5_none_voxelrcnn_vs_k5_oracle_voxelrcnn | amota | +0.1247 | [+0.0901, +0.1707] |
| k5_none_voxelrcnn_vs_k5_oracle_voxelrcnn | amotp | -0.1937 | [-0.2872, -0.1814] |
| k5_none_voxelrcnn_vs_k5_oracle_voxelrcnn | ids | -9.0000 | [-31.0000, +5.0000] |
| k5_none_voxelrcnn_vs_k5_oracle_voxelrcnn | mota | +0.1189 | [-0.0851, +0.3142] |
| k5_none_pvrcnn_vs_k5_polA_pvrcnn | amota | +0.0260 | [-0.0200, +0.0461] |
| k5_none_pvrcnn_vs_k5_polA_pvrcnn | amotp | -0.0529 | [-0.0602, +0.0295] |
| k5_none_pvrcnn_vs_k5_polA_pvrcnn | ids | -20.0000 | [-53.0000, +3.0000] |
| k5_none_pvrcnn_vs_k5_polA_pvrcnn | mota | +0.0293 | [+0.0069, +0.0754] |
| k1_none_pvrcnn_vs_k1_polA_pvrcnn | amota | +0.0000 | [-0.0151, +0.0122] |
| k1_none_pvrcnn_vs_k1_polA_pvrcnn | amotp | -0.0018 | [-0.0153, +0.0010] |
| k1_none_pvrcnn_vs_k1_polA_pvrcnn | ids | -20.0000 | [-51.0000, -1.0000] |
| k1_none_pvrcnn_vs_k1_polA_pvrcnn | mota | +0.0009 | [-0.0283, +0.0174] |
| k5_none_pvrcnn_vs_k5_oracle_pvrcnn | amota | +0.0604 | [+0.0476, +0.0835] |
| k5_none_pvrcnn_vs_k5_oracle_pvrcnn | amotp | -0.1027 | [-0.1251, -0.0697] |
| k5_none_pvrcnn_vs_k5_oracle_pvrcnn | ids | -34.0000 | [-77.0000, -2.0000] |
| k5_none_pvrcnn_vs_k5_oracle_pvrcnn | mota | +0.0541 | [+0.0124, +0.1193] |
| k5_none_pointrcnn3c_vs_k5_polA_pointrcnn3c | amota | +0.0162 | [+0.0083, +0.0363] |
| k5_none_pointrcnn3c_vs_k5_polA_pointrcnn3c | amotp | -0.0129 | [-0.1150, -0.0065] |
| k5_none_pointrcnn3c_vs_k5_polA_pointrcnn3c | ids | -39.0000 | [-73.0000, -11.0000] |
| k5_none_pointrcnn3c_vs_k5_polA_pointrcnn3c | mota | +0.0282 | [+0.0050, +0.0743] |
| k1_none_pointrcnn3c_vs_k1_polA_pointrcnn3c | amota | +0.0027 | [-0.0083, +0.0065] |
| k1_none_pointrcnn3c_vs_k1_polA_pointrcnn3c | amotp | +0.0053 | [-0.0017, +0.0150] |
| k1_none_pointrcnn3c_vs_k1_polA_pointrcnn3c | ids | -23.0000 | [-66.0000, +12.0000] |
| k1_none_pointrcnn3c_vs_k1_polA_pointrcnn3c | mota | +0.0093 | [-0.0004, +0.0290] |
