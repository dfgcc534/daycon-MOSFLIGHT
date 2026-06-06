## wlstn52 rank3 노트북 셀별 요약 (DACON 렌더 HTML, 라인=md_file.html)

**[md, line4-16]** 제목 "물리 기반 예측 + Kalman Filter 기반 예측 앙상블". 전략 4단계: (1)HyperPhysics_xy2 5-fold→OOF+test (단일 LB0.696), (2)Kalman잔차 Bi-GRU+Attn 5-fold(다양성), (3)OOF로 w1·Phys+w2·Kal grid search, (4)최적가중 test 결합.

**[code1, line17]** `!pip list` (환경).
**[code, line749]** `!cat /etc/issue`.
**[code, line781-784]** drive mount + open.zip unzip.

**[code, line5785-5796]** import: torch, nn, F, Dataset/DataLoader/WeightedRandomSampler, StandardScaler, CubicSpline, savgol_filter, tqdm.
**[code, line5797-5807]** set_seed(42); DEVICE cuda; `DT,T_PRED,R_HIT,N_T,EPS = 0.040,0.080,0.01,11,1e-8`.
**[code, line5808-5836]** load_stack: csv[[x,y,z]] → X_train/X_test (10000,11,3), Y_train(10000,3) float64.
**[code, line5837-5853]** stable_fold_id=md5(id)[:8]%5; make_kfold_splits(5); fold tr≈7980/va≈2020.

**[md, line5854-5857]** "메인 예측기 HyperPhysics_xy2 — CREE님 LB0.699+ 물리모델 기반".
**[code, line5858-5896]** SlidingWindowDataset: extended targets=[4,5,6,7,8,9,10,12], window min_win~max, w<11이면 v0=pts[1]-pts[0] 등속외삽 패딩; theta_weights=1+4·clamp(theta_last,0,1).
**[code, line5898-5915]** _ema_va_local (재귀 EMA 속도/가속); _soft_hit_loss(thr=0.013012,k=408.348).
**[code, line5917-5975]** extract_features → 24차원(v_local,a_local,speed,acc_mag,theta×6,p_std_local,v_local_abs,jerk_l,jerk_mag); dir_net softmax v_sm로 fwd/right/up 회전행렬 R; mean/std 정규화.
**[code, line5978-6002]** ResBlock; PriorBiasedLinear(weight/bias=0 init + prior_bias buffer); rodrigues_rotate.
**[code, line6004-6036]** HyperPhysics_xy2.__init__: 하이퍼 sh_thr0.013012/sh_k408.348/mse_w129.17/local_w0.0509/theta_thr1.0876/speed_thr0.0346/lr0.0054/wd0.00566; dir_net(29→24→10), temporal_net(9→32→6), dynamics_net(24→96→ResBlock→30), omega_net(24→48→3), diffusion_net(24→32→3).
**[code, line6053-6095]** forward: EMA α(σ·0.8+0.1)/β(σ·0.199+0.8), dyn 다항감쇠 exp_v/exp_a, rot_vec 3구간 omega_w softmax attention + omega_delta, theta/speed 게이트, rodrigues v 회전, pred_local=(w_v·exp(-exp_v))·v_rot+(w_a·exp(-exp_a))·al, pred_global=p_last+R·pred_local, log_var.
**[code, line6097-6104]** compute_loss=soft_hit+mse_w·MSE+local_w·0.5(exp(-log_var)·se+log_var).
**[code, line6107-6161]** train_phys_fold: epochs80, AdamW(lr/wd), CosineLR, batch256, WeightedRandomSampler(theta_weights) oversample, grad_clip1.0, early stop P=15, best=val r_Hit; seed=SEED+f.
**[code, line6164-6176]** predict_phys: state load, batch512 test 추론.

**[md, line6177]** "보조 예측기 Kalman BiGRUAttn".
**[code, line6177-6192]** yaw_angle/rotate_xy/inverse_rotate_xy/rotate_xy_seq; compute_yaw_theta(v_last atan2).
**[code, line6194-6213]** kalman_predict: 축별 1D CV Kalman, F/Q/R, t_pred=0.080 외삽.
**[code, line6215-6269]** cos_safe, noise_poly2(2차잔차std), noise_savgol(w5p2); build_scalar_features_40 → 40차원(speed/acc/jerk 통계+straightness+turn_cos+noise×3+이진플래그+log_max_acc+speed_roll8+cum_path11), 15컬럼 log1p.
**[code, line6271-6279]** build_seq_input_9ch: yaw회전 rel+v_p+a_p (11×9).
**[code, line6281-6299]** BiGRUAttn: GRU(9→48,bi), attn softmax pooling, fc(136→128→64), tanh×0.02 head.
**[code, line6301-6303]** loss_kal=d.mean()+0.3·sigmoid((d-R_HIT)/0.002).mean().
**[code, line6305-6359]** train_kalman_fold: epochs120, AdamW lr7e-4 wd1e-4, CosineLR, batch256, grad_clip1.0, P=25; 타겟=rotate_xy(Y-kal,theta); seed=SEED+f*7; StandardScaler seq/scalar.
**[code, line6361-6376]** predict_kalman: inverse_rotate → kal_test+pred_world.

**[md, line6377]** "학습".
**[code, line6377-6426]** kalman precompute(σ_obs0.5e-3,σ_proc2.0); scalar features; Phys 5-fold 루프(test=fold mean) → Phys OOF 0.6739; Kal 5-fold 루프 → Kal OOF 0.6581. (출력 line6437-6533: fold별 early stop 17~81ep)

**[md, line6534-6538]** "Model Ensemble — grid search voting, w=0.96".
**[code, line6539-6584]** grid w∈[0,1.01,0.02] OOF 최대화 → w_phys0.96/w_kal0.04, blend OOF 0.6744; gain<0.0001이면 phys-only fallback; submission.csv 저장 + sanity(nan/inf/range) + phys-only 백업. (출력 line6589-6604: blend 0.6744, range[-2.602,6.403], "Expected LB ~0.6744")
**[code, line6606-6624]** final_test=0.96·test_phys+0.04·test_kal → submission(ensemble).csv.
**[code, line6625-6640]** phys/kal state .pth + test npy 백업.
**[md, line6641]** "감사합니다."
