# TrajGRU-Physics (남뻐글이, LB 0.699 base) — notebook cell dump

## §0 markdown (cell line 4-23)
제목 "TrajGRU-Physics — 물리구조 신경망 (LB 0.699 베이스 + 구조 개선)". 환경: Win11, Python 3.14.5, numpy 2.4.4, pandas 3.0.3, sklearn 1.8.0, torch 2.11.0+cu128. 출력 submissions_nn/submission_trajgru_physics.csv. 재현성 seed 42/1337/2023 고정.

## §0 config cell (line 24-83)
- imports: torch, WeightedRandomSampler, KFold, tqdm. set_seed(seed) 함수.
- device: cuda > mps > cpu.
- 경로 상대(ROOT="."), train/ test/ train_labels.csv sample_submission.csv.
- **설정**: R_HIT=0.01, N_FOLDS=5, SEEDS=[42,1337,2023], MAX_EPOCHS=40, PATIENCE=6, BATCH=512, MIN_WIN=3, AUG_MODE="extended", USE_SEQ_ENCODER=True.

## §1 데이터 로드 (line 84-118)
- label_ids/submission_ids 교집합으로 파일 필터.
- read_xyz: timestep_ms sort, [x,y,z] float32, shape==(11,3) 강제(아니면 ValueError).
- X_train (N,11,3), X_test (M,11,3), y_train (N,3)=train_labels[x,y,z] (target=스텝12).

## §2 SlidingWindowDataset 증강 (line 119-175)
- targets = [4,5,6,7,8,9,10,12] (extended). 각 target_idx에 end_idx=target_idx-2, w∈[min_win,max_w) 전 window 생성.
- target_idx==12 → y_tensor[i], 아니면 X_orig[target_idx] (다중타깃 self-supervision).
- w<11이면 등속 외삽 패딩: v0=pts[1]-pts[0], 과거방향 역투영(line 147-153), 길이 11 통일.
- theta_weights = 1+4·clamp(theta_last,0,1) (급회전 오버샘플 가중, line 169).

## §3 extract_features + NN 컴포넌트 (line 176-322)
- _ema_va_local(diffs_local, alpha, beta): EMA 속도 vl + 가속 a (line 179-197).
- _soft_hit_loss(pred,target,thr=0.013012,k=408.348): 1-sigmoid(-(‖·‖-thr)·k).mean (line 200-201).
- extract_features (line 204-270): diffs, theta_seq(acos), theta 파생6종, dir_net softmax heading or 3step, R=[fwd,right,up] 로컬프레임, v_local/a_local/speed/acc_mag/p_std_local/v_local_abs/jerk_l/jerk_mag → cat 24-dim → z-score(mean/std_stats).
- ResBlock(dim): Linear+LN+GELU+Dropout(0.15)+Linear, LN residual (line 273-283).
- PriorBiasedLinear: zero-init weight/bias + prior_bias buffer (line 286-296).
- rodrigues_rotate(v,w) (line 299-306).
- **SeqEncoder(hidden=32)**: GRU(3→32,1layer,batch_first)+head(LN+GELU+Linear(32→3)), 마지막 Linear **zero-init** (line 309-322).

## §4 TrajPhysicsNet (line 323-482)
- 하이퍼: sh_thr=0.013012, sh_k=408.348044, mse_w=129.172037, local_w=0.050941, theta_thr=1.087618, speed_thr=0.034583, lr=0.005400, wd=0.005659 (line 329-337).
- dir_net Linear(29→24)+LN+GELU+PriorBiasedLinear(24→10) prior_dir (line 342-346).
- temporal_net Linear(9→32)+...+PriorBiasedLinear(32→6) → alpha/beta EMA gate (line 349-352).
- dynamics_net Linear(24→96)+LN+GELU+ResBlock(96)+PriorBiasedLinear(96→30) prior_dyn → w_v,w_a,exp_v,exp_a softplus 다항 감쇠 (line 359-363, 410-424).
- omega_w(3 param) + omega_net(LN+Linear(24→48)+GELU+Linear(48→3)) → 회전 attention + Rodrigues (line 365-374, 431-459).
- diffusion_net Linear(24→32)+LN+GELU+Linear(32→3) → log_var clamp[-5,5] (line 376-379, 467).
- forward: pred_local=(w_v·exp(-exp_v))·v_rotated+(w_a·exp(-exp_a))·al [+seq_encoder(diffs_local)], pred_global=p_last+R·pred_local (line 461-469).
- compute_loss: soft_hit + mse_w·MSE(global) + local_w·이분산NLL(local) (line 472-482).

## §5 K-fold 학습+앙상블 (line 483-575)
- train_one(line 486-538): tr_ds(extended,min_win3)+WeightedRandomSampler(theta_weights,replacement), va_ds(standard,min_win11). mean/std_stats fold train로 1회 산출. AdamW(lr0.0054,wd0.00566)+CosineAnnealingLR(T_max40), grad clip 1.0. early stop on **val hit-rate**, patience6, best_state load.
- predict(line 541-551): batch 1024, TTA 없이 1회 forward.
- 메인 루프(line 559-568): for seed in [42,1337,2023]: KFold(5,shuffle,random_state=seed) → train_one → oof_pred[va_idx]=predict, test_pred_sum+=predict(X_test). **총 15모델**.
- test_pred = test_pred_sum/15 (균등평균). oof_hit=mean(‖oof-y‖≤0.01) "OOF hit (LB 추정)" (line 571-575).

## §6 제출 (line 576-592)
- sub_df(id,x,y,z) → sample_submission[id] merge how=left, NaN시 ValueError. submissions_nn/submission_trajgru_physics.csv 저장. **후처리(clip/LGBM/calibration) 없음**.
