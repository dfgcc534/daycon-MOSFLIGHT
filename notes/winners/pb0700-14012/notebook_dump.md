## 노우딘 우승 노트북 (Public 0.7022 / Private 0.700) 셀별 덤프

### §0 마크다운 (L4-21)
제목 "모기 비행 궤적 예측 — GRU + Transformer + 물리 디코더". 핵심: R-Hit@1cm은 1cm 경계 안에 드느냐의 이진지표 → 좌표 직접예측 대신 운동방정식 계수 학습 + 경계 안으로 밀어넣기. 구성 9단계. 데이터경로 Kaggle 기준, Dacon은 DATA_DIR 수정 안내.

### §1 설정·재현성 (L23-82)
HIDDEN=128, N_LAYERS=2, DROPOUT=0.08, BATCH=512, PRE_EPOCHS=40, FINE_EPOCHS=60, LR=1e-3, FINE_LR=LR*0.20, N_SPLITS=10, R_HIT=0.01, EPS=1e-8, PATIENCE=15, MIN_EPOCHS=5. Transformer: N_HEADS=4, N_TF_LAYERS=2, TF_DIM=256. seed42 완전고정(random/np/torch/cuda + cudnn deterministic, use_deterministic_algorithms).

### §2 데이터 로딩 (L83-98)
read_xyz: csv를 timestep_ms sort → x,y,z float32. load_all: glob 정렬 stack + ids. train_labels를 train_ids 순서 reindex, Y_train=labels[x,y,z].

### §3 피처 엔지니어링 (L99-181)
make_seq_features: 마지막6 timestep(부족시 left-pad), step당 11피처(spd,pspd/spd,ap/spd,pnorm/spd,tcos,curv,z_spd,z_acc,cross_z,jnorm/spd,1) + 글로벌6(spd_trend,tsig,zt,espd,lspd,z_lspd)을 broadcast → dim17. extract_physics_ctx: p0,d1v,ap(접선가속),aperp(법선가속),d2v,jerk. make_target_params(horizon=2): delta=y-p0를 접선/법선 분해해 d1s,pars,perps,ts,d2s,jerks 6 GT계수(각 clip).

### §4 물리 디코더·손실 (L182-203)
physics_predict_torch: pred=p0+d1s*vs*d1v+pars*as*ap+perps*as*aperp(+d2s*vs*d2v+jerks*as*jerk), vs=h/2*ts, as=(h/2)²ts². adaptive_loss = huber(err,0,delta0.015) + 0.15*(-mean sigmoid(-(err-0.01)/0.003)) + 0.01*(relu(d1s-2.5)²+relu(0.3-d1s)²).

### §5 모델 GRU+Transformer (L204-252)
GRU(17,128,2layer,drop0.08)→CLS token prepend→TransformerEncoder(d128,head4,ff256,gelu,Pre-LN)x2→CLS만 LayerNorm→shared(Lin128→128 SiLU drop→128→64 SiLU)→6 head(d1s=softplus, pars=tanh*2, perps=tanh*8, ts=sigmoid*0.8+0.6, d2s=tanh*1, jerks=tanh*1).

### §6 Dataset·피처생성 (L253-331)
AdaptDS/TestAdaptDS(seq,p0,d1v,ap,aperp[,y],d2v,jerk). 본학습 피처 + 사전학습 데이터: h∈{1,2}, ei∈range(3,T-h) 모든 중간시점 잘라 y=x[ei+h] 대량 생성(PRE_*). seq_mean/std=PRE+TRN으로 z-score, ns()로 PRE/TRN/TST 정규화. SEQ_DIM=17. hit_rate=mean(dist≤0.01),mean dist.

### §7 학습루프 (L332-460)
KFold(10,shuffle,seed42). pretrain ld는 전체 PRE 1회생성 fold공유. fold당: 사전학습40ep(AdamW lr1e-3 wd1e-4, warmup5+cosine LambdaLR, clip2.0, loss는 d2v/jerk 미전달=4계수)→best load→파인튜닝60ep(AdamW lr2e-4, ReduceLROnPlateau max p8 f0.5, loss에 d2v/jerk 전달=6계수, early stop p20). fine<pre면 pre_best 롤백. 10fold test예측 test_preds append, oof_preds[va_idx] 저장. 평균 hit + OOF hit/dist 출력.

### §8 후처리 Weighted LGBM 잔차 (L461-588)
final_all=mean(test_preds). build_lgbm_features(seq_flat+물리ctx+base_pred+move+norms+cos). y_res=Y_train-oof_preds. sample_weight: hard(>0.01)1.5, boundary(0.01~0.013)2.0, near(0.008~0.01)1.3. LGBM(n600,lr0.02,leaves24,depth3,mcs120,sub0.8,col0.8,a0.3,l2.0,seed42) axis별. 5-fold OOF residual 검증 + 전체 full-fit test residual. RESIDUAL_CLIP±0.0035. alpha 0~0.30 스윕 OOF best 산출. 회고: 최종 alpha를 LB로 골라 Private 하락.

### §9 제출 (L589-604)
alpha 후보별 제출 전부 생성 + best_oof 제출 + 최종 alpha=0.14(pred_final=final_all+0.14*res_test) submission_FINAL_alpha_0p140.csv.
