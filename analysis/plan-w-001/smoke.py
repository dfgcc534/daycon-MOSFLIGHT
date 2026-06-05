"""plan-w-001 G_smoke — N_EACH=1 1-arch 2-epoch finite + shape check.

GOH30_reproduce.py 의 함수를 그대로 import. environment override 만 적용:
  N_EACH=1 GRU_ODE_EPOCHS=2 H_EPOCHS=2

Goal: 코드가 우리 환경 (L40S, cuda) 에서 forward/backward + train 1 step 까지 도는지 확인.
Full 30모델 학습 (~30~45min) 진입 전 fail-fast 게이트.
"""
import os
import sys
import numpy as np
import torch
import time

os.environ['N_EACH'] = '1'
os.environ['GRU_ODE_EPOCHS'] = '2'
os.environ['H_EPOCHS'] = '2'
os.environ['MODELS_DIR'] = './models_goh30_smoke'
os.environ['SUBMISSION_OUT'] = './analysis/plan-w-001/submission_smoke.csv'
os.environ['FROM_SCRATCH'] = 'True'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 모듈 경로: notes/winners/rank1-euijin42/repo/GOH30_reproduce.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                  'notes/winners/rank1-euijin42/repo'))

import GOH30_reproduce as G


def smoke():
    t0 = time.time()
    print(f'[smoke] device={G.DEVICE} N_EACH={G.N_EACH} GRU_ODE_EPOCHS={G.GRU_ODE_EPOCHS} H_EPOCHS={G.H_EPOCHS}')

    # cache build
    CACHE, STATS, train_paths, labels = G.build_cache(G.DATA_DIR)
    assert CACHE['seq'].shape == (50000, 11, 13), f"cache seq shape: {CACHE['seq'].shape}"
    assert CACHE['scal'].shape == (50000, 22), f"cache scal shape: {CACHE['scal'].shape}"
    assert CACHE['tgt'].shape == (50000, 3), f"cache tgt shape: {CACHE['tgt'].shape}"
    print(f'[smoke] cache OK ({time.time()-t0:.1f}s)')

    # train 1 seed × 1 arch (AttnGRU, 2ep)
    t1 = time.time()
    ema_g = G.train_cache_seed(0, G.AttnGRU, CACHE)
    print(f'[smoke] GRU seed0 trained ({(time.time()-t1):.1f}s)')
    # finite check
    for k, v in ema_g.items():
        if v.dtype.is_floating_point:
            assert torch.isfinite(v).all(), f'NaN in EMA key {k}'
    print('[smoke] GRU EMA finite OK')

    # train 1 seed × ODE (2ep)
    t2 = time.time()
    ema_o = G.train_cache_seed(0, G.ODEModel, CACHE)
    print(f'[smoke] ODE seed0 trained ({(time.time()-t2):.1f}s)')
    for k, v in ema_o.items():
        if v.dtype.is_floating_point:
            assert torch.isfinite(v).all(), f'NaN in ODE EMA key {k}'

    # train 1 seed × HyperPhysics (2ep)
    t3 = time.time()
    X_train = np.stack([G.load_sample(p) for p in train_paths]).astype(np.float32)
    Y_train = labels
    ema_h = G.train_h_seed(0, X_train, Y_train)
    print(f'[smoke] H seed0 trained ({(time.time()-t3):.1f}s)')
    for k, v in ema_h.items():
        if v.dtype.is_floating_point:
            assert torch.isfinite(v).all(), f'NaN in H EMA key {k}'
    print('[smoke] all 3 arches finite OK')

    # save weights so predict_all can find them
    torch.save({'model_state': ema_g}, f'{G.MODELS_DIR}/phaseG_full_0.pt')
    torch.save({'model_state': ema_o}, f'{G.MODELS_DIR}/phaseODE_full_0.pt')
    torch.save({'model_state': ema_h}, f'{G.MODELS_DIR}/phaseH_full_0.pt')

    # predict (3 model blend)
    t4 = time.time()
    sub = G.predict_all(G.DATA_DIR, STATS, train_paths)
    print(f'[smoke] predict ({time.time()-t4:.1f}s) shape={sub.shape}')
    assert sub.shape == (10000, 4), f'submission shape: {sub.shape}'
    assert list(sub.columns) == ['id', 'x', 'y', 'z'], f'columns: {sub.columns.tolist()}'
    assert sub[['x', 'y', 'z']].notna().all().all(), 'NaN in submission'
    sub.to_csv(os.environ['SUBMISSION_OUT'], index=False)
    print(f'[smoke] saved {os.environ["SUBMISSION_OUT"]}')

    print(f'[smoke] ALL PASS (total {(time.time()-t0)/60:.1f}min)')


if __name__ == '__main__':
    smoke()
