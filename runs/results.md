Held-out test set: 60 tasks (router fit on 59 calib tasks)

| System | Pass rate | 95% bootstrap CI | Avg partial score | Real $/task |
|---|---|---|---|---|
| ds_base | 56.7% | [43.3%, 70.0%] | 85.6% | $0.0371 |
| ds_harness | 50.0% | [36.7%, 63.3%] | 83.1% | $0.0633 |
| ds_harness_v2 | 58.3% | [45.0%, 71.7%] | 85.6% | $0.0598 |
| qwen_ft | 31.7% | [20.0%, 43.3%] | 72.5% | $0.0963 |
| nemo_ft | 28.3% | [16.7%, 40.0%] | 70.8% | $0.1621 |
| kimi_k3 | 46.7% | [33.3%, 60.0%] | 83.5% | $0.1398 |
| router_knn | 55.0% | [41.7%, 68.3%] | 83.4% | $0.0618 |
| router_domain | 58.3% | [46.7%, 71.7%] | 86.2% | $0.0508 |
| best_single_on_calib | 50.0% | [36.7%, 63.3%] | 83.1% | $0.0633 |
| oracle_upper_bound | 70.0% | [58.3%, 81.7%] | 94.1% | $0.0398 |

Paired exact McNemar, router_knn vs X (router-only wins / X-only wins / p):

- vs ds_base: 1 / 2 / p=1.000
- vs ds_harness: 6 / 3 / p=0.508
- vs ds_harness_v2: 3 / 5 / p=0.727
- vs qwen_ft: 16 / 2 / p=0.001
- vs nemo_ft: 19 / 3 / p=0.001
- vs kimi_k3: 11 / 6 / p=0.332
- vs router_domain: 1 / 3 / p=0.625
- vs best_single_on_calib: 6 / 3 / p=0.508

router_knn selection mix on test: {'ds_harness_v2': 28, 'ds_harness': 10, 'ds_base': 21, 'nemo_ft': 1}
best single system on calib: ds_harness

Leave-one-out kNN router over 119 pooled calib+test tasks: 54.6%
- ds_base: 54.6% (router-only 10 / ds_base-only 10, p=1.000)
- ds_harness: 52.9% (router-only 8 / ds_harness-only 6, p=0.791)
- ds_harness_v2: 63.0% (router-only 3 / ds_harness_v2-only 13, p=0.021)
- qwen_ft: 28.6% (router-only 37 / qwen_ft-only 6, p=0.000)
- nemo_ft: 31.1% (router-only 36 / nemo_ft-only 8, p=0.000)
