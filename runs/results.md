Held-out test set: 60 tasks (router fit on 60 calib tasks)

| System | Pass rate | 95% bootstrap CI | Avg partial score | Real $/task |
|---|---|---|---|---|
| ds_base | 56.7% | [43.3%, 70.0%] | 85.6% | $0.0371 |
| ds_harness | 50.0% | [36.7%, 63.3%] | 83.1% | $0.0633 |
| ds_harness_v2 | 58.3% | [45.0%, 71.7%] | 85.6% | $0.0598 |
| qwen_ft | 31.7% | [20.0%, 43.3%] | 72.5% | $0.0963 |
| nemo_ft | 28.3% | [16.7%, 40.0%] | 70.8% | $0.1621 |
| kimi_k3 | 46.7% | [33.3%, 60.0%] | 83.5% | $0.1398 |
| router_knn | 58.3% | [45.0%, 71.7%] | 85.3% | $0.0624 |
| router_domain | 58.3% | [46.7%, 71.7%] | 86.2% | $0.0508 |
| best_single_on_calib | 58.3% | [45.0%, 71.7%] | 85.6% | $0.0598 |
| oracle_upper_bound | 70.0% | [58.3%, 81.7%] | 94.1% | $0.0398 |

Paired exact McNemar, router_knn vs X (router-only wins / X-only wins / p):

- vs ds_base: 3 / 2 / p=1.000
- vs ds_harness: 8 / 3 / p=0.227
- vs ds_harness_v2: 3 / 3 / p=1.000
- vs qwen_ft: 18 / 2 / p=0.000
- vs nemo_ft: 21 / 3 / p=0.000
- vs kimi_k3: 13 / 6 / p=0.167
- vs router_domain: 2 / 2 / p=1.000
- vs best_single_on_calib: 3 / 3 / p=1.000

router_knn selection mix on test: {'ds_harness_v2': 34, 'ds_harness': 8, 'ds_base': 17, 'nemo_ft': 1}
best single system on calib: ds_harness_v2

Leave-one-out kNN router over 120 pooled calib+test tasks: 55.0%
- ds_base: 54.2% (router-only 12 / ds_base-only 11, p=1.000)
- ds_harness: 52.5% (router-only 8 / ds_harness-only 5, p=0.581)
- ds_harness_v2: 62.5% (router-only 2 / ds_harness_v2-only 11, p=0.022)
- qwen_ft: 28.3% (router-only 37 / qwen_ft-only 5, p=0.000)
- nemo_ft: 30.8% (router-only 37 / nemo_ft-only 8, p=0.000)
