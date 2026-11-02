Held-out test set: 30 tasks (router fit on 30 calib tasks)

| System | Pass rate | 95% bootstrap CI | Avg partial score | Real $/task |
|---|---|---|---|---|
| ds | 66.7% | [50.0%, 83.3%] | 82.5% | $0.0289 |
| glm | 56.7% | [40.0%, 73.3%] | 82.1% | $0.0080 |
| nemoultra | 10.0% | [0.0%, 20.0%] | 33.4% | $0.0848 |
| router_knn | 70.0% | [53.3%, 86.7%] | 87.1% | $0.0156 |
| router_domain | 70.0% | [53.3%, 86.7%] | 86.7% | $0.0163 |
| best_single_on_calib | 66.7% | [50.0%, 83.3%] | 82.5% | $0.0289 |
| oracle_upper_bound | 73.3% | [56.7%, 86.7%] | 89.1% | $0.0130 |

Paired exact McNemar, router_knn vs X (router-only wins / X-only wins / p):

- vs ds: 2 / 1 / p=1.000
- vs glm: 4 / 0 / p=0.125
- vs nemoultra: 18 / 0 / p=0.000
- vs router_domain: 1 / 1 / p=1.000
- vs best_single_on_calib: 2 / 1 / p=1.000

router_knn selection mix on test: {'glm': 18, 'ds': 12}
best single system on calib: ds

Leave-one-out kNN router over 30 pooled calib+test tasks: 63.3%
- ds: 66.7% (router-only 2 / ds-only 3, p=1.000)
- glm: 56.7% (router-only 2 / glm-only 0, p=0.500)
- nemoultra: 10.0% (router-only 16 / nemoultra-only 0, p=0.000)
