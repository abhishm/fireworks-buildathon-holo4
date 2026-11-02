window.RESULTS = {
 "stats": {
  "kimi_k3": {
   "n": 60,
   "pass": 0.4666666666666667,
   "lo": 0.35,
   "hi": 0.6,
   "score": 0.8349227970309205,
   "cost": 0.13975778,
   "complete": true,
   "kimi_same": 0.4666666666666667
  },
  "ds_base": {
   "n": 23,
   "pass": 0.391304347826087,
   "lo": 0.21739130434782608,
   "hi": 0.6086956521739131,
   "score": 0.8250518646113612,
   "cost": 0.021884678900869564,
   "complete": false,
   "kimi_same": 0.391304347826087
  },
  "ds_harness": {
   "n": 11,
   "pass": 0.45454545454545453,
   "lo": 0.18181818181818182,
   "hi": 0.7272727272727273,
   "score": 0.8858851674641148,
   "cost": 0.03581039840454545,
   "complete": false,
   "kimi_same": 0.09090909090909091
  },
  "qwen_ft": {
   "n": 60,
   "pass": 0.31666666666666665,
   "lo": 0.2,
   "hi": 0.43333333333333335,
   "score": 0.7247096409925358,
   "cost": 0.09633695950000001,
   "complete": true,
   "kimi_same": 0.4666666666666667
  },
  "nemo_ft": {
   "n": 60,
   "pass": 0.2833333333333333,
   "lo": 0.16666666666666666,
   "hi": 0.4,
   "score": 0.7082629626708574,
   "cost": 0.1620734005,
   "complete": true,
   "kimi_same": 0.4666666666666667
  }
 },
 "paired_vs_kimi": {
  "qwen_ft": {
   "sys_only": 6,
   "kimi_only": 15,
   "p": 0.0783538818359375,
   "n": 60
  },
  "nemo_ft": {
   "sys_only": 5,
   "kimi_only": 16,
   "p": 0.02660369873046875,
   "n": 60
  },
  "ds_base": {
   "sys_only": 2,
   "kimi_only": 2,
   "p": 1.0,
   "n": 23
  },
  "ds_harness": {
   "sys_only": 4,
   "kimi_only": 0,
   "p": 0.125,
   "n": 11
  }
 },
 "router_ready": false,
 "router_mix": {},
 "oracle": {
  "Nemotron LoRA": 17,
  "Qwen LoRA": 19,
  "either LoRA (oracle)": 23,
  "Kimi K3": 28,
  "any of the three (oracle)": 36
 },
 "n_common": 60
};
