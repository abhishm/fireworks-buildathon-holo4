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
   "n": 60,
   "pass": 0.5666666666666667,
   "lo": 0.43333333333333335,
   "hi": 0.6833333333333333,
   "score": 0.8555929195073931,
   "cost": 0.03707563752411672,
   "complete": true,
   "kimi_same": 0.4666666666666667
  },
  "ds_harness": {
   "n": 60,
   "pass": 0.5,
   "lo": 0.36666666666666664,
   "hi": 0.6333333333333333,
   "score": 0.8305883023345838,
   "cost": 0.06326065992370845,
   "complete": true,
   "kimi_same": 0.4666666666666667
  },
  "ds_harness_v2": {
   "n": 60,
   "pass": 0.5833333333333334,
   "lo": 0.4666666666666667,
   "hi": 0.7,
   "score": 0.8558111272256009,
   "cost": 0.0598358114,
   "complete": true,
   "kimi_same": 0.4666666666666667
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
  },
  "router": {
   "n": 60,
   "pass": 0.5833333333333334,
   "lo": 0.4666666666666667,
   "hi": 0.7,
   "score": 0.8526758584982269,
   "cost": 0.062378003013671655,
   "complete": true,
   "vs_kimi": [
    13,
    6,
    0.1670684814453125
   ]
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
   "sys_only": 11,
   "kimi_only": 5,
   "p": 0.210113525390625,
   "n": 60
  },
  "ds_harness": {
   "sys_only": 8,
   "kimi_only": 6,
   "p": 0.79052734375,
   "n": 60
  },
  "ds_harness_v2": {
   "sys_only": 12,
   "kimi_only": 5,
   "p": 0.143463134765625,
   "n": 60
  }
 },
 "router_ready": true,
 "router_mix": {
  "ds_harness_v2": 34,
  "ds_harness": 8,
  "nemo_ft": 1,
  "ds_base": 17
 },
 "oracle": [
  {
   "label": "harness v2",
   "solved": 35,
   "gain": 35
  },
  {
   "label": "+ DeepSeek (plain)",
   "solved": 40,
   "gain": 5
  },
  {
   "label": "+ harness v1",
   "solved": 41,
   "gain": 1
  },
  {
   "label": "+ Nemotron LoRA",
   "solved": 42,
   "gain": 1
  },
  {
   "label": "+ Qwen LoRA",
   "solved": 42,
   "gain": 0
  }
 ],
 "n_common": 60,
 "before_after": {
  "qwen_ft": {
   "fixed": 16,
   "broken": 1,
   "p": 0.000274658203125,
   "base": 0.06666666666666667,
   "ft": 0.31666666666666665,
   "n": 60
  },
  "nemo_ft": {
   "fixed": 13,
   "broken": 3,
   "p": 0.021270751953125,
   "base": 0.11666666666666667,
   "ft": 0.2833333333333333,
   "n": 60
  }
 }
};
