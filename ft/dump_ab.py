import json
from automationbench.domains import get_combined_dataset
from automationbench.scripts.eval import _task_name
import automationbench.runner as R, automationbench.rubric as RB
ds = get_combined_dataset(["sales","marketing","operations","support","finance","hr"])
env = R.AutomationBenchEnv(dataset=ds, rubric=RB.create_rubric(), toolset="api")
tools = env._all_oai_tools
json.dump(tools, open("../ft/data/ab_tools.json","w"), indent=1)
prompts = {}
for row in ds:
    prompts[_task_name(row)] = row["prompt"]
json.dump(prompts, open("../ft/data/ab_prompts.json","w"))
print(len(prompts), [t.get("function",t).get("name") for t in tools])
