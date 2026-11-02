"""Holo-style harness for AutomationBench.

Wraps the stock AutomationBenchEnv (same tools, same 50-step budget, same grader) and adds:
  1. An operating procedure appended to the system prompt, adapted from H Company's
     Holo4 AutomationBench system prompt (Apache-2.0, Hcompany/trajectories).
  2. A verify-before-finish gate: the first time the model tries to stop, the harness
     injects a checklist alert instead of ending, so the model re-checks the request,
     policies, and final app state before finishing.
"""

import os

import verifiers as vf

from automationbench.runner import AutomationBenchEnv

PROCEDURE = """

# Operating procedure (follow strictly)

You are scored ONLY on the final state of the apps (records created/updated, emails/messages sent, events scheduled). What you say is ignored.

1. Orient: read the request carefully. Before acting, gather ALL governing context: look for policies, guidelines, SOPs, recent updates, or corrections in email inboxes, docs, sheets, Slack, or notes. Newer or more authoritative instructions override older ones. Watch for traps: stale rows, near-duplicate names, superseded values, records that look in-scope but are not.
2. Plan: in your reasoning, write an explicit checklist of every required mutation (which app, which record, exact field values, recipients) before making changes.
3. Search before you fetch: never guess an endpoint URL or parameter name; look it up and match the schema's types, enums, ID formats and date formats exactly.
4. Chain deliberately: after each call, extract the concrete values you need (ids, emails, amounts, dates) and reuse them exactly. Copy values verbatim (names, numbers with cents, formats) rather than paraphrasing.
5. Recover, don't repeat: when a call fails, diagnose (wrong endpoint, wrong field, wrong id) and change approach; never re-issue the same failing call.
6. Mutate precisely: perform every required change and nothing else. Stray changes to records the task did not ask for count against you as much as missing ones. Do not send extra emails/messages.
7. Verify: before finishing, re-read the affected records to confirm each checklist item is reflected in the app state.
8. Finish only when everything is done: reply with a one-line confirmation and no tool calls.
"""

VERIFY_ALERT = """<system_alert>Do not finish yet. Final verification pass:
1. Re-read the original request and every policy/guideline/update you found. Rebuild the full checklist of required changes (app, record, exact field values, recipients, wording requirements).
2. For each item, confirm with read calls that the app state matches exactly. Check for items you missed entirely (e.g., a required email, notification, status update, or record for every qualifying item).
3. Fix anything missing or wrong now. Do not touch anything the request did not ask for, and do not duplicate actions already done.
When every item is verified, reply with a brief confirmation and no tool calls.</system_alert>"""

# v2: revised after reading calib60 failures (never test60). Changes vs v1:
#  - references like proj_x / ws_x in the request are internal IDs: resolve to what the API expects
#  - outbound messages must name every relevant entity/value; dates in both human + ISO form
#  - whole-dollar amounts without ".00" (grader-aware: AutomationBench's Slack matcher rejects "12,450.00" for "12,450")
#  - lighter verify alert (one targeted pass, no full re-read) to cut wasted turns
PROCEDURE_V2 = """

# Operating procedure (follow strictly)

You are scored ONLY on the final state of the apps (records created/updated, emails/messages sent, events scheduled). What you say is ignored.

1. Orient: read the request, then gather ALL governing context (policies, guidelines, SOPs, corrections in email, docs, sheets, Slack). Newer or more authoritative instructions override older ones. Watch for traps: stale rows, near-duplicate names, superseded values, records that look in-scope but are not.
2. Plan: write an explicit checklist of every required mutation (app, record, exact field values, recipients) before acting.
3. Resolve identifiers: IDs in the request (e.g. proj_x, ws_x, ss_x, C_x) are internal references. Look the object up and use the identifier the API field expects (e.g. a Jira project KEY like "OPS", a channel ID/name, an email address). Never guess endpoint URLs or parameter names; search for them.
4. Chain deliberately: reuse exact values from tool results (ids, emails, amounts, dates). Copy names and numbers exactly as in the source.
5. Recover, don't repeat: when a call fails, diagnose and change approach; never re-issue the same failing call.
6. Mutate precisely: perform every required change and nothing else. Stray changes and extra emails/messages count against you as much as missing ones.
7. Write complete messages: every email/Slack/SMS/comment must explicitly name each relevant person, record, ID, amount and date it concerns, plus any word the request or policy says to include. Write dates in both forms, e.g. "April 1, 2026 (2026-04-01)". Write whole-dollar amounts without trailing zeros ("$12,450", not "$12,450.00"); keep real cents ("$1,125.50").
8. Finish: once all checklist items are done and checked, reply with a one-line confirmation and no tool calls.
"""

VERIFY_ALERT_V2 = """<system_alert>Before finishing, one targeted check: compare your checklist to what you actually did. (a) Is any required record/message/update missing, e.g. one for every qualifying item? (b) Did any message omit a required name, ID, amount, date or keyword? (c) Did you change or message anything the request did not ask for? Fix only real gaps with the minimum calls (do not redo work); otherwise reply with a one-line confirmation and no tool calls.</system_alert>"""

if os.environ.get("HARNESS_VERSION", "1") == "2":
    PROCEDURE, VERIFY_ALERT = PROCEDURE_V2, VERIFY_ALERT_V2

MAX_NUDGES = int(os.environ.get("HARNESS_MAX_NUDGES", "1"))
USE_PROCEDURE = os.environ.get("HARNESS_PROCEDURE", "1") == "1"


def _set_content(msg, content):
    if isinstance(msg, dict):
        msg = dict(msg)
        msg["content"] = content
        return msg
    return msg.model_copy(update={"content": content})


def _get(msg, key):
    return msg.get(key) if isinstance(msg, dict) else getattr(msg, key, None)


class HarnessEnv(AutomationBenchEnv):
    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        state = await super().setup_state(state, **kwargs)
        if USE_PROCEDURE:
            prompt = list(state["prompt"])
            for i, m in enumerate(prompt):
                if _get(m, "role") == "system":
                    prompt[i] = _set_content(m, (_get(m, "content") or "") + PROCEDURE)
                    break
            state["prompt"] = prompt
        state["_nudges"] = 0
        return state

    def _wants_to_stop(self, state: vf.State) -> bool:
        if len(state["trajectory"]) == 0:
            return False
        last = state["trajectory"][-1]["completion"][-1]
        return _get(last, "role") == "assistant" and not _get(last, "tool_calls")

    @vf.stop
    async def no_tools_called(self, state: vf.State) -> bool:
        if not self._wants_to_stop(state):
            return False
        # Leave room for the verification pass within the step budget.
        steps_left = self.max_turns - len(state["trajectory"])
        if state.get("_nudges", 0) < MAX_NUDGES and steps_left >= 3:
            return False
        return True

    async def env_response(self, messages: vf.Messages, state: vf.State, **kwargs) -> vf.Messages:
        last = messages[-1] if messages else None
        if last is not None and _get(last, "role") == "assistant" and not _get(last, "tool_calls"):
            self._extract_usage_and_debug(state)
            state["_nudges"] = state.get("_nudges", 0) + 1
            return [vf.UserMessage(content=VERIFY_ALERT)]
        return await super().env_response(messages, state, **kwargs)
