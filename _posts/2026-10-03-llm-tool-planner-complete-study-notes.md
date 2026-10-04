---
title: "Using an LLM as a Tool Planner: Complete Introductory Study Notes"
date: 2026-10-03 23:46:00 -0700
categories: [LLM, Interview Preparation]
tags: [tool-calling, planning, agents, idempotency, evaluation, system-design]
description: A practical guide to LLM tool planning, execution loops, dependencies, validation, retries, supervised fine-tuning, evaluation, and task completion.
math: false
---

Study date: October 3, 2026 (America/Los_Angeles)
Source: Our guided study conversation.
Scope: Complete introductory study; advanced RL training, reward design, and large tool catalogs were not covered in depth.

## 1. Core idea

An LLM tool planner decides which tool to call, what arguments to supply, and when to call it. The application executes the call and returns its result. The model then chooses the next action.

**The LLM proposes; the application validates and executes; tool results provide evidence.**

Generating a tool-call JSON object does not execute a function or retrieve information. A proposed action is not proof of success.

The planner can choose to call a tool, ask for clarification, or finish with an answer. It can answer directly when the supplied context is sufficient.

## 2. Tool definitions

A tool definition contains a name, a description of its purpose, and an argument schema. The schema describes how to call the tool; the description helps the model decide when to use it.

```json
{
  "name": "get_calendar_events",
  "description": "Read calendar events within a time range.",
  "parameters": {
    "type": "object",
    "properties": {
      "start_time": {"type": "string"},
      "end_time": {"type": "string"},
      "timezone": {"type": "string"}
    },
    "required": ["start_time", "end_time", "timezone"]
  }
}
```

Use tools with bounded responsibilities, such as searching contacts separately from creating events. Schema details depend on the API; this is an illustrative definition.

## 3. Planning, execution, and observation

At each step, the model receives the user request, constraints, available tool definitions, previous calls, and returned results. The application should also retain durable execution state.

Example request: “Find a free 30-minute slot tomorrow afternoon and schedule a meeting with Alice.”

1. Resolve tomorrow afternoon using the user's current date and timezone.
2. Resolve Alice to the intended contact.
3. Read the relevant availability.
4. Select a valid free slot.
5. Request event creation with grounded arguments.
6. Observe the creation result.
7. Report completion once success is verified.

```python
# Conceptual pseudocode, not a specific SDK implementation.
messages = [system_instructions, user_request]

for step in range(max_steps):
    response = llm.generate(messages=messages, tools=tool_definitions)
    messages.append(response)

    if not response.tool_calls:
        return response.text

    for call in response.tool_calls:
        validate_arguments(call)
        check_authorization(call)
        result = execute_tool(call.name, call.arguments)
        messages.append(tool_result(
            tool_call_id=call.id,
            content=result,
        ))

return "Execution limit reached; report completed and unresolved work."
```

A production implementation must return validation errors and execution failures as structured observations, preserve operation identity, and avoid treating any final text as proof of task success. The loop above illustrates the interaction rather than implementing every safeguard.

## 4. Planning strategies

- **Step-by-step:** Choose an action, observe its result, and decide again. Useful when later inputs are unknown.
- **Plan then execute:** Produce a multistep plan before execution. Useful for predictable workflows.
- **Hybrid:** Establish a high-level plan and revise it as results arrive.

The model can plan “resolve contact, find availability, book” upfront, but it must wait for actual results before supplying dependent arguments.

## 5. Dependencies and parallel execution

A call can start when its required inputs are available and its dependencies are satisfied.

- Searching for Alice and booking a meeting with the returned email are sequential.
- Looking up Seattle weather and Boston weather can run in parallel.
- Finding the next meeting and retrieving weather at its location are sequential because the location is initially unknown.
- Setting an alarm and checking weather can run in parallel when both calls have all required inputs and are otherwise safe to execute independently.

Parallel calls can reduce latency. Two independent one-second lookups take roughly one second in parallel instead of two sequentially, excluding overhead. Resource limits and service constraints still apply.

## 6. Validation, grounding, and ambiguity

Valid JSON does not guarantee a correct action. An invented email address can satisfy a string schema.

Different checks serve different purposes:

- **Schema validation:** Required fields and types are valid.
- **Grounding:** Arguments come from user input or relevant trusted results.
- **Task correctness:** The contact, date, and duration match the user's intent.
- **Authorization:** The action is within what the user authorized.

A useful design lets the model select a contact ID from actual search results, while application code resolves the email. The application must validate the selected ID and permissions.

If search returns Alice Chen and Alice Wang with no distinguishing context, ask which person the user means. Do not guess.

Treat retrieved content as data, not as authority to override application policies. Enforce permissions in application code.

## 7. Tool calls versus actual results

```json
{
  "name": "get_weather",
  "arguments": {"location": "Seattle"}
}
```

This object is a request. The application must execute it before the assistant can report retrieved weather.

For “What meetings are on my calendar tomorrow?”, call a calendar tool. Model weights are not a reliable, current record of a personal calendar. Resolve the time range and timezone, query, and summarize the returned events.

Distinguish:
- Successful query with no events: “No events were found on the calendar I checked for tomorrow.”
- Failed query: The event list is unknown; do not claim there are no meetings.
- Failed weather lookup: Retry within a bound, use another available source when appropriate, or explain the inability to retrieve the forecast.

## 8. Timeouts, retries, and idempotency

A timeout means the outcome may be unknown. A service may have created a meeting before its response was lost. An immediate unprotected retry can create a duplicate.

An idempotency key identifies one intended operation.

```python
create_calendar_event(
    attendee_id="contact_123",
    start_time="2026-10-05T14:00:00-07:00",
    idempotency_key="booking_abc123",
)
```

This timestamp is illustrative, not a date derived from the study conversation.

When the service supports idempotency, retries of the same operation reuse the same key and arguments. A new key can be treated as a new operation.

- The execution layer generates and durably preserves the key.
- The LLM should not be responsible for remembering or regenerating it.
- A genuinely new booking gets a new key.
- Behavior depends on the service's idempotency contract and retention window.
- If idempotency is unavailable, reconcile operation status or existing records before deciding whether to retry. A simple search may not eliminate every race condition.

## 9. Replanning and constraints

Replanning changes the next action in response to new evidence.

- Temporary calendar-read failure: bounded retry may be appropriate.
- Creation timeout: reconcile status or use the same idempotency key.
- Explicitly unavailable slot: find another valid slot.
- No valid slot within the requested range: ask whether to expand the range.

If the user requested tomorrow afternoon, the planner cannot silently book the following morning. Replanning must preserve constraints unless the user changes them.

## 10. Task state and stopping

Track which requested outcomes are pending, complete, failed, or unresolved.

A successful tool call completes only the subtask that it performed. After creating the requested meeting, report success and stop unless another requested task remains.

A success observation might contain:

```json
{
  "status": "success",
  "event_id": "evt_123",
  "start_time": "2026-10-05T14:00:00-07:00"
}
```

If the model never receives the result, it may attempt the action again. Return observations to the model and retain success in application state so it survives restarts or lost context.

Example: “Set an alarm for 7 AM and check tomorrow's weather in Seattle.”

1. Set the alarm using resolved time information.
2. Record the tool's success.
3. Retrieve the weather for the resolved date and location.
4. Report both outcomes once known.
5. Stop.

The two calls could also execute in parallel. If the weather call fails, preserve the alarm's completed state and report the partial outcome; do not create another alarm.

**Stopping rule:** Finish when all requested outcomes are verified, or communicate what remains blocked when further authorized progress is not possible.

## 11. Supervised fine-tuning

A training trajectory can contain:

```text
User: What's the weather in Seattle?

Assistant tool call:
get_weather({"location": "Seattle"})

Tool result:
{"temperature_c": 12, "condition": "rain"}

Assistant:
It's 12°C and raining in Seattle.
```

The values are illustrative. At runtime, observations come from the external tool.

In a typical supervised setup, loss applies to assistant output tokens, including tool-call tokens and final responses; user inputs and tool observations are context. The model learns actions conditioned on requests and results.

Include examples of:
- Correct tool choice and arguments.
- Direct answers when tools are unnecessary.
- Clarification when required information is missing.
- Ambiguous contacts and empty results.
- Tool failures and appropriate recovery.
- Multiple subtasks and correct stopping.

If every example contains a tool call, the model may learn unnecessary tool use. Rewriting a supplied sentence usually needs no external tool; reading a personal calendar does.

An instruction-following model with clear tool descriptions can be a starting point. Fine-tuning can target recurring failures when suitable examples are available. Detailed training objectives and RL were outside this session's scope.

## 12. Evaluation

Evaluate at multiple levels:
- Tool selection: Was the right tool chosen?
- Arguments: Were contact, date, duration, and other values correct?
- Final outcome: Did the actual resulting state satisfy the request?
- Authorization: Were actions permitted?
- Recovery: Were ambiguity, empty results, and failures handled correctly?
- Efficiency: How many calls, retries, tokens, seconds, and dollars were used?

Correct tool selection alone does not count as success. Booking the wrong date fails the task even when the API returns success.

Allow multiple valid plans. Independent lookups can occur in either order or in parallel, so exact sequence matching can be too strict.

Three calls may be more efficient than twelve redundant calls, but actual latency and cost must be measured. Several parallel calls may be faster than fewer sequential calls.

For a request requiring a free slot, a two-second planner that creates a conflict fails; a four-second planner that books a free slot succeeds. Optimize efficiency while preserving task requirements. A busy event makes a conflicting slot unavailable; calendar events explicitly marked free need different treatment.

## 13. Corrected concept checks

1. Can contact search and dependent event creation run in parallel? No; event creation needs the resolved contact.
2. Does schema-valid JSON guarantee correctness? No; values can be invented or refer to the wrong entity.
3. Two indistinguishable Alice matches? Ask which contact the user means.
4. Is a write timeout proof of failure? No; the operation may have succeeded.
5. New idempotency key for every retry? No; preserve the same key for the same operation.
6. Who owns the key? The application's execution layer.
7. Seattle and Boston weather lookups? They can run in parallel.
8. Next meeting and weather at its unknown location? Sequential.
9. Does generating tool JSON fetch weather? No; application execution is still required.
10. Can the assistant claim successful retrieval after an error? No.
11. Who supplies the observation? The external tool.
12. What can tool-only training examples encourage? Unnecessary tool calls.
13. Personal calendar from model memory? No; query the calendar.
14. Successful empty calendar query? Report no events found in the queried scope.
15. Correct calendar tool, wrong date? Task failure.
16. How distinguish successful plans of different lengths? Measure redundant calls, latency, and cost.
17. Faster booking that violates free-slot requirements? Not an improvement.
18. Slot explicitly unavailable? Look for another valid slot.
19. Only available slot is outside the user's range? Ask before changing the constraint.
20. Missing success observation? The planner may repeat the completed action.
21. Requested alarm successfully created? Confirm it and stop if that is the entire request.
22. Alarm set, weather still pending? Retrieve the weather.
23. When stop? After all requested outcomes are verified or remaining blockers are communicated.

## 14. Interview-ready answer

An LLM tool planner receives the user's goal, constraints, tool schemas, and execution history, then proposes structured tool calls. A deterministic execution layer validates arguments, checks permissions, manages operation state and idempotency, executes tools, and returns observations. Dependent calls wait for required results; independent calls can run in parallel. The planner clarifies ambiguity, replans after failures without violating constraints, and stops once the requested outcomes are verified. Evaluation covers final task success, argument correctness, recovery, authorization, latency, and cost.

## 15. Completion and review

The introductory study is complete. Completion does not mean every concept is fully consolidated.

Prioritize reviewing:
- A generated call is a proposal, not an executed action.
- Reuse the same idempotency key for retries of the same operation.
- The execution layer owns durable operational state.
- Independent calls can execute in parallel.
- Correctness comes before optimizing speed.
- Report verified success and stop; continue only for remaining requested work.

Optional advanced follow-up: detailed SFT loss construction, RL reward design, tool retrieval for large catalogs, and more complex planning architectures.

