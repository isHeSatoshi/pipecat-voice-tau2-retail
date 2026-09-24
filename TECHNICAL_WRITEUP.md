# Technical write-up

## Goal

My goal was to build a small evaluation framework for ten retail tasks in the Tau-bench retail domain. The system under test is a voice agent, not a text-only agent. A simulated customer speaks to a retail agent, both sides hear each other through a Pipecat voice pipeline, and Tau-bench supplies the task, tools, policy, database, and evaluation.

Tau-bench also has its own voice framework. I used Pipecat anyway because the assignment specifically asks for a Pipecat voice agent. Pipecat handles the conversation; Tau-bench handles the retail world.

## Approach

I started by reading the Pipecat and Tau-bench repositories. I wanted to understand the runner, audio flow, tool bridge, task loading, and evaluator before writing a prompt.

The main design decision was to make every stage visible. A run records the customer and agent audio separately, the transcript, STT and VAD events, LLM calls, tool arguments and results, errors, termination, and reward information. I also built a Streamlit viewer so I could inspect an individual run instead of looking at one final number.

I started with dummy services, moved to the real voice stack, and kept the failed runs. The failures were useful because they showed where the problem actually lived.

## How I built it

The customer and agent run as two Pipecat pipelines. Audio moves through an in-memory PCM bus in both directions.

- MiniMax M2.7 generates the simulated customer and drives the retail agent.
- Chatterbox generates speech for each side.
- Parakeet performs speech-to-text.
- Silero VAD marks when speech starts and stops.
- Tau-bench tools provide authentication, order lookup, product details, exchanges, returns, and payments.
- The runner records the trajectory and evaluates the resulting database state.

The agent is required to ground ids and prices in tool results, state the exact write, wait for an immediate confirmation, execute one write, and report the real result.

## Challenges and how I resolved them

The first runs failed for several different reasons.

**VAD and STT were too eager.** Small audio frames were being treated as separate turns, and names and zip codes were being split. I fixed the VAD sample-rate setup and changed STT to collect an utterance before transcribing it at end-of-speech or after a safe silence gap.

**TTS pauses looked like new turns.** The agent response was being synthesized in pieces, so sentence pauses confused the conversation. I made the response continuous and added a short `checking..just a sec` acknowledgement before tool calls. It gives the customer useful feedback during a database lookup, is rate-limited, and is not added to the scored context.

**The tool schemas were not reliable.** The model received empty or incorrect argument definitions. I built the schemas from Tau-bench’s actual tool parameters and added validation for missing and unknown arguments.

**The agent could take unsafe shortcuts.** It sometimes claimed success, repeated a call, or chose the wrong payment method. I added serialized tool execution, duplicate-call protection, exact write confirmation, and offline checks for authentication loops, argument integrity, and write protocol.

**ASR errors changed the meaning of the task.** Parakeet did not only add harmless typos. It could turn `Kovacs` into something like `Kovax`, distort a zip code, or make `desk lamp` sound like a different product. Those errors flowed into authentication, product lookup, and tool arguments, so the agent was solving a different task even when its reasoning looked reasonable. I treated speech recognition as part of task state, not as a clean input layer.

**Authentication was fragile in speech.** STT could turn a first or last name into a plausible but wrong spelling. The agent now asks for the first name one letter at a time, the last name one letter at a time, and the zip code digit by digit before retrying the lookup. No customer identity or product id is hardcoded.

**Audio and transcripts did not always line up.** New runs save exact per-turn agent and customer clips. Older artifacts are preserved, but unsafe legacy per-message links are disabled instead of guessing which voice belonged to a line.

## What the final run contains

The `v4_calibrated_batch` contains 19 recorded attempts across 10 unique task ids. It is a development and test history, not a clean ten-row leaderboard.

- 9/19 attempts reached local reward `1.0`.
- 9/19 matched the expected database state.
- 6/19 timed out.
- 3/19 passed all three behavior checks.
- Strict Tau2 NL assertions were unavailable because the configured judge was not available.

The selected test tasks show why the checks matter. Task 5 failed authentication and never completed the write. Task 9 matched the database and actions but still had a failed authentication call. Task 12 matched the database but used PayPal instead of the original payment method, so the write-protocol check failed.

The release includes the source, tests, viewer, raw trajectories, traces, audio, five charts, CSV tables, provenance, and the report at [`reports/v4_calibrated_batch/report.md`](reports/v4_calibrated_batch/report.md).

## Approaches and trade-offs

I considered prompt-only fixes first because they were cheap, but they could not repair VAD, schemas, audio routing, or unsafe writes. I therefore combined prompt changes with runtime guards.

I used separate speech, language, and voice stages instead of an audio-native model because separate stages make failures easier to locate. The trade-off is added latency and more opportunities for speech recognition to damage names and identifiers.

I used an in-memory Pipecat audio path instead of a real microphone transport so the experiment would be repeatable and easy to inspect. It does not model echo cancellation, packet loss, or real-world acoustic conditions.

I used the same MiniMax model for the agent and simulated customer to keep the experiment controlled. That reduces cost, but it can also create shared failure modes. The local evaluator is deterministic and useful for debugging, but it is not a replacement for the strict Tau2 NL judge.

## Limitations and what I would improve

These results are single-trial and mixed-provenance. The same model plays both sides, the strict NL judge was unavailable, and the virtual audio path is not a deployment test.

Next I would run paired seeds, add a text-only control with the same tools and policy, enable the strict Tau2 judge, add more return/payment/escalation tasks, measure latency and cost, and validate the system on a real acoustic transport.

The main lesson is simple: a reliable voice agent needs working audio, correct turn boundaries, grounded tools, safe writes, and evaluation that checks the actual conversation—not just a confident final answer.
