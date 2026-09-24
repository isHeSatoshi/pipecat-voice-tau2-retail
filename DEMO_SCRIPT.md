# Interviewer Demo Script

## 0:00–0:35 — Start with the system

“i’m going to show a cascaded retail voice agent built with pipecat and tau-bench. the customer is simulated with an llm, its speech is generated with chatterbox, transcribed with parakeet, segmented with silero vad, and passed to a customer-service agent that can call grounded retail tools. the final database state is evaluated, not just the words in the transcript.”

Show the README architecture and the list of models.

## 0:35–1:15 — Show the research and design choices

“i explored the pipecat and tau-bench repositories first, read the pipeline and runner code, and chose a cascaded design because it makes the audio, reasoning, tool arguments, tool result, and database state visible. tau-bench owns the retail environment and policy; pipecat owns the live voice loop.”

Point to:

- the two-direction virtual PCM bus;
- separate agent and user audio;
- grounded Tau-bench tool schemas;
- prompt v4 as a state machine;
- the runtime write guard.

## 1:15–2:15 — Show the failure that drove the work

Open the baseline report and one baseline trace.

“the baseline was not just a weak prompt. silero vad was not initialized correctly, tool schemas exposed empty arguments, repeated calls reached the environment, tts gaps looked like completed customer turns, and the traces did not contain enough evidence to debug the failure.”

Show:

- baseline authentication loops;
- empty or incorrect tool arguments;
- missing STT, LLM, TTS, and tool events;
- timeout behavior.

## 2:15–3:25 — Show the fixes

Explain the fixes in this order:

1. initialize and clean up the VAD correctly;
2. buffer speech until end of speech;
3. build tools from the real Tau-bench parameter schema;
4. serialize tool calls and reject malformed arguments;
5. ask for first name and last name letter by letter;
6. ask for zip code digit by digit after a failed lookup;
7. treat preferences like `battery → USB → AC` as strict priority;
8. discard an old proposal when the customer changes the request;
9. state exact write details and wait for an immediate yes;
10. ignore non-substantive “checking” phrases.

Open `viewer/app.py` and show the transcript, tool call, trace timeline, and separate audio tabs.

## 3:25–4:00 — Show results honestly

Open the report charts and explain:

“there are 19 recorded attempts across 10 unique task ids. nine reached local reward 1.0, six timed out, and only three passed all three behavior checks. strict Tau2 NL assertions were unavailable, so these are labeled local results.”

Point out:

- task 7: successful exchange with separate agent/customer audio;
- task 5: authentication failure and no successful write;
- task 12: local reward 1.0 but wrong payment method, caught by write protocol;
- retries and failed artifacts are intentionally retained.

## 4:00–4:30 — Show a real conversation

Open:

- [`docs/report/conversation_agent_task7.txt`](docs/report/conversation_agent_task7.txt);
- [`docs/report/conversation_user_task7.txt`](docs/report/conversation_user_task7.txt);
- [`docs/report/audio/conversation_agent_task7.wav`](docs/report/audio/conversation_agent_task7.wav);
- [`docs/report/audio/conversation_user_task7.wav`](docs/report/audio/conversation_user_task7.wav).

“the first name, last name, and zip code are spoken in recoverable form. the agent then resolves the low-brightness AC-adapter preference, proposes the exchange, waits for confirmation, executes the write, and reports the result.”

## 4:30–5:00 — Close with next steps

“the next work is multiple seeds, a text-only control, strict Tau2 judging, more payment and return tasks, regression tests for authentication and confirmation, latency/cost metrics, and a real acoustic transport. the repository keeps the raw traces and audio so every claim can be checked.”

Final line:

“the point of the project is not to hide the failures; it is to make the failures measurable and the improvements reproducible.”
