# Run summary — `v4_calibrated_batch`

## Config

| key | value |
| --- | --- |
| domain | `retail` |
| stt_impl | `parakeet` |
| tts_impl | `chatterbox` |
| agent_llm_impl | `minimax` |
| user_llm_impl | `minimax` |
| agent_model | `MiniMax-M2.7` |
| user_model | `MiniMax-M2.7` |
| minimax_api_base | `https://api.minimax.io/anthropic` |
| max_conversation_seconds | `480` |
| seed | `42` |
| prompt_variant | `v4` |

## Headline

- Tasks: **14**
- Mean reward: **0.000**
- Pass count (reward == 1.0): **0/14**
- Total wall time: **4459.7 s**

## Per task

| task | trial | reward | termination | duration (s) |
| --- | --- | --- | --- | --- |
| 0 | 1 | 0.0 | user_stop | 275.7 |
| 1 | 1 | 0.0 | agent_stop | 343.3 |
| 2 | 1 | 0.0 | agent_stop | 305.9 |
| 3 | 1 | 0.0 | user_stop | 201.0 |
| 3 | 1 | 0.0 | agent_stop | 266.6 |
| 4 | 1 | 0.0 | agent_stop | 232.1 |
| 5 | 1 | 0.0 | timeout | 360.2 |
| 5 | 1 | 0.0 | timeout | 360.2 |
| 5 | 1 | 0.0 | timeout | 360.6 |
| 6 | 1 | 0.0 | user_stop | 270.1 |
| 6 | 1 | 0.0 | timeout | 360.8 |
| 6 | 1 | 0.0 | user_stop | 282.5 |
| 7 | 1 | 0.0 | timeout | 360.4 |
| 6 | 1 | 0.0 | timeout | 480.3 |
