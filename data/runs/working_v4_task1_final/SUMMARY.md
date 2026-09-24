# Run summary — `working_v4_task1_final`

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
| max_conversation_seconds | `360` |
| seed | `42` |
| prompt_variant | `v4` |

## Headline

- Tasks: **1**
- Local reward: **1.000**
- Strict Tau2 reward: **unavailable (NL assertion not scored)**
- DB match: **true**
- Actions matched: **5/5**
- Total conversation time: **176.3 s**

## Per task

| task | trial | local reward | strict reward | DB match | actions | termination | duration (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1.0 | N/A | true | 5/5 | user_stop | 176.3 |

## Failure-mode checks

| task | check | passed | message |
| --- | --- | --- | --- |
| 1 | auth_loop | ✅ | no repeated authentication call; auth prompts=1 |
| 1 | tool_argument_integrity | ✅ | 6 calls passed recorded argument/error checks |
| 1 | write_protocol | ✅ | write calls=1, expected write categories=1 |
