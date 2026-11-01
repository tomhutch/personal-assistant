# Intent Usage

## Enable the deterministic parser

Set `USE_DETERMINISTIC_INTENT=true` in the environment. Initialize the pinned
OHF-Voice sentence corpus after cloning:

```bash
git submodule update --init --recursive
```

## Supported intents

The parser loads compatible English templates from the vendored OHF-Voice
corpus for:

### Lights

| Intent | Example |
| --- | --- |
| Turn on | “Turn on the lights in the kitchen.” |
| Turn off | “Turn off the lights in the kitchen.” |
| Set brightness | “Set the bedroom brightness to 50%.” |
| Set color | “Set the bedroom color to red.” |
| Set color temperature | “Set the bedroom color temperature to 2700 Kelvin.” |

### Media players

| Intent | Example |
| --- | --- |
| Next track | “Next track on the TV.” |
| Previous track | “Previous track on the TV.” |
| Pause | “Pause the TV.” |
| Resume | “Resume the TV.” |
| Mute | “Mute the TV.” |
| Unmute | “Unmute the TV.” |
| Search and play | “Play The Office.” |
| Set volume | “Set the TV volume to 90 percent.” |
| Change volume relatively | “Decrease the volume in the living room.” |

### Timers

| Intent | Example |
| --- | --- |
| Start | “Turn off the lights in 1 hour 5 minutes.” |
| Cancel | “Cancel the 1 hour timer.” |
| Cancel all | “Cancel all timers.” |
| Increase | “Add 5 minutes to the kitchen timer.” |
| Decrease | “Remove 1 hour from the kitchen timer.” |
| Pause | “Pause the timer in the kitchen.” |
| Resume | “Resume the timer in the kitchen.” |
| Check status | “1 hour timer status.” |

### To-do lists

| Intent | Example |
| --- | --- |
| Add item | “Add apples to my Trader Joe's list.” |
| Complete item | “Check off take out the trash from my chores list.” |
| Remove item | “Remove take out the trash from my chores list.” |
| Add shopping item | “Add apples to my shopping list.” |
| Complete shopping item | “Check off apples from my shopping list.” |
| Read a list | “What's on the shopping list?” |

The custom read-list phrase requires the word “list” to avoid matching unrelated
questions such as “what's on the shopping”. It reads actionable items from the
matched todo entity.

### Weather

| Intent | Example |
| --- | --- |
| Get weather | “What is the weather like in London?” or “What’s the weather?” |

The parser uses discovered entity names, areas, and floors as slots. Templates
requiring unavailable slots or unsupported entity domains are skipped.
Examples use entities and lists from the upstream corpus; phrase targets must
exist in, or be understood by, the configured Home Assistant instance.

## Handling flow

HassIL matches the transcript against the local, pinned sentence corpus.
Basic turn-on/off and numeric light-brightness commands are validated against
the discovered entities and sent directly to Home Assistant. Other supported
intents are passed to Home Assistant's `conversation.process` service, which
executes the built-in intent and returns speech for the assistant to play.

## Custom templates

Add project-specific HassIL `intents`, `expansion_rules`, or `lists` to
[`src/intents/custom_intents.yaml`](src/intents/custom_intents.yaml). These are
merged on top of the upstream templates. The file currently contains a
commented example; uncomment and adapt it to enable a custom sentence.