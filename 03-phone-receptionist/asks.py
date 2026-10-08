"""What the receptionist just asked the caller. One shared vocabulary for three users:

  - dialog.py records it in the call state ("what am I waiting for?"),
  - simulate.py picks the answer a synthetic caller gives,
  - the call record keeps it for inspection.

Strings, not an Enum: they go straight into the saved JSON and read well in a log.
"""

GREETING = "greeting"            # the call just started: "How can I help you today?"
REASON = "reason"                # "Could you tell me briefly what your call is about?"
NAME = "name"                    # "Could I take your name, please?"
SPELLING = "spelling"            # "Could you spell your name for me, letter by letter?"
NUMBER = "number"                # "What is the best number to call you back on?"
NUMBER_AGAIN = "number_again"    # asked a second time after the caller would not give one
REPEAT = "repeat"                # "Sorry, I didn't catch that."
CONFIRM = "confirm"              # the read-back: "So that's ... Is that right?"
CORRECTION = "correction"        # "What should I change: your name, your number, or what it's about?"
ANYTHING_ELSE = "anything_else"  # "Is there anything else I can help you with?"
NOTHING = "nothing"              # the call is over: nothing more is expected

ALL = (GREETING, REASON, NAME, SPELLING, NUMBER, NUMBER_AGAIN, REPEAT, CONFIRM, CORRECTION, ANYTHING_ELSE, NOTHING)
