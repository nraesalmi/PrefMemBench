"""The online half of the memory side: the extraction loop.

`extract.py` replaces gold replay's mapping for the SOURCE of tool calls.
The tools, the store and the resolver are unchanged -- that separation is
what lets extraction be scored on its own.
"""
