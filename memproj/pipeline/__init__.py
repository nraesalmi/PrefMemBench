"""The fixed recommender and the runner that writes predictions.jsonl.

Everything here is identical across memory methods: only `build_context()`
changes between experiments (README, Method interface). The pipeline half lives in
this package; the memory half supplies the methods it calls.
"""
