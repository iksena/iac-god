"""Single-agent harness baseline for IaCGOD.

Runs the same benchmark CSVs through an off-the-shelf coding harness
(Claude Code) instead of the LangGraph multi-agent pipeline, so that the
multi-agent architecture can be compared against a strong general-purpose
baseline holding the model, the validators and the scoring constant.
"""
