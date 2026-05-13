"""AWS-side fan-out for the hmr_world stage.

Local-mode runs never import anything from this package. The stage
dispatches into ``batch_runner`` only when ``hmr_world.runner == "batch"``;
the handler module is the container entrypoint in deployed builds.
"""
