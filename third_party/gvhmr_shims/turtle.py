# Shim for `from turtle import forward` in GVHMR's body_model.py.
# GVHMR only imports the name `forward` from the turtle stdlib module
# (presumably by accident — it's not used for graphics). On headless
# environments without tkinter, the real turtle module fails to import.
# This shim provides the name so the import succeeds.

def forward(*args, **kwargs):
    pass
