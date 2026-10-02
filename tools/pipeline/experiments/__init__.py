"""Measuring the pipeline rather than guessing at it: variant runs and a benchmark scene.

Nothing here is a stage and nothing here runs in the worker or the training image. The
modules are scripts with their pure parts importable, so a test can check the tables
they write without a network, an API or a GPU.
"""
