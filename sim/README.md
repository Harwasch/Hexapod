# Simulation

SPICE decks and their results. A simulation only closes a requirement if it is
reproducible, so commit the deck alongside the numbers it produced.

Cross-check against closed form wherever one exists, and read the
`observations` field — ngspice prints `singular matrix` and then writes
plausible numbers anyway.
