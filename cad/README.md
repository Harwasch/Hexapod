# CAD

build123d models. Each file defines the geometry as code so a changed number is
a re-render, not a redraw.

Export STEP for the KiCad 3D view and STL for meshing. Link each model back to
the requirement it realises with a `File` relation in `requirements/`.
