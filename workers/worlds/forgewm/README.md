# ForgeWM worker

See [shared setup, verification limits and customization](../MODEL_SETUP.md).

`manifest.json` pins source and every checkpoint repository. `requirements.lock`
is the resolved Python 3.12 inference environment. The adapter is implemented
against actual upstream functions; GPU inference and the complete container
build have not been validated in this workspace.
