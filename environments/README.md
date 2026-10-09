# Tested environment reference

`macos-arm64-py310-cpu.txt` records all resolved packages from an isolated Python
3.10 macOS ARM64 environment installed on 2026-10-07. That environment passed
`pip check` and 83 fixture tests. Machine/runtime details are in `validation.json`.
It is a CPU validation reference, not a verified Linux CUDA environment or a
cross-platform lock with artifact hashes.

To recreate on compatible macOS/Python hardware, from a fresh environment:

```bash
python -m pip install -r environments/macos-arm64-py310-cpu.txt
python -m pip install --no-deps -e .
python -m pip check
python -m pytest -q
```

The project itself is deliberately excluded from the package list: install it
from the recorded source revision rather than resolving a similarly named PyPI
package. For Linux/CUDA, install an appropriate GPU PyTorch build and verify the
project in a fresh environment; archive that resolved environment with the actual
training runs. A future dependency resolution may differ from this reference.

The author's original shared `pt` environment also passed the fixture tests,
but its `pip check` reported OpenCV below this project's minimum and unrelated
Captum/SAM NumPy conflicts. It is not the clean reference environment.
