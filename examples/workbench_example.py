"""Run with Python, or with %gui qt and %run in IPython/Jupyter."""

import subpx
from subpx._workbench_data import demo_image


if __name__ == "__main__":
    image = demo_image()
    window = subpx.launch_workbench(image, name="Synthetic staggered contact holes")
