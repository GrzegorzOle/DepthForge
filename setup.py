from pathlib import Path

from setuptools import setup

HERE = Path(__file__).parent

long_description = (HERE / "README.md").read_text(encoding="utf-8")


def read_requirements(filename: str) -> list[str]:
    """Parse a pip requirements file, skipping comments and blank lines.

    requirements.txt is the single source of truth for runtime dependencies —
    keep them here rather than duplicating the list into install_requires,
    where the two copies previously drifted out of sync.
    """
    lines = (HERE / filename).read_text(encoding="utf-8").splitlines()
    return [
        stripped
        for line in lines
        if (stripped := line.strip()) and not stripped.startswith(("#", "-"))
    ]


setup(
    name="depthforge",
    version="0.1.2",
    author="Grzegorz Oleksy",
    author_email="oleksy@cdest.eu",
    description=(
        "Depth map generation from 2D images for 3D tactile "
        "reproductions for the visually impaired"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GrzegorzOle/DepthForge",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    # src/ is a flat module directory, not a package — there is no __init__.py.
    package_dir={"": "src"},
    py_modules=["depth_forge", "depth_pipeline"],
    python_requires=">=3.10",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        # Only convert.py and the tools/ conversion helpers need these.
        "convert": ["torch>=2.0.0", "transformers>=4.40.0"],
    },
    entry_points={
        "console_scripts": [
            "depthforge=depth_forge:main",
            "depthforge-pipeline=depth_pipeline:main",
        ],
    },
)
