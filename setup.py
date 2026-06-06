from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="depthforge",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Depth Map Generation using YOLOv and OpenVINO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/DepthForge",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "opencv-python==4.13.0.92",
        "openvino==2026.2.0",
        "torch==2.12.0",
        "torchvision==0.27.0",
        "numpy==1.24.3",
        "matplotlib==3.7.2",
        "pillow==10.0.1",
        "scipy==1.17.1",
        "scikit-image==0.26.0",
        "numpy-stl>=3.1.0",
        "transformers>=4.40.0",
    ],
    entry_points={
        "console_scripts": [
            "depthforge=src.depth_forge:main",
        ],
    },
)