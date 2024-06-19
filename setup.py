import setuptools
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

exec((here / "mlogium" / "__init__.py").read_text(encoding="utf-8"))

setuptools.setup(
    name="mlogium",
    version=__version__,
    author="albi-c",
    description="high level mindustry logic language",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "pyperclip>=1.9.0"
    ],
    package_data={
        "mlogium": ["stdlib/*"]
    },
    url="https://github.com/albi-c/mlogium",
    project_urls={
        "Bug Tracker": "https://github.com/albi-c/mlogium/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(),
    python_requires=">=3.12",
    entry_points={
        "console_scripts": {
            "mlogium = mlogium.cli:main"
        }
    }
)