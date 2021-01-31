from setuptools import setup, find_namespace_packages

requirements = [
    "numpy",
    "opencv-python",
    "nptdms",
    "configparser",
    "pandas",
    "tqdm",
    "matplotlib",
    "seaborn",
    "scipy",
    "vtk",
    "pyyaml",
    "statsmodels",
    "requests",
    "pyexcel",
    "pyexcel-xlsx",
    "pyjson",
]

setup(
    name="fcutils",
    version="1.1",
    author_email="federicoclaudi@protonmail.com",
    description="bunch of utility functions",
    packages=find_namespace_packages(exclude=()),
    include_package_data=True,
    url="https://github.com/FedeClaudi/fcutils",
    author="Federico Claudi",
    zip_safe=False,
    install_requires=requirements,
)
