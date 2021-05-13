from setuptools import setup, find_namespace_packages

requirements = [
    "numpy",
    "opencv-python",
    "pandas",
    "matplotlib",
    "seaborn",
    "scipy",
    "pyyaml",
    "statsmodels",
    "loguru",
]

setup(
    name="fcutils",
    version="1.1.2.5",
    author_email="federicoclaudi@protonmail.com",
    description="bunch of utility functions",
    packages=find_namespace_packages(exclude=()),
    include_package_data=True,
    url="https://github.com/FedeClaudi/fcutils",
    author="Federico Claudi",
    zip_safe=False,
    install_requires=requirements,
)
