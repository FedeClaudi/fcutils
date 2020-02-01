from setuptools import setup, find_namespace_packages

setup(
    name="behaviour",
    version="0.0.1",
    author_email="federicoclaudi@protonmail.com",
    description="bunch of utility functions to analyse behaviour data",
    packages=find_namespace_packages(exclude=()),
    include_package_data=True,
    url="https://github.com/BrancoLab/Behaviour",
    author="Federico Claudi",
    zip_safe=False,
)