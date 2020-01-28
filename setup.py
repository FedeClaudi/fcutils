from setuptools import setup, find_namespace_packages

setup(
    name="fcutils",
    version="0.1.3.2",
    author_email="federicoclaudi@protonmail.com",
    description="bunch of utility functions",
    packages=find_namespace_packages(exclude=()),
    include_package_data=True,
    url="https://github.com/FedeClaudi/fcutils",
    author="Federico Claudi",
    zip_safe=False,
)