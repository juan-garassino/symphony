
# setup.py
from setuptools import find_packages, setup

# list dependencies from file
with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content]

setup(name='symphony',
      description="package description",
      packages=find_packages(exclude=["tests"]), # NEW: find packages automatically
      install_requires=requirements) # NEW
