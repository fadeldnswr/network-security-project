'''
This file is used for packaging and distributing
Python projects. It's used by setuptools to define
configuration of this project.
'''

from setuptools import find_packages, setup
from typing import List

# Create requirements installation function
def get_requirements() -> List[str]:
  '''
  The function will return a
  list of requirements
  '''
  requirements_list:List[str] = []
  try:
    with open("requirements.txt", "r") as file:
      # Read lines from the file
      lines = file.readlines()
      
      # Process each line
      for line in lines:
        requirements = line.strip()
        # Ignore the empty lines and -e.
        if requirements and requirements != "-e .":
          requirements_list.append(requirements)
  except FileNotFoundError:
    print("requirements.txt file not found")
  return requirements_list

# Setup the project
setup(
  name="Network Security Project",
  version="0.0.1",
  author="Fadel Achmad Daniswara",
  author_email="fadelachmad04@gmail.com",
  packages=find_packages(),
  install_requires=get_requirements()
)