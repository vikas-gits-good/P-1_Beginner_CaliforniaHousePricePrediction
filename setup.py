from setuptools import find_packages, setup
from typing import List

Run_text = "-e ."


def get_packages(File_path: str = None) -> List[str]:
    requirements = []

    with open(file=File_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if Run_text in requirements:
            requirements.remove(Run_text)
    return requirements


with open("README.md", "r", encoding="UTF-8") as f:
    long_desc = f.read()

__version__ = "0.1.0"
Repo_name = "P-1_Beginner_CaliforniaHousePricePrediction"
Author_name = "vikas-gits-good"
Author_email = "vikas.c.conappa@protonmail.com"


setup(
    name="California-House-Price-Prediction",
    version=__version__,
    author=Author_name,
    author_email=Author_email,
    description="ML app to predict house price in California",
    long_description=long_desc,
    url=f"https://github.com/{Author_name}/{Repo_name}",
    package_dir={"": "src"},
    packages=find_packages(),
    requires=get_packages(File_path="requirements.txt"),
)
