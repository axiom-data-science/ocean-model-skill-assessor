from setuptools import setup

install_requires = [
    "matplotlib",
    "pandas"
    "xarray",
]

setup(
    name="skill_assessor",
    description="Skill Assess Ocean Models",
    license="MIT",
    packages=["skill_assessor"],
    long_description=open("README.md").read(),
    python_requires=">=3.8",
    tests_require=["pytest"],
    zip_safe=True
)
