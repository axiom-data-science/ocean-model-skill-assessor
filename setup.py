from setuptools import setup


setup(
    use_scm_version={
        "write_to": "ocean_model_skill_assessor/_version.py",
        "write_to_template": '__version__ = "{version}"',
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
    },
    entry_points={"console_scripts": ["omsa=ocean_model_skill_assessor.CLI:main"]},
)
