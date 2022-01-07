from setuptools import setup


setup(
    use_scm_version={
        "write_to": "ragmac_xdem/_version.py",
        "write_to_template": '__version__ = "{version}"',
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
    },
    install_requires=[
        'geoutils @ git+https://github.com/GlacioHack/GeoUtils.git@main#egg=GeoUtils',
        'xdem @ git+https://github.com/GlacioHack/xdem.git@main#egg=xdem',
    ]
)
