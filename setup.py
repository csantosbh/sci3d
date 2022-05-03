from setuptools import setup

setup(
    name='sci3d',
    version='0.0.1',
    packages=['sci3d', 'sci3d.api', 'sci3d.plottypes'],
    url='https://github.com/csantosbh/sci3d',
    license='GPL-3.0',
    author='Claudio Fernandes',
    author_email='claudiosf1@hotmail.com',
    description='Lightweight interactive Scientific Visualization library',
    install_requires=[
        'nanogui_sci3d @ git+https://github.com/csantosbh/nanogui@features/sci3d-requirements#egg=nanogui_sci3d',
        'numpy',
        'icecream',
    ],
    package_data={'sci3d': ['plottypes/shaders/*.glsl']},
    include_package_data=True,
)
