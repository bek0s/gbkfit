
import os
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeExtensionBuild(build_ext):

    def run(self):

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):

        ext_dir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))

        debug_flag = f'{ext.name}_DEBUG'
        build_type = 'Debug' if debug_flag in os.environ else 'Release'

        cmake_args = [
            f'-DCMAKE_BUILD_TYPE={build_type}',
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}',
            f'-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={ext_dir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}'
        ]

        build_args = []

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

        print()


setup(
    name='gbkfit',
    description='description',
    long_description='long description',
    author='Georgios Bekiaris',
    author_email='gbkfit@gmail.com',
    url='https://github.com/bek0s/gbkfit',
    license='BSD (3-clause)',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Astronomy'
    ],
    packages=find_packages('src'),
    package_dir={'': 'src'},
    python_requires='>=3.9',
    install_requires=[
        'astropy',
        'matplotlib',
        'numpy',
        'pandas',
        'ruamel.yaml',
        'scikit-image',
        'scipy'
    ],
    entry_points={
        'console_scripts': [
            'gbkfit-cli = gbkfit.apps.cli:main'
        ],
        'gui_scripts': [
            'gbkfit-gui = gbkfit.apps.gui:main'
        ]
    },
    ext_package='gbkfit/driver/native',
    ext_modules=[CMakeExtension('libgbkfit', 'src/gbkfit/driver/native')],
    cmdclass={'build_ext': CMakeExtensionBuild}
)
