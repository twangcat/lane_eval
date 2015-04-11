from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy
import subprocess

process = subprocess.Popen(
    ['pkg-config', '--libs', 'opencv'],
    stdout=subprocess.PIPE)
out, err = process.communicate()
libs = [lib for lib in str(out).split() if '.' in lib]
opencv_path = set(['/'.join(lib.split('/')[:-2]) for lib in libs]).pop()
print 'Your opencv path:', opencv_path

setup(
  cmdclass = {'build_ext': build_ext},
  ext_modules = [Extension(
    "group_rectangles_with_aux",
    language = "c++",
    sources = ["_group_rectangles_with_aux.pyx", "group_rectangles_with_aux.cpp"],
    include_dirs=[numpy.get_include(), opencv_path + '/include'],
    extra_link_args=libs)]
)
