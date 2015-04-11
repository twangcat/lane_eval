from distutils.core import setup, Extension

# define the extension module
gr_module = Extension('group_rect_module', sources=['groupRectangles.cpp'])

# run the setup
setup(ext_modules=[gr_module])
