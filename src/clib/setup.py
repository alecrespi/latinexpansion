import numpy
from distutils.core import setup, Extension

def main():
    setup(name="latinexpansion",
          version="1.0.0",
          description="Latin Hypercube Sampling expansion algorithm",
          author="Alessandro C.",
          ext_modules=[
                Extension(
                  "latinexpansion", 
                  ["latinexpansionmodule.c"],
                  include_dirs=[numpy.get_include()]
                )])

if __name__ == "__main__":
    main()