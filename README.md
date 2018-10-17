# py-concord-fista

Requirement: python 3.5+, Cython

# usage

run ```python3 setup.py build_ext --inplace``` to create .c and .so files.
run ```mv src/cc_fista.c .```.
import format: ```from cc_fista import cc_fista```.
example usage can be seen in [hcp_cc.py](hcp_cc.py).