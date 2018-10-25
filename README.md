# py-concord-fista

Requirement: python 3.5+

# usage

If using Cython: rename cc_fista.py to cc_fista.pyx, run ```python3 setup.py build_ext --inplace``` to create .c and .so files. (Not much improvement on speed)

import using ```from cc_fista import cc_fista```, and example usage can be seen in [hcp_cc.py](hcp_cc.py). You need to create config.yaml that directs to your data, sample config is [config_sample.yaml](config_sample.yaml).