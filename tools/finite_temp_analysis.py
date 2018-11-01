#!/usr/bin/env python

import glob
import numpy
import matplotlib.pyplot as pl
import sys
from pauxy.trial_density_matrices.onebody import OneBody
from pauxy.qmc.calc import init_communicator, setup_calculation
from pauxy.qmc.thermal_afqmc import ThermalAFQMC
from pauxy.thermal_propagation.utils import get_propagator
from pauxy.analysis.thermal import analyse_energy
from pauxy.utils.io import to_json


files = glob.glob(sys.argv[1])
data = analyse_energy(files)
print (data.to_string(index=None))
# data.to_csv('data', header=True, index=True, sep='\t', mode='w')