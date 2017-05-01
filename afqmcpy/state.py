import afqmcpy.hubbard as hubbard
import numpy as np
import random
import json

class State:

    def __init__(self, model, qmc_opts):

        # Generic method option
        self.method = qmc_opts['method']
        self.nwalkers = qmc_opts['nwalkers']
        self.dt = qmc_opts['dt']
        self.nsteps = qmc_opts['nsteps']
        self.nmeasure = qmc_opts['nmeasure']
        self.temp = qmc_opts['temperature']
        if model['name'] == 'Hubbard':
            # sytem packages all generic information + model specific information.
            self.system = hubbard.Hubbard(model)
            self.gamma = np.arccosh(np.exp(0.5*self.dt*self.system.U))
            self.auxf = np.array([[np.exp(self.gamma), np.exp(-self.gamma)],
                                  [np.exp(-self.gamma), np.exp(self.gamma)]])
            # self.auxf = self.auxf * np.exp(-0.5*dt*self.system.U*self.system.ne)
            # Constant energy factor emerging from HS transformation.
            self.cfac = 0.5*self.system.U*self.system.ne
            if self.method ==  'CPMC':
                self.projectors = hubbard.Projectors(self.system, self.dt)

        random.seed(qmc_opts['rng_seed'])
        # Handy to keep original dicts so they can be printed at run time.
        self.model = model
        self.qmc_opts = qmc_opts

    def write_json(self):

        print ("# Input options: ")
        print (json.dumps({'model': self.model, 'qmc_options': self.qmc_opts}, indent=4))
        print ("# End of input options ")
