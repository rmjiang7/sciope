# Copyright 2019 Prashant Singh, Fredrik Wrede and Andreas Hellander
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Sequential Monte-Carlo Approximate Bayesian Computation (SMC-ABC)
"""

# Imports
from sciope.inference.abc_inference import ABC
from sciope.inference.inference_base import InferenceBase
from sciope.core import core
from sciope.utilities.distancefunctions import euclidean as euc
from sciope.utilities.summarystats import burstiness as bs
from sciope.utilities.housekeeping import sciope_logger as ml
from sciope.utilities.epsadaptations import eps_dividing as epsd
import numpy as np
import dask
from dask.distributed import futures_of, as_completed, wait


# Class definition: Bandits-ABC rejection sampling
class SMCABC(InferenceBase):
    def __init__(self, data, sim, prior_functions, epsilons=1, summaries_function=bs.Burstiness(),
                 distance_function=euc.EuclideanDistance(), summaries_divisor=None,
                 use_logger=False, perturbation_kernel, eps_adaptation_function=epsd.eps_halving):
        self.name = 'SMC-ABC'
        super(SMCABC, self).__init__(self.name, data, sim, use_logger)

        self.prior_functions = prior_functions
        self.summaries_function = summaries_function
        self.epsilons = epsilons
        self.distance_function = distance_function
        self.summaries_divisor = summaries_divisor
        self.perturbation_kernel = perturbation_kernel
        self.t = None

        if type(self.epsilons) is list:
            self.t = len(self.epsilons)

        if type(self.prior_functions) is list and self.t is None:
            self.t = len(self.prior_functions)

        # At this point we do not know the number of populations we will have,
        # as the user is free to specify 't' epsilons or a single initial epsilon,
        # and Sciope can automatically assign the decrements for subsequent epsilons.
        # Ditto for priors - one can choose to have the same single prior type for all populations,
        # so, we do not know 't' during the initialization.
        #
        # * The method 'setup_population_attributes' should be used later to initialize 't' and the weights *
        self.weights = None
        self.num_accepted = None
        if self.use_logger:
            self.logger = ml.SciopeLogger().get_logger()
            self.logger.info("Sequential Monte-Carlo Approximate Bayesian Computation initialized")


    def setup_population_attributes(self, t, num_accepted):
        self.t = t
        if type(num_accepted) is list:
            assert len(num_accepted) == t, "The value of t does not match the length of list of desired number of " \
                                           "accepted samples in each population."
        self.num_accepted = num_accepted
        self.weights = np.ones(t)

    def infer(self, num_samples, batch_size, t=5, chunk_size=10, ensemble_size=1, normalize=True):
        if self.t is not None and self.t != t:
            if self.use_logger:
                self.logger.warning("Supplied value of t does not match length of supplied prior functions or epsilons."
                                    " Giving preference to the supplied list of priors or epsilons.")
            else:
                print("Supplied value of t does not match supplied length of prior/epsilon lists. "
                      "Using the length of lists as the value of t.")
            t = self.t

        # initialize the attributes
        self.setup_population_attributes(t, num_samples)

        # initialize the epsilons
        if type(self.epsilons) is not list:
            # the user only supplied the initial epsilon
            # automatically calculate the subsequent epsilons
            # the following function call will give us a list where the 0th index holds the initial epsilon
            self.epsilons = self.eps_adaptation_function(self.epsilons)

        