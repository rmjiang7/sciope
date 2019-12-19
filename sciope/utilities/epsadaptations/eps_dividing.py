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
The Epsilon-Dividing Epsilon Adaptation Functions for
Sequential Monte-Carlo Approximate Bayesian Computation (SMC-ABC)
"""


def eps_halving(initial_eps, t):
    return eps_divide_k(initial_eps, t, 2.0)


def eps_divide_k(initial_eps, t, k):
    epsilons = []
    epsilon = initial_eps
    for i in range(t):
        epsilons.append(epsilon)
        epsilon /= k
    return epsilons
