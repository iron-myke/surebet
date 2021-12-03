import os
import pandas as pd
from scipy.optimize import minimize, Bounds, shgo, brute, basinhopping, differential_evolution
import json
import numpy as np
import optuna