from pymor.basic import *
import sys

from pymor_dealii.pymor.operator import DealIIMatrixOperator
from pymor_dealii.pymor.vectorarray import DealIIVectorSpace
from pymor_dealii.pymor.gui import DealIIVisualizer
sys.path.insert(0,"../lib")
# instantiate deal.II model
from dealii_heat_equation import HeatExample
cpp_disc = HeatExample(parameter_file="parameters.prm")

cpp_disc.print_configuration()
cpp_disc.run()
# wrap as pyMOR discretization
