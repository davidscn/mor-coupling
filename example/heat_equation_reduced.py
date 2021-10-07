from pymor.basic import *
import sys

from pymor_dealii.pymor.operator import DealIIMatrixOperator
from pymor_dealii.pymor.vectorarray import DealIIVectorSpace
from pymor_dealii.pymor.gui import DealIIVisualizer
sys.path.insert(0,"../lib")

# import the deal.II model
from dealii_heat_equation import HeatExample
# instantiate deal.II model and print some information
dealii = HeatExample(parameter_file="parameters.prm")
dealii.print_configuration()

# setup the system
dealii.setup_system()

### Create reduced model
d = StationaryModel(DealIIMatrixOperator(dealii.stationary_matrix()))



DealIIVectorSpace(dealii.get_rhs())



dealii.run()
# wrap as pyMOR discretization
