from pymor.basic import *
from pymor.models.interface import Model
from pymor.operators.interface import Operator
from pymor.operators.constructions import ZeroOperator
import sys

from pymor_dealii.pymor.operator import DealIIMatrixOperator
from pymor_dealii.pymor.vectorarray import DealIIVectorSpace
from pymor_dealii.pymor.vectorarray import DealIIVector
from pymor_dealii.pymor.gui import DealIIVisualizer
from pymor.vectorarrays.interface import VectorArray


import pymor_dealii_bindings as pd2
sys.path.insert(0, "../../lib")
from dealii_heat_equation import HeatExample



class CouplingOperator(Operator):
    """Read and apply data from preCICE

    A class that defines the read direction of preCICE, where new coupling
    data is read and applied.

    Parameters
    ----------
    dealii
        The dealii implementation within deal.II.
    solution
        A VectorOperator describing the solution
    coupling_rhs
        The VectorOperator resulting from the coupling_input assembly
    coupling_input
        A VectorOperator holding the new coupling data obtained by preCICE
    """

    def __init__(self, dealii, solution, coupling_rhs, coupling_input):
        dealii.initialize_precice(solution.as_range_array()._list[0].impl, coupling_input.as_range_array()._list[0].impl)
        self.__auto_init(locals())

    def apply_advance(self, U, mu=None):
        assert U.source == self.solution.source
        dealii.advance(U.as_range_array()._list[0].impl, self.coupling_input.as_range_array()._list[0].impl)

    def apply(self, U, mu=None):
        pass

    def apply_assemble(self, U, mu=None):
        assert U.source == self.solution.source
        dealii.assemble_rhs(self.coupling_input.as_range_array()._list[0].impl,U.as_range_array()._list[0].impl, self.coupling_rhs.as_range_array()._list[0].impl)
        return self.coupling_rhs

class StationaryPreciceModel(Model):
    """Generic interface class for coupled simulation via pyMOR and preCICE.

    In a first place, the full order model is solved using the following
    parameters.

    Parameters
    ----------
    operator
        The |Operator| of the linear problem.
    coupling_operator
        Operator handling the coupling contributions and the exchange with preCICE
    """

    def __init__(self, operator, coupling_operator,
                 output_functional=None, products=None,
                 error_estimator=None, visualizer=None, name=None):

        assert output_functional is None or output_functional.source == operator.source

        super().__init__(products=products, error_estimator=error_estimator,
                         visualizer=visualizer, name=name)

        self.__auto_init(locals())
        self.solution_space = operator.source
        if output_functional is not None:
            self.dim_output = output_functional.range.dim

    _compute_allowed_kwargs = frozenset({'coupling_input', 'coupling_output'})

    def _compute_solution(self, mu=None, **kwargs):
        # this should only be called when compute has been called with
        # solution=False and e.g. output=True
        return self._compute(solution=True, mu=mu, **kwargs)['solution']

    def _compute(self, solution, mu=None, **kwargs):
        retval = {}

        # Assemble the RHS originating from the coupling data
        rhs = self.coupling_operator.apply_assemble(solution)
        # Solve the system and retrieve the solution as an VectorOperator
        retval['solution'] = VectorOperator(self.operator.apply_inverse(rhs.array, mu=mu))
        # Advance the coupled system, i.e., exchange data etc
        self.coupling_operator.apply_advance(solution, mu=mu)

        return retval

# instantiate deal.II model and print some information
dealii = HeatExample(parameter_file="parameters.prm")
# Create the grid
dealii.make_grid()
# setup the system, i.e., matrices etc.
dealii.setup_system()

# Initialize the python visible vector representation
solution = VectorOperator(
    DealIIVectorSpace.make_array([dealii.get_solution()]))
coupling_rhs = VectorOperator(DealIIVectorSpace.make_array([dealii.get_rhs()]))
coupling_data = VectorOperator(
    DealIIVectorSpace.make_array([dealii.get_coupling_data()]))

coupling_operator=CouplingOperator(dealii, solution, coupling_rhs, coupling_data)
# Create (not yet reduced) model
model = StationaryPreciceModel(DealIIMatrixOperator(dealii.stationary_matrix()), coupling_operator)

# Result file number counter
counter = 0
# Let preCICE steer the coupled simulation
while dealii.is_coupling_ongoing():
    counter+=1
    # Compute the solution of the time step
    solution = model.compute(solution)['solution']
    # and output the results
    dealii.output_results(solution.as_range_array()._list[0].impl, counter)
