from pymor.basic import *
from pymor.models.interface import Model
from pymor.operators.interface import Operator
from pymor.operators.constructions import ZeroOperator
import sys

from pymor_dealii.pymor.operator import DealIIMatrixOperator
from pymor_dealii.pymor.vectorarray import DealIIVectorSpace
from pymor_dealii.pymor.gui import DealIIVisualizer
sys.path.insert(0,"../lib")


class CouplingInputOperator(Operator):
    """Read and apply data from preCICE

    A class that defines the read direction of preCICE, where new coupling
    data is read and applied.

    Parameters
    ----------
    """
    def __init__(self, *args):
        pass

    def apply(self, U, mu=None):
        pass


class StationaryPreciceModel(Model):
    """Generic interface class for coupled simulation via pyMOR and preCICE.

    In a first place, the full order model is solved using the following
    parameters.

    Parameters
    ----------
    operator
        The |Operator| of the linear problem.
    rhs
        An additional source term of the problem. A |VectorArray| (which can be zero as well)
    coupling_input_operator
        Essentially the advance operator of preCICE, where data is exchanged in read direction
        and the boundary conditions are applied to the model
    coupling_output_operator
        Essentially the advance operator of preCICE, where data is exchanged in write direction
    """
    def __init__(self, operator, rhs, coupling_input_operator, coupling_output_operator,
                 output_functional=None, products=None,
                 error_estimator=None, visualizer=None, name=None):

        if isinstance(rhs, VectorArray):
            assert rhs in operator.range
            rhs = VectorOperator(rhs, name='rhs')

        assert rhs.range == operator.range and rhs.source.is_scalar and rhs.linear
        assert output_functional is None or output_functional.source == operator.source
        assert coupling_input_operator.range == operator.range
        assert coupling_output_operator.source == operator.source

        super().__init__(products=products, error_estimator=error_estimator, visualizer=visualizer, name=name)

        self.__auto_init(locals())
        self.solution_space = operator.source
        if output_functional is not None:
            self.dim_output = output_functional.range.dim

    _compute_allowed_kwargs = frozenset({'coupling_input', 'coupling_output'})

    def _compute_solution(self, mu=None, **kwargs):
        # this should only be called when compute has been called with
        # solution=False and e.g. output=True
        return self._compute(solution=True, mu=mu, **kwargs)['solution']

    def _compute(self, solution=False, coupling_output=False, mu=None, **kwargs):
        retval = {}
        # Compute solution = compute the solution of the actual system
        # coupling output  = pass new data to preCICE
        if solution or coupling_output:
            coupling_input = kwargs.pop(coupling_input, None)
            if coupling_input is None:
                self.logger.warn('Solving without coupling input')
                coupling_input = self.coupling_input_operator.source.zeros()
            assert coupling_input in self.coupling_input_operator.source
            assert len(coupling_input) == 1

            rhs = self.rhs.as_range_array(mu) + self.coupling_input_operator.apply(coupling_input, mu=mu)

            retval['solution'] = self.operator.apply_inverse(rhs, mu=mu)

        if coupling_output:
            retval['coupling_output'] = self.coupling_output_operator.apply(retval['solution'], mu=mu)

        return retval

    # def _compute(self, solution=False, coupling_output=False, mu=None, **kwargs):
    #     retval = {}
    #     # Compute solution = compute the solution of the actual system
    #     # coupling output  = pass new data to preCICE
    #     coupling_input = kwargs.pop(coupling_input, None)
    #     if coupling_input is None:
    #         self.logger.warn('Solving without coupling input')
    #         coupling_input = self.coupling_input_operator.source.zeros()
    #     assert coupling_input in self.coupling_input_operator.source
    #     assert len(coupling_input) == 1

    #     rhs = self.rhs.as_range_array(mu) + self.coupling_input_operator.apply(coupling_input, mu=mu)

    #     retval['solution'] = self.operator.apply_inverse(rhs, mu=mu)

    #     self.coupling_operator.advance()

    #     return retval

# import the deal.II model
from dealii_heat_equation import HeatExample

# instantiate deal.II model and print some information
dealii = HeatExample(parameter_file="parameters.prm")
# Create the grid
dealii.make_grid()
# setup the system, i.e., matrices etc.
dealii.setup_system()

solution = VectorOperator(DealIIVectorSpace.make_array([dealii.get_solution()]))
rhs = VectorOperator(DealIIVectorSpace.make_array([dealii.get_rhs()]))

# initialize the adapter and everything related
coupling_data = dealii.initialize_precice()
### Create (not yet reduced) model
model = StationaryPreciceModel(DealIIMatrixOperator(dealii.stationary_matrix()), 0, CouplingInputOperator)

while dealii.is_coupling_ongoing():
    #1 assemblre rhs -> input_operator_apply
    #2 solve -> apply_inverse
    #3 advance -> output_operator_apply
    model.compute