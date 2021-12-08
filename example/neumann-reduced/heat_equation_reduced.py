from pymor.models.interface import Model
from pymor.operators.constructions import IdentityOperator
import sys

from pymor_dealii.pymor.operator import DealIIMatrixOperator
from pymor_dealii.pymor.vectorarray import DealIIVectorSpace


sys.path.insert(0, "../../lib")
from dealii_heat_equation import HeatExample


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

    def __init__(self, operator, coupling_operator=None,
                 output_functional=None, products=None,
                 error_estimator=None, visualizer=None, name=None):

        if coupling_operator is None:
            coupling_operator = IdentityOperator(operator.range)
        assert output_functional is None or output_functional.source == operator.source

        super().__init__(products=products, error_estimator=error_estimator,
                         visualizer=visualizer, name=name)

        self.__auto_init(locals())
        self.solution_space = operator.source
        if output_functional is not None:
            self.dim_output = output_functional.range.dim

    _compute_allowed_kwargs = frozenset({'coupling_input', 'coupling_output'})

    def _compute_solution(self, mu=None, coupling_input=None, **kwargs):
        # Assemble the RHS originating from the coupling data
        rhs = self.coupling_operator.apply(coupling_input)
        # Solve the system and retrieve the solution as an VectorOperator
        solution = self.operator.apply_inverse(rhs, mu=mu)

        return solution


# instantiate deal.II model and print some information
dealii = HeatExample(parameter_file="parameters.prm")
# Create the grid
dealii.make_grid()
# setup the system, i.e., matrices etc.
dealii.setup_system()

# Initialize the python visible vector representation
solution = DealIIVectorSpace.make_array([dealii.get_solution()])
coupling_data = DealIIVectorSpace.make_array([dealii.get_coupling_data()])

dealii.initialize_precice(solution._list[0].impl, coupling_data._list[0].impl)
# Create (not yet reduced) model
model = StationaryPreciceModel(DealIIMatrixOperator(dealii.stationary_matrix()))

# Result file number counter
counter = 0
# Let preCICE steer the coupled simulation
while dealii.is_coupling_ongoing():
    counter += 1
    # Compute the solution of the time step
    rhs = model.solution_space.zeros()
    dealii.assemble_rhs(coupling_data._list[0].impl, solution._list[0].impl, rhs._list[0].impl)
    new_solution = model.solve(coupling_input=rhs)
    dealii.advance(solution._list[0].impl, coupling_data._list[0].impl)
    solution = new_solution
    # and output the results
    dealii.output_results(solution._list[0].impl, counter)
