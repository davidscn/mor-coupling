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

    def __init__(self, operator,
                 coupling_input_operator=None,
                 coupling_output_operator=None,
                 output_functional=None, products=None,
                 error_estimator=None, visualizer=None, name=None):

        if coupling_input_operator is None:
            coupling_input_operator = IdentityOperator(operator.range)
        if coupling_output_operator is None:
            coupling_output_operator = IdentityOperator(operator.source)
        assert output_functional is None or output_functional.source == operator.source

        super().__init__(products=products, error_estimator=error_estimator,
                         visualizer=visualizer, name=name)

        self.__auto_init(locals())
        self.solution_space = operator.source
        if output_functional is not None:
            self.dim_output = output_functional.range.dim

    _compute_allowed_kwargs = frozenset({'coupling_input'})

    def _compute_solution(self, mu=None, coupling_input=None, **kwargs):
        # Assemble the RHS originating from the coupling data
        rhs = self.coupling_input_operator.apply(coupling_input)
        # Solve the system and retrieve the solution as an VectorOperator
        solution = self.operator.apply_inverse(rhs, mu=mu)
        coupling_output = self.coupling_output_operator.apply(solution, mu=mu)

        return {'solution': solution, 'coupling_output': coupling_output}


class PreciceCoupler:

    def __init__(self, initial_coupling_input):
        self._coupling_data = initial_coupling_input.zeros()._list[0].impl
        dealii.initialize_precice(initial_coupling_input._list[0].impl, self._coupling_data)

    def advance(self, coupling_input):
        rhs = coupling_input.zeros()
        dealii.assemble_rhs(self._coupling_data, coupling_input._list[0].impl, rhs._list[0].impl)
        dealii.advance(coupling_input._list[0].impl, self._coupling_data)
        return rhs


# instantiate deal.II model and print some information
dealii = HeatExample(parameter_file="parameters.prm")
# Create the grid
dealii.make_grid()
# setup the system, i.e., matrices etc.
dealii.setup_system()

# Create (not yet reduced) model
model = StationaryPreciceModel(DealIIMatrixOperator(dealii.stationary_matrix()))

coupling_output = model.solution_space.zeros()
coupler = PreciceCoupler(coupling_output)
# Result file number counter
counter = 0
# Let preCICE steer the coupled simulation
while dealii.is_coupling_ongoing():
    counter += 1
    # Compute the solution of the time step
    coupling_input = coupler.advance(coupling_output)
    data = model.compute(solution=True, coupling_input=coupling_input)
    solution, coupling_output = data['solution'], data['coupling_output']
    # and output the results
    dealii.output_results(solution._list[0].impl, counter)
