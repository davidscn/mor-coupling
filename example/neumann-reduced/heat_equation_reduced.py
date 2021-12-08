from pickle import dump
import sys

from pymor.algorithms.pod import pod
from pymor.models.interface import Model
from pymor.operators.constructions import IdentityOperator, ZeroOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace

from pymor_dealii.pymor.operator import DealIIMatrixOperator

sys.path.insert(0, "../../lib")
from dealii_heat_equation import HeatExample


USE_ROM = len(sys.argv) == 2 and int(sys.argv[1])


class StationaryPreciceModel(Model):
    """Generic interface class for coupled simulation via pyMOR and preCICE.

    In a first place, the full order model is solved using the following
    parameters.

    Parameters
    ----------
    operator
        The |Operator| of the linear problem.
    coupling_input_operator
        Operator mapping the coupling contributions to the rhs.
    coupling_output_operator
        Operator mapping the solution to the coupling contributions.
    """

    def __init__(self, operator,
                 coupling_input_operator=None, coupling_output_operator=None,
                 output_functional=None, products=None, error_estimator=None,
                 visualizer=None, name=None):

        coupling_input_operator = coupling_input_operator or IdentityOperator(operator.range)
        coupling_output_operator = coupling_output_operator or IdentityOperator(operator.source)
        output_functional = output_functional or ZeroOperator(NumpyVectorSpace(0), operator.source)
        assert coupling_input_operator.range == operator.range
        assert coupling_output_operator.source == operator.source
        assert output_functional.source == operator.source

        super().__init__(products=products, error_estimator=error_estimator,
                         visualizer=visualizer, name=name)

        self.__auto_init(locals())
        self.solution_space = operator.source
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
fom = StationaryPreciceModel(DealIIMatrixOperator(dealii.stationary_matrix()))
coupling_output = fom.solution_space.zeros()
coupler = PreciceCoupler(coupling_output)

if USE_ROM:
    raise NotImplementedError
else:
    model = fom

counter = 0
solution = model.solution_space.empty()
# Let preCICE steer the coupled simulation
while dealii.is_coupling_ongoing():
    counter += 1
    # Compute the solution of the time step
    coupling_input = coupler.advance(coupling_output)
    data = model.compute(solution=True, coupling_input=coupling_input)
    solution.append(data['solution'])
    coupling_output = data['coupling_output']
    # and output the results
    dealii.output_results(solution._list[-1].impl, counter)

if not USE_ROM:
    basis, svals = pod(solution, rtol=1e-5)
    print(svals)
    with open('reduced_basis.dat', 'wb') as f:
        dump(basis.to_numpy(), f)
