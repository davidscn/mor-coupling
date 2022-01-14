from pickle import dump, load
import sys

from pymor.algorithms.pod import pod
from pymor.algorithms.projection import project
from pymor.models.interface import Model
from pymor.operators.constructions import IdentityOperator, ZeroOperator
from pymor.operators.interface import Operator
from pymor.vectorarrays.numpy import NumpyVectorSpace

from pymor_dealii.pymor.operator import DealIIMatrixOperator

sys.path.insert(0, "../../lib")
from dealii_heat_equation import HeatExample


# USE_ROM == False: Solve full-order model and build reduced basis
# USE_ROM == True:  Solve reduced-order model
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


class CouplingInputOperator(Operator):

    linear = True

    def __init__(self, space):
        self.source = self.range = self.space = space

    def apply(self, U, mu=None):
        assert U in self.source
        if len(U) != 1:
            raise NotImplementedError
        rhs = self.range.zeros()
        dealii.assemble_rhs(U._list[0].impl, rhs._list[0].impl)
        return rhs


class PreciceCoupler:

    def __init__(self, space):
        self.coupling_input_space = self.coupling_output_space = space

    def init(self, initial_coupling_output):
        initial_coupling_input = self.coupling_input_space.zeros()
        dealii.initialize_precice(initial_coupling_output._list[0].impl, initial_coupling_input._list[0].impl)
        return initial_coupling_input

    def advance(self, coupling_output):
        coupling_input = self.coupling_input_space.zeros()
        dealii.advance(coupling_output._list[0].impl, coupling_input._list[0].impl)
        return coupling_input


# instantiate deal.II model and print some information
dealii = HeatExample(parameter_file="parameters.prm")
# Create the grid
dealii.make_grid()
# setup the system, i.e., matrices etc.
dealii.setup_system()

# Create full-order model
operator = DealIIMatrixOperator(dealii.stationary_matrix())
coupling_input_operator = CouplingInputOperator(operator.source)
fom = StationaryPreciceModel(operator, coupling_input_operator=coupling_input_operator)

# Setup coupling with PreCICE
coupling_output = fom.solution_space.zeros()
dealii.set_initial_condition(coupling_output._list[0].impl)
coupler = PreciceCoupler(fom.solution_space)
coupling_input = coupler.init(coupling_output)

# Choose model to simulate
if USE_ROM:
    # load pre-computed reduced basis
    with open('reduced_basis.dat', 'rb') as f:
        RB = fom.solution_space.from_numpy(load(f))

    # build reduced-order model
    projected_operator                 = project(fom.operator, RB, RB)
    projected_coupling_input_operator  = project(fom.coupling_input_operator, RB, None)
    projected_coupling_output_operator = project(fom.coupling_output_operator, None, RB)
    model = StationaryPreciceModel(projected_operator,
                                   projected_coupling_input_operator,
                                   projected_coupling_output_operator)
else:
    model = fom


# Let preCICE steer the coupled simulation
solution = model.solution_space.empty()
while dealii.is_coupling_ongoing():
    # Compute the solution of the time step
    data = model.compute(solution=True, coupling_input=coupling_input)
    solution.append(data['solution'])
    coupling_output = data['coupling_output']
    coupling_input = coupler.advance(coupling_output)


# Output the solution to VTK
if USE_ROM:
    # reconstruct high-dimensional solution field
    solution = RB.lincomb(solution.to_numpy())
for i, s in enumerate(solution, start=1):
    dealii.output_results(s._list[0].impl, i)


# Build reduced basis
if not USE_ROM:
    RB, svals = pod(solution, rtol=1e-3)
    print(svals)
    with open('reduced_basis.dat', 'wb') as f:
        dump(RB.to_numpy(), f)
