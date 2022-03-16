from pickle import dump, load
import sys
from time import perf_counter

import numpy as np

from pymor.algorithms.pod import pod
from pymor.algorithms.projection import project
from pymor.models.interface import Model
from pymor.operators.constructions import IdentityOperator, ZeroOperator, LincombOperator
from pymor.operators.list import ListVectorArrayOperatorBase
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.vectorarrays.numpy import NumpyVectorSpace

from pymor_dealii.pymor.operator import DealIIMatrixOperator

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


class CouplingInputOperator(ListVectorArrayOperatorBase):

    linear = True

    def __init__(self, space):
        self.source = self.range = self.space = space

    def _apply_one_vector(self, u, mu=None, prepare_data=None):
        rhs = self.range.zero_vector()
        dealii.assemble_rhs(u.impl, rhs.impl)
        return rhs

    # Operator is self-adjoint ..
    _apply_adjoint_one_vector = _apply_one_vector


class PreciceCoupler:

    def __init__(self, space):
        self.coupling_input_space = self.coupling_output_space = space

    def init(self, initial_coupling_output):
        initial_coupling_input = self.coupling_input_space.zeros()
        dealii.initialize_precice(initial_coupling_output.vectors[0].impl, initial_coupling_input.vectors[0].impl)
        return initial_coupling_input

    def advance(self, coupling_output):
        coupling_input = self.coupling_input_space.zeros()
        dealii.advance(coupling_output.vectors[0].impl, coupling_input.vectors[0].impl)
        return coupling_input


def solve(model):
    # Setup coupling with PreCICE
    coupling_output = fom.solution_space.zeros()
    dealii.set_initial_condition(coupling_output.vectors[0].impl)
    coupler = PreciceCoupler(fom.solution_space)
    coupling_input = coupler.init(coupling_output)

    tic = perf_counter()
    # Let preCICE steer the coupled simulation
    solution = model.solution_space.empty()
    # assemble the system operator for given mu to avoid re-assembly in each iteration
    assembled_model = model.with_(operator=model.operator.assemble(model.parameters.parse([1, 0.1])))
    while dealii.is_coupling_ongoing():
        # Compute the solution of the time step
        data = assembled_model.compute(solution=True, coupling_input=coupling_input)
        solution.append(data['solution'])
        coupling_output = data['coupling_output']
        coupling_input = coupler.advance(coupling_output)
    toc = perf_counter()
    dealii.reset_precice()
    return solution, toc-tic


# instantiate deal.II model and print some information
dealii = HeatExample(parameter_file="parameters.prm")
# Create the grid
dealii.make_grid_and_sparsity_pattern()
# setup the system, i.e., matrices etc.
THRESHOLD_X, THRESHOLD_Y = 1.5, 0.5
matrices = [dealii.create_system_matrix(1, 0, THRESHOLD_X, THRESHOLD_Y),
            dealii.create_system_matrix(0, 1, THRESHOLD_X, THRESHOLD_Y)]

# Create full-order model
operators = [DealIIMatrixOperator(matrix) for matrix in matrices]
coefficients = [ProjectionParameterFunctional('coefficient', 2, i) for i in range(2)]
operator = LincombOperator(operators, coefficients)
coupling_input_operator = CouplingInputOperator(operator.source)
fom = StationaryPreciceModel(operator, coupling_input_operator=coupling_input_operator)

tic = perf_counter()
U, t_fom = solve(fom)
t_fom = perf_counter() - tic

RB, svals = pod(U, rtol=1e-3)


# build reduced-order model
projected_operator                 = project(fom.operator, RB, RB)
projected_coupling_input_operator  = project(fom.coupling_input_operator, RB, None)
projected_coupling_output_operator = project(fom.coupling_output_operator, None, RB)
rom = StationaryPreciceModel(projected_operator,
                             projected_coupling_input_operator,
                             projected_coupling_output_operator)


u_rom, t_rom = solve(rom)
U_rom = RB.lincomb(u_rom.to_numpy())
err = (U - U_rom).norm()

# for i, s in enumerate(solution, start=1):
#     dealii.output_results(s.vectors[0].impl, i)


# Build reduced basis
print(f'Time required for FOM solution: {t_fom}s')
print(f'Time required for ROM solution: {t_rom}s')
print(f'Singular values: {svals}')
print(f'Norms: {U.norm()}')
print(f'Errors: {err}')
