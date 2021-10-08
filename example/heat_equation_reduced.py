from pymor.basic import *
from pymor.models.interface import Model
import sys

from pymor_dealii.pymor.operator import DealIIMatrixOperator
from pymor_dealii.pymor.vectorarray import DealIIVectorSpace
from pymor_dealii.pymor.gui import DealIIVisualizer
sys.path.insert(0,"../lib")


class StationaryPreciceModel(Model):
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

        if solution or coupling_output:
            coupling_input = kwargs.pop(coupling_input, None)
            if coupling_input is None:
                self.logger.warn('Solving without copuling input')
                coupling_input = self.coupling_input_operator.source.zeros()
            assert coupling_input in self.coupling_input_operator.source
            assert len(coupling_input) == 1

            rhs = self.rhs.as_range_array(mu) + self.coupling_input_operator.apply(coupling_input, mu=mu)

            retval['solution'] = self.operator.apply_inverse(rhs, mu=mu)

        if coupling_output:
            retval['coupling_output'] = self.coupling_output_operator.apply(retval['solution'], mu=mu)

        return retval


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
