from pymor.basic import *

# instantiate deal.II model
from dealii_heat_equation import HeatExample


# wrap as pyMOR discretization
from pymor_dealii.pymor.operator import DealIIMatrixOperator
from pymor_dealii.pymor.vectorarray import DealIIVectorSpace
from pymor_dealii.pymor.gui import DealIIVisualizer


def run(plot_error=True):
    d = StationaryModel(
        operator=DealIIMatrixOperator(
            cpp_disc.stationary_system_matrix()) * ProjectionParameterFunctional('coeff'),

        rhs=VectorOperator(DealIIVectorSpace.make_array([cpp_disc.rhs()])),

        products={'energy': DealIIMatrixOperator(cpp_disc.mu_mat())},

        visualizer=DealIIVisualizer(cpp_disc)
    )
    parameter_space = d.parameters.space((1, 10))

    # choose reduction method
    reductor = CoerciveRBReductor(
        d,
        product=d.energy_product,
        coercivity_estimator=ExpressionParameterFunctional(
            "max(mu)", d.parameters)
    )

    # greedy basis generation
    greedy_data = rb_greedy(d, reductor, parameter_space.sample_uniformly(3),
                            extension_params={'method': 'gram_schmidt'}, max_extensions=5)

    # get reduced order model
    rd = greedy_data['rom']

    # validate reduced order model
    result = reduction_error_analysis(rd, d, reductor,
                                      test_mus=parameter_space.sample_randomly(
                                          10),
                                      basis_sizes=reductor.bases['RB'].dim + 1,
                                      condition=True, error_norms=[d.energy_norm],
                                      plot=plot_error)

    # visualize solution for parameter with maximum reduction error
    mu_max = result['max_error_mus'][0, -1]
    U = d.solve(mu_max)
    U_rb = reductor.reconstruct(rd.solve(mu_max))
    return result, U, U_rb, d


if __name__ == '__main__':
    # print/plot results of validation
    from matplotlib import pyplot as plt
    result, U, U_rb, d = run()
    print(result['summary'])
    ERR = U - U_rb
    d.visualize([U, U_rb, ERR], legend=[
                'fom', 'rom', 'error'], discretization=False)
    plt.show()
