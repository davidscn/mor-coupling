#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/revision.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <adapter/adapter.h>
#include <adapter/parameters.h>
#include <adapter/time_handler.h>

#include <fstream>
#include <iostream>

namespace Heat_Transfer
{
  using namespace dealii;

  template <int dim>
  class HeatEquation
  {
  public:
    HeatEquation(const std::string &parameter_file);

    void
    make_grid();

    void
    setup_system(double coefficient1,
                 double coefficient2,
                 double threshold_x,
                 double threshold_y);

    void
    assemble_rhs(const Vector<double> &heat_flux_, Vector<double> &rhs_);

    void
    output_results(const Vector<double> &solution_, const int file_index) const;

    const SparseMatrix<double> &
    stationary_system_matrix() const;

    void
    advance(const Vector<double> &solution_, Vector<double> &heat_flux_);

    bool
    is_coupling_ongoing() const;

    void
    set_initial_condition(Vector<double> &solution_);

    void
    initialize_precice(Vector<double> &solution_,
                       Vector<double> &coupling_data_);

    const Parameters::AllParameters parameters;

  private:
    void
    print_configuration() const;

    Triangulation<dim>       triangulation;
    const types::boundary_id interface_boundary_id = 0;
    const types::boundary_id dirichlet_boundary_id = 1;
    FE_Q<dim>                fe;
    DoFHandler<dim>          dof_handler;

    AffineConstraints<double> constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> stationary_system_matrix_;

    Vector<double> solution;
    Vector<double> system_rhs;

    mutable TimerOutput                      timer;
    Adapter::Time                            time;
    Adapter::Adapter<dim, 1, Vector<double>> adapter;

    const double theta;
    const double alpha;
    const double beta;
  };

  template <int dim>
  class AnalyticSolution : public Function<dim>
  {
  public:
    AnalyticSolution(const double alpha, const double beta)
      : Function<dim>()
      , alpha(alpha)
      , beta(beta)
    {}

    virtual double
    value(const Point<dim> &, const unsigned int component = 0) const override
    {
      (void)component;
      AssertIndexRange(component, 1);
      return 1.5;
    }

  private:
    const double alpha;
    const double beta;
  };

  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide(const double alpha, const double beta)
      : Function<dim>()
      , alpha(alpha)
      , beta(beta)
    {}

    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));
      return 0;
    }

  private:
    const double alpha;
    const double beta;
  };

  // Returns a spatially varying coefficient.
  // If the domain is below the threshold of x and y, coefficient 1 is returned,
  // If the domain is above the threshold of x and y, coefficient 2 is returned.
  template <int dim, typename value_type = double>
  class Coefficient : public Function<dim, value_type>
  {
  public:
    Coefficient(double coefficient1,
                double coefficient2,
                double threshold_x,
                double threshold_y)
      : Function<dim>()
      , coefficient1(coefficient1)
      , coefficient2(coefficient2)
      , threshold_x(threshold_x)
      , threshold_y(threshold_y)
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));

      if (p[0] < threshold_x && p[1] < threshold_y)
        return coefficient1;
      else
        return coefficient2;
    }

  private:
    const double coefficient1;
    const double coefficient2;
    const double threshold_x;
    const double threshold_y;
  };


  template <int dim>
  HeatEquation<dim>::HeatEquation(const std::string &parameter_file)
    : parameters(parameter_file)
    , fe(parameters.poly_degree)
    , dof_handler(triangulation)
    , timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)
    , time(parameters.end_time, parameters.delta_t)
    , adapter(parameters, interface_boundary_id)
    , theta(1)
    , alpha(3)
    , beta(0)
  {
    print_configuration();
  }

  template <int dim>
  const SparseMatrix<double> &
  HeatEquation<dim>::stationary_system_matrix() const
  {
    return stationary_system_matrix_;
  }


  template <int dim>
  void
  HeatEquation<dim>::print_configuration() const
  {
    static const unsigned int n_threads = MultithreadInfo::n_threads();

    // Query adapter and deal.II info
    const std::string adapter_info =
      GIT_SHORTREV == std::string("") ?
        "unknown" :
        (GIT_SHORTREV + std::string(" on branch ") + GIT_BRANCH);
    const std::string dealii_info =
      DEAL_II_GIT_SHORTREV == std::string("") ?
        "unknown" :
        (DEAL_II_GIT_SHORTREV + std::string(" on branch ") +
         DEAL_II_GIT_BRANCH);

    std::cout
      << "-----------------------------------------------------------------------------"
      << std::endl
      << "--     . running with " << n_threads << " thread"
      << (n_threads == 1 ? "" : "s") << std::endl;

    std::cout << "--     . adapter revision " << adapter_info << std::endl;
    std::cout << "--     . deal.II " << DEAL_II_PACKAGE_VERSION << " (revision "
              << dealii_info << ")" << std::endl;
    std::cout
      << "-----------------------------------------------------------------------------"
      << std::endl
      << std::endl;
  }


  template <int dim>
  void
  HeatEquation<dim>::advance(const Vector<double> &solution_,
                             Vector<double> &      heat_flux_)
  {
    timer.enter_subsection("advance preCICE");
    // We fake the implicit coupling here. The checkpointing is from the solver
    // perspective a NOP, but we tell preCICE that we stores as checkpoint
    std::vector<Vector<double> *> dummy(0);
    adapter.save_current_state_if_required(dummy, time);

    adapter.advance(solution_, heat_flux_, time.get_delta_t());

    adapter.reload_old_state_if_required(dummy, time);
    timer.leave_subsection("advance preCICE");
  }


  template <int dim>
  bool
  HeatEquation<dim>::is_coupling_ongoing() const
  {
    return adapter.precice.isCouplingOngoing();
  }


  template <int dim>
  void
  HeatEquation<dim>::set_initial_condition(Vector<double> &solution_)
  {
    AnalyticSolution<dim> initial_condition(alpha, beta);
    initial_condition.set_time(0);
    VectorTools::interpolate(dof_handler, initial_condition, solution_);
  }


  template <int dim>
  void
  HeatEquation<dim>::initialize_precice(Vector<double> &solution_,
                                        Vector<double> &coupling_data_)
  {
    timer.enter_subsection("initialize preCICE");
    coupling_data_ = 0;
    output_results(solution_, 0);

    adapter.initialize(dof_handler, solution_, coupling_data_);
    timer.leave_subsection("initialize preCICE");
  }


  template <int dim>
  void
  HeatEquation<dim>::make_grid()
  {
    GridGenerator::hyper_rectangle(triangulation,
                                   Point<dim>{1, 0},
                                   Point<dim>{2, 1},
                                   true);

    const unsigned int global_refinement = 4;
    triangulation.refine_global(global_refinement);
    AssertThrow(interface_boundary_id == adapter.deal_boundary_interface_id,
                ExcMessage("Wrong interface ID in the Adapter specified"));
  }

  template <int dim>
  void
  HeatEquation<dim>::setup_system(double coefficient1,
                                  double coefficient2,
                                  double threshold_x,
                                  double threshold_y)
  {
    dof_handler.distribute_dofs(fe);

    std::cout << std::endl
              << "===========================================" << std::endl
              << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl
              << std::endl;

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ true);
    sparsity_pattern.copy_from(dsp);
    stationary_system_matrix_.reinit(sparsity_pattern);

    const Coefficient<dim> coefficient(coefficient1,
                                       coefficient2,
                                       threshold_x,
                                       threshold_y);
    MatrixCreator::create_laplace_matrix(dof_handler,
                                         QGauss<dim>(fe.degree + 1),
                                         stationary_system_matrix_,
                                         &coefficient);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    // Here we apply homogenous Dirichlet Boundary conditions on the right-hand
    // side of the domain in order to make the system uniquely solvable.
    // Inhomogenous DBC would be more challenging, because when deal.II applies
    // the DBC, the RHS vector and the solution vector are modified as well (in
    // addition to the matrix). However, we want to keep the RHS and the
    // solution vector independent from the deal.II code here, since they might
    // be modified in the python code. Strictly speaking, the RHS and solution
    // vector are modified now already, but only with zeros, which is not
    // problematic.

    constraints.condense(stationary_system_matrix_, system_rhs);
    {
      solution   = 0;
      system_rhs = 0;
      Functions::ConstantFunction<dim>          boundary_values_function(0);
      std::map<types::global_dof_index, double> boundary_values;
      VectorTools::interpolate_boundary_values(dof_handler,
                                               dirichlet_boundary_id,
                                               boundary_values_function,
                                               boundary_values);

      MatrixTools::apply_boundary_values(boundary_values,
                                         stationary_system_matrix_,
                                         solution,
                                         system_rhs);
    }
  }


  template <int dim>
  void
  HeatEquation<dim>::assemble_rhs(const Vector<double> &heat_flux_,
                                  Vector<double> &      rhs_)
  {
    timer.enter_subsection("assemble rhs");
    rhs_ = 0;
    // Constantly zero at the moment
    RightHandSide<dim> rhs_function(alpha, beta);
    rhs_function.set_time(0);

    Assert(fe.n_components() == rhs_function.n_components,
           ExcDimensionMismatch(fe.n_components(), rhs_function.n_components));

    UpdateFlags update_flags =
      UpdateFlags(update_values | update_quadrature_points | update_JxW_values);

    const QGauss<dim>     quadrature(fe.degree + 1);
    const QGauss<dim - 1> f_quadrature(fe.degree + 1);
    FEValues<dim>         fe_values(StaticMappingQ1<dim>::mapping,
                            fe,
                            quadrature,
                            update_flags);
    FEFaceValues<dim>     fe_f_values(StaticMappingQ1<dim>::mapping,
                                  fe,
                                  f_quadrature,
                                  update_flags);

    const unsigned int dofs_per_cell   = fe_values.dofs_per_cell,
                       n_q_points      = fe_values.n_quadrature_points,
                       n_face_q_points = f_quadrature.size();

    std::vector<types::global_dof_index> dofs(dofs_per_cell);
    Vector<double>                       cell_vector(dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                   endc = dof_handler.end();

    std::vector<double> rhs_values(n_q_points);
    std::vector<double> local_flux(n_face_q_points);

    for (; cell != endc; ++cell)
      {
        fe_values.reinit(cell);

        const std::vector<double> &weights = fe_values.get_JxW_values();
        rhs_function.value_list(fe_values.get_quadrature_points(), rhs_values);

        cell_vector = 0;
        for (unsigned int point = 0; point < n_q_points; ++point)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              cell_vector(i) += rhs_values[point] *
                                fe_values.shape_value(i, point) *
                                weights[point];
            }

        for (const auto &face : cell->face_iterators())
          if (face->at_boundary() &&
              (face->boundary_id() == interface_boundary_id))
            {
              fe_f_values.reinit(cell, face);
              fe_f_values.get_function_values(heat_flux_, local_flux);

              for (unsigned int f_q_point = 0; f_q_point < n_face_q_points;
                   ++f_q_point)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    cell_vector(i) -= fe_f_values.shape_value(i, f_q_point) *
                                      local_flux[f_q_point] *
                                      fe_f_values.JxW(f_q_point);
                  }
            }
        cell->get_dof_indices(dofs);
        constraints.distribute_local_to_global(cell_vector, dofs, rhs_);
      } // end cell loop
    timer.leave_subsection("assemble rhs");
  }



  template <int dim>
  void
  HeatEquation<dim>::output_results(const Vector<double> &solution_,
                                    const int             file_index) const
  {
    timer.enter_subsection("output results");
    std::cout << "Writing solution to " << std::to_string(file_index)
              << std::endl;

    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution_, "Temperature");

    data_out.build_patches();

    const std::string filename =
      "solution-" + std::to_string(file_index) + ".vtk";
    std::ofstream output(filename);
    data_out.write_vtk(output);
    timer.leave_subsection("output results");
  }
} // namespace Heat_Transfer
