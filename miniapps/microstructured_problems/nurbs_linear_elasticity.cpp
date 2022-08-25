/*
 * Example to calculate heat transfer equation on a microstructure with locally
 * dependent neumann boundary conditions on one side, and strong imposed
 * Dirichlet conditions on the other side
 *
 * The example is a mix between the miniapps/nurbs example and ex27
 */

#include <fstream>
#include <iostream>

#include "mfem.hpp"

#ifndef MFEM_USE_SUITESPARSE
#error This Implementation requires build with MFEM_USE_SUITESPARSE=ON
#endif

using namespace std;
using namespace mfem;

int main(int argc, char *argv[]) {
  ////////
  // 1. Settings
  ////////
  // Mesh and Refinement
  int ref_levels = -1;  // Level of refinement
  int order = 1;        // Order of refinement
  const char *mesh_file = "foo.mesh";
  // Problem setup
  double user_mu = -1.;     // poisson ratio
  double user_lambda = 1.;  // First Lamé constant
  // Boundary Conditions
  int boundary_index_dirichlet = 2;
  int boundary_index_neumann = 1;
  // Irregular Boundary Condition
  double load_factor = 1.;
  int dim{3};
  auto LoadFunction = [&load_factor, &dim](const Vector &coordinate,
                                           Vector &load) -> void {
    double factor{};
    const int dim{3};
    // Assign pointer to origin
    load[0] = 0.;
    for (unsigned i{1}; i < dim; i++) {
      load[i] = -coordinate[i];
      factor += coordinate[i] * coordinate[i];
    }
    // Normalize and scale
    factor = load_factor / std::sqrt(factor);
    // Rescale vector
    for (unsigned i{1}; i < dim; i++) {
      load[i] *= factor;
    }
    return;
  };

  // Parse command-line options
  OptionsParser args(argc, argv);
  // Mesh options
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&ref_levels, "-r", "--refine",
                 "Number of times to refine the mesh uniformly, -1 for auto.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree).");
  // Boundary Condition options
  args.AddOption(&boundary_index_dirichlet, "-db", "--dirichlet-boundary",
                 "Boundary ID for Dirichlet condition");
  args.AddOption(&boundary_index_neumann, "-nb", "--neumann-boundary",
                 "Boundary ID for Neumann condition");
  args.AddOption(&load_factor, "-load", "--load-factor",
                 "Load-factor for force, pointing to origin");
  // Problem setup
  args.AddOption(&user_mu, "-mu", "--material-mu",
                 "Constant Poisson ratio of the material");
  args.AddOption(&user_lambda, "-lambad", "--material-lambda",
                 "Constant First Lamé-constant of the material");

  // Check options
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  } else {
    args.PrintOptions(cout);
  }

  ////////
  // 2. Mesh import and refinement
  ////////
  // Read mesh from the given file
  Mesh *mesh = new Mesh(mesh_file, 1, 1);
  dim = mesh->Dimension();

  if (mesh->NURBSext) {
    mesh->DegreeElevate(order, order);
  } else {
    std::cerr << "Frontend specialized for NURBS meshes" << std::endl;
    return 3;
  }
  // Refine
  for (int l{0}; l < ref_levels; l++) {
    mesh->UniformRefinement();
  }

  mesh->PrintInfo();

  // Define Finite element space on mesh
  // Create a FE space for NURBS
  FiniteElementSpace *fespace = mesh->GetNodes()->FESpace();
  std::cout << "Number of finite element unknowns: " << fespace->GetTrueVSize()
            << std::endl;

  ////////
  // 3. Determine surfaces for Boundary conditions
  ////////
  // Check the total number of boundaries identified in the mesh file
  if (mesh->bdr_attributes.Max() - 1 <
      std::max(boundary_index_dirichlet, boundary_index_neumann)) {
    cout << "Mesh does not provide sufficient number of boundaries.";
    return 1;
  }
  // Create boundary vectors of bools
  Array<int> is_Neumann_boundary(mesh->bdr_attributes.Max());
  Array<int> is_Dirichlet_boundary(mesh->bdr_attributes.Max());

  is_Neumann_boundary = 0;
  is_Neumann_boundary[boundary_index_neumann] = 1;
  is_Dirichlet_boundary = 0;
  is_Dirichlet_boundary[boundary_index_dirichlet] = 1;

  // Essential Boundary conditions need to be treated differently within the
  // linear system
  Array<int> ess_tdof_list(0);
  if (mesh->bdr_attributes.Size()) {
    // For a continuous basis the linear system must be modified to enforce an
    // essential (Dirichlet) boundary condition. In the DG case this is not
    // necessary as the boundary condition will only be enforced weakly.
    fespace->GetEssentialTrueDofs(is_Dirichlet_boundary, ess_tdof_list);
  }

  ////////
  // 4. Define values on boundary conditions
  ////////

  // Define Coefficient
  VectorFunctionCoefficient load_function(dim, LoadFunction);
  // Assemble the linear form for the right hand side vector.
  LinearForm *b = new LinearForm(fespace);

  // Add the desired value for n.Grad(u) on the Neumann boundary
  b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(load_function),
                           is_Neumann_boundary);
  cout << "Assembling RHS ... " << flush;
  b->Assemble();
  cout << "\t[FINISHED]" << std::endl;

  ////////
  // 5. Define and init solution and bilinear form (operators)
  ////////

  // -- Initial solution --
  // Define the solution vector u as a finite element grid function
  GridFunction displacement_field(fespace);
  // Initialize u with initial guess of zero.
  displacement_field = 0.0;

  // -- Problem setup --
  // Set up the bilinear form a(.,.) on the finite element space corresponding
  // to the Laplacian operator -Delta, by adding the Diffusion domain
  // integrator.
  Vector lambda_(mesh->attributes.Max());
  Vector mu_(mesh->attributes.Max());
  lambda_ = user_lambda;
  mu_ = user_mu;
  PWConstCoefficient lambda_func(lambda_);
  PWConstCoefficient mu_func(mu_);

  BilinearForm *a = new BilinearForm(fespace);
  a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func, mu_func));
  cout << "Assembling Matrix ... " << flush;
  a->Assemble();
  cout << "\t[FINISHED]" << std::endl;

  ////////
  // 6. Construct the linear system
  ////////
  OperatorPtr A;
  Vector B, X;
  cout << "Forming linear system ... " << flush;
  a->FormLinearSystem(ess_tdof_list, displacement_field, *b, A, X, B);
  cout << "\t[FINISHED]" << std::endl;

  // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the
  // system.
  UMFPackSolver umf_solver;
  umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
  umf_solver.SetOperator(*A);
  umf_solver.Mult(B, X);

  // Recover the grid function corresponding to U. This is the local finite
  // element solution.
  a->RecoverFEMSolution(X, *b, displacement_field);

  ////////
  // 7. Save results into files
  ////////
  ofstream mesh_ofs("refined.mesh");
  mesh_ofs.precision(8);
  mesh->Print(mesh_ofs);
  ofstream sol_ofs("sol.gf");
  sol_ofs.precision(8);
  displacement_field.Save(sol_ofs);

  ////////
  // 8. Free Memory
  ////////
  delete a;
  delete b;
  delete fespace;
  delete mesh;

  return 0;
}
