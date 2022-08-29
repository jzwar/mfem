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
#error This example requires that MFEM is built with MFEM_USE_SUITESPARSE=ON
#endif

using namespace std;
using namespace mfem;

int main(int argc, char *argv[]) {
  //////////
  // Initialize
  //////////
  Mpi::Init(argc, argv);
  int num_procs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  Hypre::Init();

  ////////
  // 1. Settings
  ////////
  // Solver Settings
  const char *device_config = "cpu";

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
    if (myid == 0) {
      args.PrintUsage(cout);
    }
    return 1;
  } else {
    if (myid == 0) {
      args.PrintOptions(cout);
    }
  }

  // Setup device to user configuration (e.g., GPU, CUDA,...)
  Device device(device_config);
  if (myid == 0) {
    device.Print();
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
    if (myid == 0) {
      std::cerr << "Frontend specialized for NURBS meshes" << std::endl;
    }
    return 3;
  }
  // Refine
  for (int l{0}; l < ref_levels; l++) {
    mesh->UniformRefinement();
  }

  // Create a parallel mesh from serial mesh
  ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
  if (myid == 0) {
    mesh->PrintInfo();
  }

  // Scale load
  const double surface_area = EvaluateFunctionOnSurface(
      mesh, boundary_index_neumann, [](const Vector &) { return 1; });

  // Scale Load factor
  load_factor /= surface_area;

  delete mesh;

  // Define Finite element space on mesh
  // Create a FE space for NURBS
  ParFiniteElementSpace *fespace =
      (ParFiniteElementSpace *)pmesh->GetNodes()->FESpace();
  HYPRE_BigInt size = fespace->GlobalTrueVSize();
  if (myid == 0) {
    std::cout << "Number of finite element unknowns: " << size << std::endl
              << "Assembling: " << std::endl;
  }

  ////////
  // 3. Determine surfaces for Boundary conditions
  ////////
  // Check the total number of boundaries identified in the mesh file
  if (pmesh->bdr_attributes.Max() - 1 <
      std::max(boundary_index_dirichlet, boundary_index_neumann)) {
    cout << "Mesh does not provide sufficient number of boundaries.";
    return 1;
  }
  // Create boundary vectors of bools
  Array<int> is_Neumann_boundary(pmesh->bdr_attributes.Max());
  Array<int> is_Dirichlet_boundary(pmesh->bdr_attributes.Max());

  is_Neumann_boundary = 0;
  is_Neumann_boundary[boundary_index_neumann] = 1;
  is_Dirichlet_boundary = 0;
  is_Dirichlet_boundary[boundary_index_dirichlet] = 1;

  // Essential Boundary conditions need to be treated differently within the
  // linear system
  Array<int> ess_tdof_list(0);
  if (pmesh->bdr_attributes.Size()) {
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
  ParLinearForm *b = new ParLinearForm(fespace);

  // Add the desired value for n.Grad(u) on the Neumann boundary
  b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(load_function),
                           is_Neumann_boundary);
  if (myid == 0) {
    cout << "Assembling RHS ... " << flush;
  }
  b->Assemble();
  if (myid == 0) {
    cout << "\t[FINISHED]" << std::endl;
  }

  ////////
  // 5. Define and init solution and bilinear form (operators)
  ////////

  // -- Initial solution --
  // Define the solution vector u as a finite element grid function
  ParGridFunction displacement_field(fespace);
  // Initialize u with initial guess of zero.
  displacement_field = 0.0;

  // -- Problem setup --
  // Set up the bilinear form a(.,.) on the finite element space corresponding
  // to the Laplacian operator -Delta, by adding the Diffusion domain
  // integrator.
  Vector lambda_(pmesh->attributes.Max());
  Vector mu_(pmesh->attributes.Max());
  lambda_ = user_lambda;
  mu_ = user_mu;
  PWConstCoefficient lambda_func(lambda_);
  PWConstCoefficient mu_func(mu_);

  ParBilinearForm *a = new ParBilinearForm(fespace);
  a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func, mu_func));
  if (myid == 0) {
    cout << "Assembling Matrix ... " << flush;
  }
  a->Assemble();
  if (myid == 0) {
    cout << "\t[FINISHED]" << std::endl;
  }

  ////////
  // 6. Construct the linear system
  ////////
  HypreParMatrix A;
  Vector B, X;
  if (myid == 0) {
    cout << "Forming linear system ... " << flush;
  }
  a->FormLinearSystem(ess_tdof_list, displacement_field, *b, A, X, B);
  if (myid == 0) {
    cout << "\t[FINISHED]" << std::endl;
    cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
  }

  // 14. Define and apply a parallel PCG solver for A X = B with the BoomerAMG
  //     preconditioner from hypre.
  HypreSolver *amg = new HypreBoomerAMG(A);
  HyprePCG *pcg = new HyprePCG(A);
  pcg->SetTol(1e-12);
  pcg->SetMaxIter(2000);
  pcg->SetPrintLevel(5);
  pcg->SetPreconditioner(*amg);
  pcg->Mult(B, X);

  // Recover the grid function corresponding to U. This is the local finite
  // element solution.
  a->RecoverFEMSolution(X, *b, displacement_field);

  ////////
  // 7. Save results into files
  ////////
  {
    ostringstream mesh_name, sol_name;
    mesh_name << "mesh." << setfill('0') << setw(6) << myid;
    sol_name << "sol." << setfill('0') << setw(6) << myid;

    ofstream mesh_ofs(mesh_name.str().c_str());
    mesh_ofs.precision(8);
    pmesh->Print(mesh_ofs);

    ofstream sol_ofs(sol_name.str().c_str());
    sol_ofs.precision(8);
    displacement_field.Save(sol_ofs);
  }

  ////////
  // 8. Free Memory
  ////////
  delete pcg;
  delete amg;
  delete a;
  delete b;
  // delete fespace;
  // delete pmesh;

  return 0;
}

// Evaluate Objective function
double EvaluateFunctionOnSurface(
    const Mesh &serial_mesh, const Array<int> &bdr,
    std::function<double(const IntegrationPoint &)> &surface_function) {
  // Init return value
  double func_value{};

  // Retrieve FE information
  const FiniteElementSpace *fes = mesh.GetNodes()->FESpace();
  Array<int> dof_ids;

  // Loop over all available Boundaries
  for (int i = 0; i < mesh.GetNBE(); i++) {
    if (bdr[mesh.GetBdrAttribute(i) - 1] == 0) {
      continue;
    }

    FaceElementTransformations *FTr = mesh.GetBdrFaceTransformations(i);
    if (FTr == nullptr) {
      continue;
    }

    const int int_order = 2 * fe.GetOrder() + 3;
    const IntegrationRule &ir = IntRules.Get(FTr->FaceGeom, int_order);

    fes->GetElementDofs(FTr->Elem1No, dof_ids);

    // Loop over Integration points
    for (int j = 0; j < ir.GetNPoints(); j++) {
      // Prepare integration values
      const IntegrationPoint &ip = ir.IntPoint(j);
      IntegrationPoint eip;
      FTr->Loc1.Transform(ip, eip);
      FTr->Face->SetIntPoint(&ip);

      // Calculate Jacobi-determinant
      double face_weight = FTr->Face->Weight();

      // Evaluate surface function value
      const double surface_function_at_ip = surface_function(eip);

      // Measure the length of the boundary
      surface_area += ip.weight * face_weight;

      // Sum up contributions
      func_value += surface_function_at_ip * ip.weight * face_weight;
    }
  }

  // Return the average value of alpha * n.Grad(x) + beta * x
  return func_value;
}