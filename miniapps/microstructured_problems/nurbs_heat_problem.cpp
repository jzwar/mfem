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

using namespace std;
using namespace mfem;

// Forward declaration of objctive function
double EvaluateObjectiveFunction(
    const GridFunction &x, const Array<int> &bdr,
    std::function<double(const IntegrationPoint &)> target_temp,
    double &surface_area);

// Define the Target Temperatur profile for the optimization problem
auto TargetTemperatureProfile = [](const IntegrationPoint &position) -> double {
  const auto &x = position.x;
  double T = 25. + (x / 4) * 5;
  return T;
};

// Irregular Boundary Condiiton
auto NeumannBoundaryProfile = [](const Vector &coordinate) -> double {
  const auto &x = coordinate[0];
  return 27. * (4. - x) * x * x / 256. + 1.;
};

int main(int argc, char *argv[]) {
  ////////
  // 1. Settings
  ////////
  // Set default values
  int ref_levels = -1;       // Level of refinement
  double kappa = -1.;        // SIPG penalty parameter
  double conductivity = 1.;  // conductivity in laplace problem
  Array<int> order(1);       // Order to be set on NURBS
  order[0] = 1;
  const char *mesh_file = "foo.mesh";

  // Boundary Conditions
  bool check_boundary_integrals = false;  // Verify Boundary integrals in post
  double dirichlet_bc_val = 0.0;  // Default values for boundary conditions
  double neumann_bc_value = 1.0;  // Default values for boundary conditions
  int boundary_index_dirichlet = 2;
  int boundary_index_neumann = 1;

  // Parse command-line options
  OptionsParser args(argc, argv);
  // Mesh options
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&ref_levels, "-r", "--refine",
                 "Number of times to refine the mesh uniformly, -1 for auto.");
  // Solver options
  args.AddOption(&kappa, "-k", "--kappa",
                 "Sets the SIPG penalty parameters, should be positive."
                 " Negative values are replaced with (order+1)^2.");
  // Problem options
  args.AddOption(&conductivity, "-mat", "--material-value",
                 "Constant value for conductivity"
                 "coefficient before the Laplace operator");
  // Boundary Condition options
  args.AddOption(&boundary_index_dirichlet, "-db", "--dirichlet-boundary",
                 "Boundary ID for Dirichlet condition");
  args.AddOption(&boundary_index_neumann, "-nb", "--neumann-boundary",
                 "Boundary ID for Neumann condition");
  args.AddOption(&dirichlet_bc_val, "-dbc", "--dirichlet-value",
                 "Constant scaling value for Dirichlet Boundary Condition.");
  args.AddOption(&neumann_bc_value, "-nbc", "--neumann-value",
                 "Constant value for Neumann Boundary Condition.");
  args.AddOption(&check_boundary_integrals, "-cbi", "--check-boundary",
                 "-no-cbi", "--no-boundary-check",
                 "Verify Boundary Conditions calculating boundary integrals.");
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
  int dimension{mesh->Dimension()};

  // Refine
  for (int l{0}; l < ref_levels; l++) {
    mesh->UniformRefinement();
  }
  mesh->PrintInfo();

  // Define Finite element space on mesh
  FiniteElementCollection *fec;     // FE collection that stores dof
  NURBSExtension *NURBSext = NULL;  // NURBSextension collects multipatches
  int own_fec = 0;                  // Use default fec for NURBS Mesh

  if (mesh->NURBSext) {
    // Create a FE space for NURBS
    fec = new NURBSFECollection(order[0]);
    own_fec = 1;

    // Resize order to fit NURBS parametric dimension and set order value to
    // requested order
    int nkv = mesh->NURBSext->GetNKV();
    if (order.Size() == 1) {
      int tmp = order[0];
      order.SetSize(nkv);
      order = tmp;
    }
    if (order.Size() != nkv) {
      mfem_error("Wrong number of orders set.");
    }
    NURBSext = new NURBSExtension(mesh->NURBSext, order);
  } else if (order[0] == -1) {  // Isoparametric
    if (mesh->GetNodes()) {
      fec = mesh->GetNodes()->OwnFEC();
      own_fec = 0;
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
    } else {
      cout << "Mesh does not have FEs --> Assume order 1.\n";
      fec = new H1_FECollection(1, dimension);
      own_fec = 1;
    }
  } else {
    if (order.Size() > 1) {
      cout << "Wrong number of orders set, needs one.\n";
    }
    fec = new H1_FECollection(abs(order[0]), dimension);
    own_fec = 1;
  }

  FiniteElementSpace *fespace = new FiniteElementSpace(mesh, NURBSext, fec);
  cout << "Number of finite element unknowns: " << fespace->GetTrueVSize()
       << endl;

  ////////
  // 3. Determine surfaces for Boundary conditions
  ////////
  // Check the total number of boundaries identified in the mesh file
  if (mesh->bdr_attributes.Max() <
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
  ConstantCoefficient matCoef(conductivity);
  ConstantCoefficient dbcCoef(dirichlet_bc_val);
  ConstantCoefficient neumann_flux_const(neumann_bc_value);

  // For Later:
  FunctionCoefficient neumann_flux(NeumannBoundaryProfile);

  ////////
  // 5. Define and init solution vector and bilinear form (operators)
  ////////
  // Define the solution vector u as a finite element grid function
  GridFunction u(fespace);
  // Initialize u with initial guess of zero.
  u = 0.0;
  // Set up the bilinear form a(.,.) on the finite element space corresponding
  // to the Laplacian operator -Delta, by adding the Diffusion domain
  // integrator.
  BilinearForm a(fespace);
  a.AddDomainIntegrator(new DiffusionIntegrator(matCoef));
  a.Assemble();
  // Assemble the linear form for the right hand side vector.
  LinearForm b(fespace);

  // Set the Dirichlet values in the solution vector
  u.ProjectBdrCoefficient(dbcCoef, is_Dirichlet_boundary);

  // Add the desired value for n.Grad(u) on the Neumann boundary
  b.AddBoundaryIntegrator(new BoundaryLFIntegrator(neumann_flux),
                          is_Neumann_boundary);
  // b.AddBoundaryIntegrator(new BoundaryLFIntegrator(neumann_flux_const),
  // is_Neumann_boundary);
  b.Assemble();

  ////////
  // 6. Construct the linear system
  ////////
  OperatorPtr A;
  Vector B, X;
  a.FormLinearSystem(ess_tdof_list, u, b, A, X, B);

#ifndef MFEM_USE_SUITESPARSE
  // Define a simple symmetric Gauss-Seidel preconditioner and use it to
  // solve the system AX=B with PCG in the symmetric case, and GMRES in the
  // non-symmetric one.
  {
    GSSmoother M((SparseMatrix &)(*A));
    if (sigma == -1.0) {
      PCG(*A, M, B, X, 1, 500, 1e-12, 0.0);
    } else {
      GMRES(*A, M, B, X, 1, 500, 10, 1e-12, 0.0);
    }
  }
#else
  // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the
  // system.
  UMFPackSolver umf_solver;
  umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
  umf_solver.SetOperator(*A);
  umf_solver.Mult(B, X);
#endif
  // Recover the grid function corresponding to U. This is the local finite
  // element solution.
  a.RecoverFEMSolution(X, b, u);

  ////////
  // 7. Save results into files
  ////////
  ofstream mesh_ofs("refined.mesh");
  mesh_ofs.precision(8);
  mesh->Print(mesh_ofs);
  ofstream sol_ofs("sol.gf");
  sol_ofs.precision(8);
  u.Save(sol_ofs);

  ////////
  // 7.1 Calculate and print out objective function value
  ////////
  double surface_area;
  const double obj_func_value =
      EvaluateObjectiveFunction(u,  // Solution Field (GridFunction)
                                is_Neumann_boundary,  // Boundary Bool Array
                                TargetTemperatureProfile, surface_area);

  cout << "Objective Function evaluated to be: " << obj_func_value << std::endl;
  // Write the objective function value into output file
  std::ofstream outputfilestream("objective.out");
  outputfilestream << obj_func_value;
  outputfilestream.close();

  //  ////////
  //  // 8. Determine errors on boundaries
  //  ////////
  //  if (check_boundary_integrals) {
  //    // Integrate the solution on the Dirichlet boundary and compare to the
  //    // expected value.
  //    double error, avg = IntegrateBC(u, is_Dirichlet_boundary, 0.0, 1.0,
  //    dirichlet_bc_val, error);
  //
  //    bool hom_dbc = (dirichlet_bc_val == 0.0);
  //    error /=  hom_dbc ? 1.0 : fabs(dirichlet_bc_val);
  //    mfem::out << "Average of solution on Gamma_dbc:\t"
  //              << avg << ", \t"
  //              << (hom_dbc ? "absolute" : "relative")
  //              << " error " << error << endl;
  //    // Integrate n.Grad(u) on the inhomogeneous Neumann boundary and compare
  //    // to the expected value.
  //    double error, avg = IntegrateBC(u, is_Neumann_boundary, 1.0, 0.0,
  //    neumann_bc_value, error);
  //
  //    bool hom_nbc = (neumann_bc_value == 0.0);
  //    error /=  hom_nbc ? 1.0 : fabs(neumann_bc_value);
  //    mfem::out << "Average of n.Grad(u) on Gamma_nbc:\t"
  //              << avg << ", \t"
  //              << (hom_nbc ? "absolute" : "relative")
  //              << " error " << error << endl;
  //   }
  //   {
  //      // Integrate n.Grad(u) on the homogeneous Neumann boundary and compare
  //      to
  //      // the expected value of zero.
  //      Array<int> nbc0_bdr(mesh->bdr_attributes.Max());
  //      nbc0_bdr = 0;
  //      nbc0_bdr[3] = 1;
  //
  //      double error, avg = IntegrateBC(u, nbc0_bdr, 1.0, 0.0, 0.0, error);
  //
  //      bool hom_nbc = true;
  //      mfem::out << "Average of n.Grad(u) on Gamma_nbc0:\t"
  //                << avg << ", \t"
  //                << (hom_nbc ? "absolute" : "relative")
  //                << " error " << error << endl;
  //   }
  //   {
  //      // Integrate n.Grad(u) + a * u on the Robin boundary and compare to
  //      the
  //      // expected value.
  //      double error;
  //      double avg = IntegrateBC(u, rbc_bdr, 1.0, rbc_a_val, rbc_b_val,
  //      error);
  //
  //      bool hom_rbc = (rbc_b_val == 0.0);
  //      error /=  hom_rbc ? 1.0 : fabs(rbc_b_val);
  //      mfem::out << "Average of n.Grad(u)+a*u on Gamma_rbc:\t"
  //                << avg << ", \t"
  //                << (hom_rbc ? "absolute" : "relative")
  //                << " error " << error << endl;
  //   }
  //

  ////////
  // 9. Free Memory
  ////////
  delete fec;
  delete mesh;

  return 0;
}

// Evaluate Objective function
double EvaluateObjectiveFunction(
    const GridFunction &x, const Array<int> &bdr,
    std::function<double(const IntegrationPoint &)> target_temp,
    double &surface_area) {
  // Init return value
  double obj_func_value{};

  // Retrieve FE information
  const FiniteElementSpace &fes = *x.FESpace();
  MFEM_ASSERT(fes.GetVDim() == 1, "");
  Mesh &mesh = *fes.GetMesh();
  Vector shape, loc_dofs;
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

    const FiniteElement &fe = *fes.GetFE(FTr->Elem1No);
    MFEM_ASSERT(fe.GetMapType() == FiniteElement::VALUE, "");
    const int int_order = 2 * fe.GetOrder() + 3;
    const IntegrationRule &ir = IntRules.Get(FTr->FaceGeom, int_order);

    fes.GetElementDofs(FTr->Elem1No, dof_ids);
    x.GetSubVector(dof_ids, loc_dofs);

    // Retrieve shape function values
    shape.SetSize(fe.GetDof());

    // Loop over Integration points
    for (int j = 0; j < ir.GetNPoints(); j++) {
      const IntegrationPoint &ip = ir.IntPoint(j);
      IntegrationPoint eip;
      FTr->Loc1.Transform(ip, eip);
      FTr->Face->SetIntPoint(&ip);
      double face_weight = FTr->Face->Weight();
      double val = 0.0;
      const double target_t_at_ir = target_temp(eip);
      fe.CalcShape(eip, shape);
      val += (shape * loc_dofs);

      // Measure the length of the boundary
      surface_area += ip.weight * face_weight;

      // Integrate |x - gamma|^2
      val -= target_t_at_ir;
      obj_func_value += (val * val) * ip.weight * face_weight;
    }
  }

  // Compute l2 norm of the error in the boundary condition (negative
  // quadrature weights may produce negative 'error')
  obj_func_value =
      (obj_func_value >= 0.0) ? sqrt(obj_func_value) : -sqrt(-obj_func_value);

  // Return the average value of alpha * n.Grad(x) + beta * x
  return obj_func_value;
}
