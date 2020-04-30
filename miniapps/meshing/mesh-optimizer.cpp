// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//            --------------------------------------------------
//            Mesh Optimizer Miniapp: Optimize high-order meshes
//            --------------------------------------------------
//
// This miniapp performs mesh optimization using the Target-Matrix Optimization
// Paradigm (TMOP) by P.Knupp et al., and a global variational minimization
// approach. It minimizes the quantity sum_T int_T mu(J(x)), where T are the
// target (ideal) elements, J is the Jacobian of the transformation from the
// target to the physical element, and mu is the mesh quality metric. This
// metric can measure shape, size or alignment of the region around each
// quadrature point. The combination of targets & quality metrics is used to
// optimize the physical node positions, i.e., they must be as close as possible
// to the shape / size / alignment of their targets. This code also demonstrates
// a possible use of nonlinear operators (the class TMOP_QualityMetric, defining
// mu(J), and the class TMOP_Integrator, defining int mu(J)), as well as their
// coupling to Newton methods for solving minimization problems. Note that the
// utilized Newton methods are oriented towards avoiding invalid meshes with
// negative Jacobian determinants. Each Newton step requires the inversion of a
// Jacobian matrix, which is done through an inner linear solver.
//
// Compile with: make mesh-optimizer
//
// Sample runs:
//   Adapted analytic Hessian:
//     mesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 2 -tid 4 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8
//   Adapted analytic Hessian with size+orientation:
//     mesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 14 -tid 4 -ni 100 -ls 2 -li 100 -bnd -qt 1 -qo 8 -fd 1
//   Adapted analytic Hessian with shape+size+orientation
//     mesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 87 -tid 4 -ni 100 -ls 2 -li 100 -bnd -qt 1 -qo 8 -fd 1
//   Adapted discrete size:
//     mesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 7 -tid 5 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8
//   Adapted size+aspect ratio to discrete material indicator
//     mesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 7 -tid 6 -ni 100  -ls 2 -li 100 -bnd -qt 1 -qo 8
//   Adapted discrete size+orientation
//     mesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 14 -tid 8 -ni 100  -ls 2 -li 100 -bnd -qt 1 -qo 8 -fd 1
//   Adapted discrete aspect-ratio+orientation
//     mesh-optimizer -m square01.mesh -o 2 -rs 2 -mid 87 -tid 8 -ni 100  -ls 2 -li 100 -bnd -qt 1 -qo 8 -fd 1
//   Adapted discrete aspect ratio (3D)
//     mesh-optimizer -m cube.mesh -o 2 -rs 0 -mid 302 -tid 7 -ni 20  -ls 2 -li 100 -bnd -qt 1 -qo 8

//   Blade shape:
//     mesh-optimizer -m blade.mesh -o 4 -rs 0 -mid 2 -tid 1 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8
//   Blade shape with FD-based solver:
//     mesh-optimizer -m blade.mesh -o 4 -rs 0 -mid 2 -tid 1 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8 -fd 1
//   Blade limited shape:
//     mesh-optimizer -m blade.mesh -o 4 -rs 0 -mid 2 -tid 1 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8 -lc 5000
//   ICF shape and equal size:
//     mesh-optimizer -o 3 -rs 0 -mid 9 -tid 2 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8
//   ICF shape and initial size:
//     mesh-optimizer -o 3 -rs 0 -mid 9 -tid 3 -ni 100 -ls 2 -li 100 -bnd -qt 1 -qo 8
//   ICF shape:
//     mesh-optimizer -o 3 -rs 0 -mid 1 -tid 1 -ni 100 -ls 2 -li 100 -bnd -qt 1 -qo 8
//   ICF limited shape:
//     mesh-optimizer -o 3 -rs 0 -mid 1 -tid 1 -ni 100 -ls 2 -li 100 -bnd -qt 1 -qo 8 -lc 10
//   ICF combo shape + size (rings, slow convergence):
//     mesh-optimizer -o 3 -rs 0 -mid 1 -tid 1 -ni 1000 -ls 2 -li 100 -bnd -qt 1 -qo 8 -cmb
//   3D pinched sphere shape (the mesh is in the mfem/data GitHub repository):
//   * mesh-optimizer -m ../../../mfem_data/ball-pert.mesh -o 4 -rs 0 -mid 303 -tid 1 -ni 20 -ls 2 -li 500 -fix-bnd
//   2D non-conforming shape and equal size:
//     mesh-optimizer -m ./amr-quad-q2.mesh -o 2 -rs 1 -mid 9 -tid 2 -ni 200 -ls 2 -li 100 -bnd -qt 1 -qo 8


#include "../../mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;


class HessianCoefficient : public MatrixCoefficient
{
private:
   int type;
   int typemod = 5;

public:
   HessianCoefficient(int dim, int type_)
      : MatrixCoefficient(dim), typemod(type_) { }

   virtual void SetType(int typemod_) { typemod = typemod_; }
   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      Vector pos(3);
      T.Transform(ip, pos);
      (this)->Eval(K,pos);
   }

   virtual void Eval(DenseMatrix &K)
   {
      Vector pos(3);
      for (int i=0; i<K.Size(); i++) {pos(i)=K(i,i);}
      (this)->Eval(K,pos);
   }

   virtual void Eval(DenseMatrix &K, Vector pos)
   {
      if (typemod == 0)
      {
         K(0, 0) = 1.0 + 3.0 * std::sin(M_PI*pos(0));
         K(0, 1) = 0.0;
         K(1, 0) = 0.0;
         K(1, 1) = 1.0;
      }
      else if (typemod==1) //size only circle
      {
         const double small = 0.001, big = 0.01;
         const double xc = pos(0) - 0.5, yc = pos(1) - 0.5;
         const double r = sqrt(xc*xc + yc*yc);
         double r1 = 0.15; double r2 = 0.35; double sf=30.0;
         const double eps = 0.5;

         const double tan1 = std::tanh(sf*(r-r1)),
                      tan2 = std::tanh(sf*(r-r2));

         double ind = (tan1 - tan2);
         if (ind > 1.0) {ind = 1.;}
         if (ind < 0.0) {ind = 0.;}
         double val = ind * small + (1.0 - ind) * big;
         //K(0, 0) = eps + 1.0 * (tan1 - tan2);
         K(0, 0) = 1.0;
         K(0, 1) = 0.0;
         K(1, 0) = 0.0;
         K(1, 1) = 1.0;
         K(0,0) *= pow(val,0.5);
         K(1,1) *= pow(val,0.5);
      }
      else if (typemod==2) // size only sine wave
      {
         const double small = 0.001, big = 0.01;
         const double X = pos(0), Y = pos(1);
         double ind = std::tanh((10*(Y-0.5) + std::sin(4.0*M_PI*X)) + 1) -
                      std::tanh((10*(Y-0.5) + std::sin(4.0*M_PI*X)) - 1);
         if (ind > 1.0) {ind = 1.;}
         if (ind < 0.0) {ind = 0.;}
         double val = ind * small + (1.0 - ind) * big;
         K(0, 0) = pow(val,0.5);
         K(0, 1) = 0.0;
         K(1, 0) = 0.0;
         K(1, 1) = pow(val,0.5);
      }
      else if (typemod==3) //circle with size and AR
      {
         const double small = 0.001, big = 0.01;
         const double xc = pos(0)-0.5, yc = pos(1)-0.5;
         const double rv = xc*xc + yc*yc;
         double r = 0;
         if (rv>0.) {r = sqrt(rv);}

         double r1 = 0.25; double r2 = 0.30; double sf=30.0;
         const double szfac = 1;
         const double asfac = 40;
         const double eps2 = szfac/asfac;
         const double eps1 = szfac;

         double tan1 = std::tanh(sf*(r-r1)+1),
                tan2 = std::tanh(sf*(r-r2)-1);
         double wgt = 0.5*(tan1-tan2);

         tan1 = std::tanh(sf*(r-r1)),
         tan2 = std::tanh(sf*(r-r2));

         double ind = (tan1 - tan2);
         if (ind > 1.0) {ind = 1.;}
         if (ind < 0.0) {ind = 0.;}
         double szval = ind * small + (1.0 - ind) * big;

         double th = std::atan2(yc,xc)*180./M_PI;
         if (wgt > 1) { wgt = 1; }
         if (wgt < 0) { wgt = 0; }

         double maxval = eps2 + eps1*(1-wgt)*(1-wgt);
         double minval = eps1;
         double avgval = 0.5*(maxval+minval);
         double ampval = 0.5*(maxval-minval);
         double val1 = avgval + ampval*sin(2.*th*M_PI/180.+90*M_PI/180.);
         double val2 = avgval + ampval*sin(2.*th*M_PI/180.-90*M_PI/180.);

         K(0,1) = 0.0;
         K(1,0) = 0.0;
         K(0,0) = val1;
         K(1,1) = val2;

         K(0,0) *= pow(szval,0.5);
         K(1,1) *= pow(szval,0.5);
      }
      else if (typemod == 4) //sharp sine wave
      {
         const double small = 0.001, big = 0.01;
         const double xc = pos(0), yc = pos(1);
         const double r = sqrt(xc*xc + yc*yc);

         double tfac = 40;
         double yl1 = 0.45;
         double yl2 = 0.55;
         double wgt = std::tanh((tfac*(yc-yl1) + 2*std::sin(4.0*M_PI*xc)) + 1) -
                      std::tanh((tfac*(yc-yl2) + 2*std::sin(4.0*M_PI*xc)) - 1);
         if (wgt > 1) { wgt = 1; }
         if (wgt < 0) { wgt = 0; }
         double szval = wgt * small + (1.0 - wgt) * big;

         const double eps2 = 20;
         const double eps1 = 1;
         K(1,1) = eps1/eps2 + eps1*(1-wgt)*(1-wgt);
         K(0,0) = eps1;
         K(0,1) = 0.0;
         K(1,0) = 0.0;

         //K(0,0) *= pow(szval,0.5);
         //K(1,1) *= pow(szval,0.5);
      }
      else if (typemod == 5) //sharp rotated sine wave
      {
         double xc = pos(0)-0.5, yc = pos(1)-0.5;
         double th = 15.5*M_PI/180.;
         double xn =  cos(th)*xc + sin(th)*yc;
         double yn = -sin(th)*xc + cos(th)*yc;
         double th2 = (th > 45.*M_PI/180) ? M_PI/2 - th : th;
         double stretch = 1/cos(th2);
         xc = xn/stretch;
         yc = yn;
         double tfac = 20;
         double s1 = 3;
         double s2 = 2;
         double yl1 = -0.025;
         double yl2 =  0.025;
         double wgt = std::tanh((tfac*(yc-yl1) + s2*std::sin(s1*M_PI*xc)) + 1) -
                      std::tanh((tfac*(yc-yl2) + s2*std::sin(s1*M_PI*xc)) - 1);
         if (wgt > 1) { wgt = 1; }
         if (wgt < 0) { wgt = 0; }

         const double eps2 = 20;
         const double eps1 = 1;
         K(1,1) = eps1/eps2 + eps1*(1-wgt)*(1-wgt);
         K(0,0) = eps1;
         K(0,1) = 0.0;
         K(1,0) = 0.0;
      }
      else if (typemod == 6) //BOUNDARY LAYER REFINEMENT
      {
         const double szfac = 1;
         const double asfac = 500;
         const double eps = szfac;
         const double eps2 = szfac/asfac;
         double yscale = 1.5;
         yscale = 2 - 2/asfac;
         double yval = 0.25;
         K(0, 0) = eps;
         K(1, 1) = eps2 + szfac*yscale*pos(1);
         K(0, 1) = 0.0;
         K(1, 0) = 0.0;
      }
   }
};

class TMOPEstimator : public ErrorEstimator
{
protected:
   long current_sequence;
   double total_error;
   Array<int> aniso_flags;

   FiniteElementSpace *fespace;
   MatrixCoefficient *target_spec;
   GridFunction *size, tarsize;
   GridFunction *aspr, taraspr;
   bool discrete_field_flag;

   Vector SizeErr, AspErr;

   /// Check if the mesh of the solution was modified.
   bool MeshIsModified()
   {
      long mesh_sequence = size->FESpace()->GetMesh()->GetSequence();
      MFEM_ASSERT(mesh_sequence >= current_sequence, "");
      return (mesh_sequence > current_sequence);
   }

   /// Compute the element error estimates.
   void ComputeEstimates();

public:
   TMOPEstimator(FiniteElementSpace &fes,
                 GridFunction &_size,
                 GridFunction &_aspr)
      : current_sequence(-1),
        total_error(0.),
        fespace(&fes),
        size(&_size),
        aspr(&_aspr),
        tarsize(),
        discrete_field_flag(true)  {}
   /// Return the total error from the last error estimate.
   double GetTotalError() const { return total_error; }

   void SetAnalyticTargetSpec(MatrixCoefficient *mspec)
   {target_spec = mspec; discrete_field_flag=false;}

   virtual const Vector &GetLocalErrors() { return SizeErr; }

   virtual const GridFunction &GetLocalSolution() { return tarsize; }

   virtual const Vector &GetSizeError()
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return SizeErr;
   }

   virtual const Vector &GetAsprError()
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return AspErr;
   }

   virtual void Reset() { current_sequence = -1; }

   virtual ~TMOPEstimator() {}

};

void TMOPEstimator::ComputeEstimates()
{
   // Compute error for each element
   Vector size_sol;
   Vector aspr_sol;
   const int NE = fespace->GetNE(),
             dim = fespace->GetMesh()->Dimension();

   GridFunction *nodes = fespace->GetMesh()->GetNodes();
   Vector nodesv(nodes->GetData(), nodes->Size());
   const int pnt_cnt = nodesv.Size()/dim;

   if (!discrete_field_flag)
   {
      DenseMatrix K; K.SetSize(dim);
      size_sol.SetSize(pnt_cnt);
      aspr_sol.SetSize(pnt_cnt);

      HessianCoefficient *target_spec_mod = dynamic_cast<HessianCoefficient *>
                                            (target_spec);

      for (int i = 0; i < pnt_cnt; i++)
      {
         for (int j = 0; j < dim; j++) { K(j,j) = nodesv(i+j*pnt_cnt); }
         target_spec_mod->Eval(K);
         Vector col1, col2;
         K.GetColumn(0, col1);
         K.GetColumn(1, col2);

         size_sol(i) = K.Det();
         aspr_sol(i) = col2.Norml2()/col1.Norml2(); // l2/l1 in 2D
      }
      size->SetDataAndSize(size_sol.GetData(),size_sol.Size());
      aspr->SetDataAndSize(aspr_sol.GetData(),aspr_sol.Size());
   }
   MFEM_ASSERT(size->Min() > 0,"Target element size should be greater than 0");
   MFEM_ASSERT(aspr->Min() > 0,
               "Target element aspect-ratio should be greater than 0");

   L2_FECollection avg_fec(0, fespace->GetMesh()->Dimension());
   FiniteElementSpace avg_fes(fespace->GetMesh(), &avg_fec);

   // Target and current Size
   tarsize.SetSpace(&avg_fes);
   size->GetElementAverages(tarsize);
   SizeErr.SetSize(NE);
   for (int i = 0; i < NE; i++)
   {
      double curr_size = fespace->GetMesh()->GetElementVolume(i);
      double tar_size  = tarsize(i);

      SizeErr(i) = curr_size/tar_size;
   }

   // Target AspectRatio
   taraspr.SetSpace(&avg_fes);
   AspErr.SetSize(NE);
   Vector pos0V(fespace->GetFE(0)->GetDof());
   Array<int> pos_dofs;
   for (int i = 0; i < NE; i++)
   {
      aspr->FESpace()->GetElementDofs(i, pos_dofs);
      aspr->GetSubVector(pos_dofs, pos0V);
      double prod = 1.;
      for (int j = 0; j < pos0V.Size(); j++)
      {
         prod *= pos0V(j);
      }
      taraspr(i) = pow(prod,1./pos0V.Size());
   }

   // Current AspectRatio
   Vector curr_aspr_vec(NE);
   const FiniteElement *fe;
   fe = fespace->GetFE(0);
   const IntegrationRule *ir = NULL;
   if (!ir)
   {
      ir = &(IntRules.Get(fe->GetGeomType(), 2*fe->GetOrder() + 3)); // <---
   }
   int dof = fe->GetDof();
   DenseMatrix Dsh, Jpr, PMatI;
   Dsh.SetSize(dof, dim), PMatI.SetSize(dof, dim), Jpr.SetSize(dim);
   Array<int> vdofs;

   for (int i = 0; i < NE; i++)
   {
      fe = fespace->GetFE(i);
      fespace->GetElementVDofs(i, vdofs);
      for (int j = 0; j < dof; j++)
      {
         int nodidx = vdofs[j];
         for (int k = 0; k < dim; k++)
         {
            PMatI(j,k) = nodesv(nodidx+k*pnt_cnt);
         }
      }
      double prod = 1;
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         fe->CalcDShape(ip, Dsh);
         MultAtB(PMatI, Dsh, Jpr);
         Vector col1, col2;
         Jpr.GetColumn(0, col1);
         Jpr.GetColumn(1, col2);
         prod *= col2.Norml2()/col1.Norml2();
      }
      prod = pow(prod,1./ir->GetNPoints());
      curr_aspr_vec(i) = prod;
   }

   for (int i = 0; i < NE; i++)
   {
      //double curr_aspr = fespace->GetMesh()->GetElementAspectRatio(i, 0);
      double curr_aspr = curr_aspr_vec(i);
      double tar_aspr  = taraspr(i);
      AspErr(i) = curr_aspr/tar_aspr;
//      std::cout << i << " "  << SizeErr(i) << " "
//                << AspErr(i) << " k10sizeasprerr\n";
   }

   current_sequence = size->FESpace()->GetMesh()->GetSequence();
}

class TMOPRefiner : public MeshOperator
{
protected:
   TMOPEstimator &estimator;

   long   max_elements;
   long num_marked_elements;

   Array<Refinement> marked_elements;
   long current_sequence;

   int non_conforming;
   int nc_limit;
   int amrmetric; //0-Size, 1-AspectRatio, 2-Size+AspectRatio
   int dim;

   double GetNorm(const Vector &local_err, Mesh &mesh) const;

   /** @brief Apply the operator to theG mesh->
       @return STOP if a stopping criterion is satisfied or no elements were
       marked for refinement; REFINED + CONTINUE otherwise. */
   virtual int ApplyImpl(Mesh &mesh);

public:
   /// Construct a ThresholdRefiner using the given ErrorEstimator.
   TMOPRefiner(TMOPEstimator &est, int amrmetric_, int dim_);

   // default destructor (virtual)

   /// Use nonconforming refinement, if possible (triangles, quads, hexes).
   void PreferNonconformingRefinement() { non_conforming = 1; }

   /** @brief Use conforming refinement, if possible (triangles, tetrahedra)
       -- this is the default. */
   void PreferConformingRefinement() { non_conforming = -1; }

   /** @brief Set the maximum ratio of refinement levels of adjacent elements
       (0 = unlimited). */
   void SetNCLimit(int nc_limit)
   {
      MFEM_ASSERT(nc_limit >= 0, "Invalid NC limit");
      this->nc_limit = nc_limit;
   }

   /// Get the number of marked elements in the last Apply() call.
   long GetNumMarkedElements() const { return num_marked_elements; }

   /// Reset the associated estimator.
   virtual void Reset();
};

TMOPRefiner::TMOPRefiner(TMOPEstimator &est, int amrmetric_, int dim_)
   : estimator(est), amrmetric(amrmetric_), dim(dim_)
{
   max_elements = std::numeric_limits<long>::max();

   num_marked_elements = 0L;
   current_sequence = -1;

   non_conforming = -1;
   nc_limit = 0;
}

int TMOPRefiner::ApplyImpl(Mesh &mesh)
{
   num_marked_elements = 0;
   marked_elements.SetSize(0);
   current_sequence = mesh.GetSequence();

   const long num_elements = mesh.GetGlobalNE();
   if (num_elements >= max_elements) { return STOP; }

   const int NE = mesh.GetNE();
   Vector SizeErr = estimator.GetSizeError();
   Vector AspErr = estimator.GetAsprError();
   MFEM_ASSERT(SizeErr.Size() == NE, "invalid size of local_err");

   int inum=0;
   for (int el = 0; el < NE; el++)
   {
      if (dim == 2)
      {
         if ( ( amrmetric == 1 ) || ( amrmetric == 2 && SizeErr(el) > 4./3))
         {
            if (AspErr(el)  < 2./3)
            {
               marked_elements.Append(Refinement(el));
               marked_elements[inum].ref_type = 1;
               inum += 1;
            }
            else if (AspErr(el) > 4./3)
            {
               marked_elements.Append(Refinement(el));
               marked_elements[inum].ref_type = 2;
               inum += 1;
            }
         }
         else if ( ( amrmetric == 0 || amrmetric == 2 ) && SizeErr(el) > 8./5)
         {
            marked_elements.Append(Refinement(el));
            marked_elements[inum].ref_type = 3;
            inum += 1;
         }
      }
      else
      {
         MFEM_ABORT(" dim=3 not implement yet");
      }
   }

   num_marked_elements = mesh.ReduceInt(marked_elements.Size());
   if (num_marked_elements == 0) { return STOP; }
   mesh.GeneralRefinement(marked_elements, non_conforming, nc_limit);
   return CONTINUE + REFINED;
}

void TMOPRefiner::Reset()
{
   estimator.Reset();
   current_sequence = -1;
   num_marked_elements = 0;
}



double weight_fun(const Vector &x);
void DiffuseField(GridFunction &field, int smooth_steps);

double discrete_size_2d(const Vector &x)
{
   const int opt = 2;
   const double small = 0.001, big = 0.01;
   double val = 0.;

   if (opt == 1) // sine wave.
   {
      const double X = x(0), Y = x(1);
      val = std::tanh((10*(Y-0.5) + std::sin(4.0*M_PI*X)) + 1) -
            std::tanh((10*(Y-0.5) + std::sin(4.0*M_PI*X)) - 1);
   }
   else if (opt == 2) // semi-circle
   {
      const double xc = x(0) - 0.0, yc = x(1) - 0.5;
      const double r = sqrt(xc*xc + yc*yc);
      double r1 = 0.45; double r2 = 0.55; double sf=30.0;
      val = 0.5*(1+std::tanh(sf*(r-r1))) - 0.5*(1+std::tanh(sf*(r-r2)));
   }

   val = std::max(0.,val);
   val = std::min(1.,val);

   return val * small + (1.0 - val) * big;
}

double material_indicator_2d(const Vector &x)
{
   double xc = x(0)-0.5, yc = x(1)-0.5;
   double th = 22.5*M_PI/180.;
   double xn =  cos(th)*xc + sin(th)*yc;
   double yn = -sin(th)*xc + cos(th)*yc;
   double th2 = (th > 45.*M_PI/180) ? M_PI/2 - th : th;
   double stretch = 1/cos(th2);
   xc = xn/stretch; yc = yn/stretch;
   double tfac = 20;
   double s1 = 3;
   double s2 = 3;
   double wgt = std::tanh((tfac*(yc) + s2*std::sin(s1*M_PI*xc)) + 1);
   if (wgt > 1) { wgt = 1; }
   if (wgt < 0) { wgt = 0; }
   return wgt;
}

double discrete_ori_2d(const Vector &x)
{
   return M_PI * x(1) * (1.0 - x(1)) * cos(2 * M_PI * x(0));
}

double discrete_aspr_2d(const Vector &x)
{
   double xc = x(0)-0.5, yc = x(1)-0.5;
   double th = 22.5*M_PI/180.;
   double xn =  cos(th)*xc + sin(th)*yc;
   double yn = -sin(th)*xc + cos(th)*yc;
   double th2 = (th > 45.*M_PI/180) ? M_PI/2 - th : th;
   double stretch = 1/cos(th2);
   xc = xn; yc = yn;

   double tfac = 20;
   double s1 = 3;
   double s2 = 2;
   double wgt = std::tanh((tfac*(yc) + s2*std::sin(s1*M_PI*xc)) + 1)
                - std::tanh((tfac*(yc) + s2*std::sin(s1*M_PI*xc)) - 1);
   if (wgt > 1) { wgt = 1; }
   if (wgt < 0) { wgt = 0; }
   return 0.1 + 1*(1-wgt)*(1-wgt);
}

void discrete_aspr_3d(const Vector &x, Vector &v)
{
   int dim = x.Size();
   v.SetSize(dim);
   double l1, l2, l3;
   l1 = 1.;
   l2 = 1. + 5*x(1);
   l3 = 1. + 10*x(2);
   v[0] = l1/pow(l2*l3,0.5);
   v[1] = l2/pow(l1*l3,0.5);
   v[2] = l3/pow(l2*l1,0.5);
}

void TMOPupdate(NonlinearForm &a, Mesh &mesh, FiniteElementSpace &fespace,
                bool move_bnd)
{
   int dim = fespace.GetFE(0)->GetDim();
   if (move_bnd == false)
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      a.SetEssentialBC(ess_bdr);
   }
   else
   {
      const int nd  = fespace.GetBE(0)->GetDof();
      int n = 0;
      for (int i = 0; i < mesh.GetNBE(); i++)
      {
         const int attr = mesh.GetBdrElement(i)->GetAttribute();
         MFEM_VERIFY(!(dim == 2 && attr == 3),
                     "Boundary attribute 3 must be used only for 3D meshes. "
                     "Adjust the attributes (1/2/3/4 for fixed x/y/z/all "
                     "components, rest for free nodes), or use -fix-bnd.");
         if (attr == 1 || attr == 2 || attr == 3) { n += nd; }
         if (attr == 4) { n += nd * dim; }
      }
      Array<int> ess_vdofs(n), vdofs;
      n = 0;
      for (int i = 0; i < mesh.GetNBE(); i++)
      {
         const int attr = mesh.GetBdrElement(i)->GetAttribute();
         fespace.GetBdrElementVDofs(i, vdofs);
         if (attr == 1) // Fix x components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j]; }
         }
         else if (attr == 2) // Fix y components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j+nd]; }
         }
         else if (attr == 3) // Fix z components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j+2*nd]; }
         }
         else if (attr == 4) // Fix all components.
         {
            for (int j = 0; j < vdofs.Size(); j++)
            { ess_vdofs[n++] = vdofs[j]; }
         }
      }
      a.SetEssentialVDofs(ess_vdofs);
   }
};

// Additional IntegrationRules that can be used with the --quad-type option.
IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);
IntegrationRules IntRulesCU(0, Quadrature1D::ClosedUniform);


int main(int argc, char *argv[])
{
   // 0. Set the method's default parameters.
   const char *mesh_file = "icf.mesh";
   int mesh_poly_deg     = 1;
   int rs_levels         = 0;
   double jitter         = 0.0;
   int metric_id         = 1;
   int target_id         = 1;
   double lim_const      = 0.0;
   int quad_type         = 1;
   int quad_order        = 8;
   int newton_iter       = 10;
   double newton_rtol    = 1e-10;
   int lin_solver        = 2;
   int max_lin_iter      = 100;
   bool move_bnd         = true;
   bool combomet         = 0;
   int amr_flag          = 1;
   int amrmetric         = 2;
   bool normalization    = false;
   bool visualization    = true;
   int verbosity_level   = 0;
   int hessiantype       = 1;
   int fdscheme          = 0;
   int adapt_eval        = 0;

   // 1. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&jitter, "-ji", "--jitter",
                  "Random perturbation scaling factor.");
   args.AddOption(&metric_id, "-mid", "--metric-id",
                  "Mesh optimization metric:\n\t"
                  "1  : |T|^2                          -- 2D shape\n\t"
                  "2  : 0.5|T|^2/tau-1                 -- 2D shape (condition number)\n\t"
                  "7  : |T-T^-t|^2                     -- 2D shape+size\n\t"
                  "9  : tau*|T-T^-t|^2                 -- 2D shape+size\n\t"
                  "14: 0.5*(1-cos(theta_A - theta_W)   -- 2D Sh+Sz+Alignment\n\t"
                  "22 : 0.5(|T|^2-2*tau)/(tau-tau_0)   -- 2D untangling\n\t"
                  "50 : 0.5|T^tT|^2/tau^2-1            -- 2D shape\n\t"
                  "55 : (tau-1)^2                      -- 2D size\n\t"
                  "56 : 0.5(sqrt(tau)-1/sqrt(tau))^2   -- 2D size\n\t"
                  "58 : |T^tT|^2/(tau^2)-2*|T|^2/tau+2 -- 2D shape\n\t"
                  "77 : 0.5(tau-1/tau)^2               -- 2D size\n\t"
                  "211: (tau-1)^2-tau+sqrt(tau^2)      -- 2D untangling\n\t"
                  "252: 0.5(tau-1)^2/(tau-tau_0)       -- 2D untangling\n\t"
                  "301: (|T||T^-1|)/3-1              -- 3D shape\n\t"
                  "302: (|T|^2|T^-1|^2)/9-1          -- 3D shape\n\t"
                  "303: (|T|^2)/3*tau^(2/3)-1        -- 3D shape\n\t"
                  "315: (tau-1)^2                    -- 3D size\n\t"
                  "316: 0.5(sqrt(tau)-1/sqrt(tau))^2 -- 3D size\n\t"
                  "321: |T-T^-t|^2                   -- 3D shape+size\n\t"
                  "352: 0.5(tau-1)^2/(tau-tau_0)     -- 3D untangling");
   args.AddOption(&target_id, "-tid", "--target-id",
                  "Target (ideal element) type:\n\t"
                  "1: Ideal shape, unit size\n\t"
                  "2: Ideal shape, equal size\n\t"
                  "3: Ideal shape, initial size\n\t"
                  "4: Given full analytic Jacobian (in physical space)\n\t"
                  "5: Ideal shape, given size (in physical space)");
   args.AddOption(&lim_const, "-lc", "--limit-const", "Limiting constant.");
   args.AddOption(&quad_type, "-qt", "--quad-type",
                  "Quadrature rule type:\n\t"
                  "1: Gauss-Lobatto\n\t"
                  "2: Gauss-Legendre\n\t"
                  "3: Closed uniform points");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
   args.AddOption(&newton_iter, "-ni", "--newton-iters",
                  "Maximum number of Newton iterations.");
   args.AddOption(&newton_rtol, "-rtol", "--newton-rel-tolerance",
                  "Relative tolerance for the Newton solver.");
   args.AddOption(&lin_solver, "-ls", "--lin-solver",
                  "Linear solver: 0 - l1-Jacobi, 1 - CG, 2 - MINRES.");
   args.AddOption(&max_lin_iter, "-li", "--lin-iter",
                  "Maximum number of iterations in the linear solve.");
   args.AddOption(&move_bnd, "-bnd", "--move-boundary", "-fix-bnd",
                  "--fix-boundary",
                  "Enable motion along horizontal and vertical boundaries.");
   args.AddOption(&combomet, "-cmb", "--combo-met", "-no-cmb", "--no-combo-met",
                  "Combination of metrics.");
   args.AddOption(&amr_flag, "-amr", "--amr-flag",
                  "1 - AMR after TMOP");
   args.AddOption(&amrmetric, "-amrm", "--amr-metric",
                  "0 - Size, 1 - AspectRatio, 2 - Size + AspectRatio");
   args.AddOption(&hessiantype, "-ht", "--Hessian Target type",
                  "1-6");
   args.AddOption(&normalization, "-nor", "--normalization", "-no-nor",
                  "--no-normalization",
                  "Make all terms in the optimization functional unitless.");
   args.AddOption(&fdscheme, "-fd", "--fd_approximation",
                  "Enable finite difference based derivative computations.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&verbosity_level, "-vl", "--verbosity-level",
                  "Set the verbosity level - 0, 1, or 2.");
   args.AddOption(&adapt_eval, "-ae", "--adaptivity evaluatior",
                  "0 - Advection based (DEFAULT), 1 - GSLIB.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Initialize and refine the starting mesh->
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   const int dim = mesh->Dimension();
   cout << "Mesh curvature: ";
   if (mesh->GetNodes()) { cout << mesh->GetNodes()->OwnFEC()->Name(); }
   else { cout << "(NONE)"; }
   cout << endl;

   // 3. Define a finite element space on the mesh-> Here we use vector finite
   //    elements which are tensor products of quadratic finite elements. The
   //    number of components in the vector finite element space is specified by
   //    the last parameter of the FiniteElementSpace constructor.
   H1_FECollection fec(mesh_poly_deg, dim);
   FiniteElementSpace fespace(mesh, &fec, dim);

   // 4. Make the mesh curved based on the above finite element space. This
   //    means that we define the mesh elements through a fespace-based
   //    transformation of the reference element.
   mesh->SetNodalFESpace(&fespace);

   // 5. Set up an empty right-hand side vector b, which is equivalent to b=0.
   Vector b(0);

   // 6. Get the mesh nodes (vertices and other degrees of freedom in the finite
   //    element space) as a finite element grid function in fespace. Note that
   //    changing x automatically changes the shapes of the mesh elements.
   GridFunction x(&fespace);
   GridFunction xnew(&fespace);
   GridFunction x0new(&fespace);
   mesh->SetNodalGridFunction(&x);

   // 7. Define a vector representing the minimal local mesh size in the mesh
   //    nodes. We index the nodes using the scalar version of the degrees of
   //    freedom in fespace. Note: this is partition-dependent.
   //
   //    In addition, compute average mesh size and total volume.
   Vector h0(fespace.GetNDofs());
   h0 = infinity();
   double volume = 0.0;
   Array<int> dofs;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      // Get the local scalar element degrees of freedom in dofs.
      fespace.GetElementDofs(i, dofs);
      // Adjust the value of h0 in dofs based on the local mesh size.
      const double hi = mesh->GetElementSize(i);
      for (int j = 0; j < dofs.Size(); j++)
      {
         h0(dofs[j]) = min(h0(dofs[j]), hi);
      }
      volume += mesh->GetElementVolume(i);
   }
   const double small_phys_size = pow(volume, 1.0 / dim) / 100.0;

   // 8. Add a random perturbation to the nodes in the interior of the domain.
   //    We define a random grid function of fespace and make sure that it is
   //    zero on the boundary and its values are locally of the order of h0.
   //    The latter is based on the DofToVDof() method which maps the scalar to
   //    the vector degrees of freedom in fespace.
   GridFunction rdm(&fespace);
   rdm.Randomize();
   rdm -= 0.25; // Shift to random values in [-0.5,0.5].
   rdm *= jitter;
   // Scale the random values to be of order of the local mesh size.
   for (int i = 0; i < fespace.GetNDofs(); i++)
   {
      for (int d = 0; d < dim; d++)
      {
         rdm(fespace.DofToVDof(i,d)) *= h0(i);
      }
   }
   Array<int> vdofs;
   for (int i = 0; i < fespace.GetNBE(); i++)
   {
      // Get the vector degrees of freedom in the boundary element.
      fespace.GetBdrElementVDofs(i, vdofs);
      // Set the boundary values to zero.
      for (int j = 0; j < vdofs.Size(); j++) { rdm(vdofs[j]) = 0.0; }
   }
   x -= rdm;
   x.SetTrueVector();
   x.SetFromTrueVector();

   // 9. Save the starting (prior to the optimization) mesh to a file. This
   //    output can be viewed later using GLVis: "glvis -m perturbed.mesh".
   {
      ofstream mesh_ofs("perturbed.mesh");
      mesh->Print(mesh_ofs);
   }

   // 10. Store the starting (prior to the optimization) positions.
   GridFunction x0(&fespace);
   x0 = x;

   // 11. Form the integrator that uses the chosen metric and target.
   double tauval = -0.1;
   TMOP_QualityMetric *metric = NULL;
   switch (metric_id)
   {
      case 1: metric = new TMOP_Metric_001; break;
      case 2: metric = new TMOP_Metric_002; break;
      case 7: metric = new TMOP_Metric_007; break;
      case 9: metric = new TMOP_Metric_009; break;
      case 14: metric = new TMOP_Metric_SSA2D; break;
      case 22: metric = new TMOP_Metric_022(tauval); break;
      case 50: metric = new TMOP_Metric_050; break;
      case 55: metric = new TMOP_Metric_055; break;
      case 56: metric = new TMOP_Metric_056; break;
      case 58: metric = new TMOP_Metric_058; break;
      case 77: metric = new TMOP_Metric_077; break;
      case 87: metric = new TMOP_Metric_SS2D; break;
      case 211: metric = new TMOP_Metric_211; break;
      case 252: metric = new TMOP_Metric_252(tauval); break;
      case 301: metric = new TMOP_Metric_301; break;
      case 302: metric = new TMOP_Metric_302; break;
      case 303: metric = new TMOP_Metric_303; break;
      case 315: metric = new TMOP_Metric_315; break;
      case 316: metric = new TMOP_Metric_316; break;
      case 321: metric = new TMOP_Metric_321; break;
      case 352: metric = new TMOP_Metric_352(tauval); break;
      default: cout << "Unknown metric_id: " << metric_id << endl; return 3;
   }
   TargetConstructor::TargetType target_t;
   TargetConstructor *target_c = NULL;
   HessianCoefficient *adapt_coeff = NULL;
   H1_FECollection ind_fec(mesh_poly_deg, dim);
   DiscreteAdaptTC *tcd = NULL;
   AnalyticAdaptTC *tca = NULL;
   FiniteElementSpace ind_fes(mesh, &ind_fec);
   FiniteElementSpace ind_fesv(mesh, &ind_fec, dim);
   GridFunction size(&ind_fes), aspr(&ind_fes), disc(&ind_fes), ori(&ind_fes);
   GridFunction aspr3d(&ind_fesv), size3d(&ind_fesv);
   switch (target_id)
   {
      case 1: target_t = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; break;
      case 2: target_t = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
      case 3: target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
      case 4: // Analytic
      {
         target_t = TargetConstructor::GIVEN_FULL;
         tca = new AnalyticAdaptTC(target_t);
         adapt_coeff = new HessianCoefficient(dim, hessiantype);
         tca->SetAnalyticTargetSpec(NULL, NULL, adapt_coeff);
         target_c = tca;
         break;
      }
      case 5: // Discrete size 2D
      {
         target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE;
         tcd = new DiscreteAdaptTC(target_t);
         if (adapt_eval == 0)
         {
            tcd->SetAdaptivityEvaluator(new AdvectorCG);
         }
         else
         {
#ifdef MFEM_USE_GSLIB
            tcd->SetAdaptivityEvaluator(new InterpolatorFP);
#endif
         }
         FunctionCoefficient ind_coeff(discrete_size_2d);
         size.ProjectCoefficient(ind_coeff);
         tcd->SetSerialDiscreteTargetSize(size);
         tcd->FinalizeSerialDiscreteTargetSpec();
         target_c = tcd;
         break;
      }
      case 6: // Discrete size + aspect ratio - 2D
      {
         GridFunction d_x(&ind_fes), d_y(&ind_fes);

         target_t = TargetConstructor::GIVEN_SHAPE_AND_SIZE;
         tcd = new DiscreteAdaptTC(target_t);
         FunctionCoefficient ind_coeff(material_indicator_2d);
         disc.ProjectCoefficient(ind_coeff);
         if (adapt_eval == 0)
         {
            tcd->SetAdaptivityEvaluator(new AdvectorCG);
         }
         else
         {
#ifdef MFEM_USE_GSLIB
            tcd->SetAdaptivityEvaluator(new InterpolatorFP);
#endif
         }

         //Diffuse the interface
         DiffuseField(disc,2);

         //Get  partials with respect to x and y of the grid function
         disc.GetDerivative(1,0,d_x);
         disc.GetDerivative(1,1,d_y);

         //Compute the squared magnitude of the gradient
         for (int i = 0; i < size.Size(); i++)
         {
            size(i) = std::pow(d_x(i),2)+std::pow(d_y(i),2);
         }
         const double max = size.Max();

         for (int i = 0; i < d_x.Size(); i++)
         {
            d_x(i) = std::abs(d_x(i));
            d_y(i) = std::abs(d_y(i));
         }
         const double eps = 0.01;
         const double ratio = 20.0;
         const double big_small_ratio = 40.0;

         for (int i = 0; i < size.Size(); i++)
         {
            size(i) = (size(i)/max);
            aspr(i) = (d_x(i)+eps)/(d_y(i)+eps);
            aspr(i) = 0.1 + 0.9*(1-size(i))*(1-size(i));
            if (aspr(i) > ratio) {aspr(i) = ratio;}
            if (aspr(i) < 1.0/ratio) {aspr(i) = 1.0/ratio;}
         }
         Vector vals;
         const int NE = mesh->GetNE();
         double volume = 0.0, volume_ind = 0.0;

         for (int i = 0; i < NE; i++)
         {
            ElementTransformation *Tr = mesh->GetElementTransformation(i);
            const IntegrationRule &ir =
               IntRules.Get(mesh->GetElementBaseGeometry(i), Tr->OrderJ());
            size.GetValues(i, ir, vals);
            for (int j = 0; j < ir.GetNPoints(); j++)
            {
               const IntegrationPoint &ip = ir.IntPoint(j);
               Tr->SetIntPoint(&ip);
               volume     += ip.weight * Tr->Weight();
               volume_ind += vals(j) * ip.weight * Tr->Weight();
            }
         }

         const double avg_zone_size = volume / NE;

         const double small_avg_ratio = (volume_ind + (volume - volume_ind) /
                                         big_small_ratio) /
                                        volume;

         const double small_zone_size = small_avg_ratio * avg_zone_size;
         const double big_zone_size   = big_small_ratio * small_zone_size;

         for (int i = 0; i < size.Size(); i++)
         {
            const double val = size(i);
            const double a = (big_zone_size - small_zone_size) / small_zone_size;
            size(i) = big_zone_size / (1.0+a*val);
         }


         DiffuseField(size, 2);
         DiffuseField(aspr, 2);

         tcd->SetSerialDiscreteTargetSize(size);
         tcd->SetSerialDiscreteTargetAspectRatio(aspr);
         tcd->FinalizeSerialDiscreteTargetSpec();
         target_c = tcd;
         break;
      }
      case 7: // Discrete aspect ratio 3D
      {
         target_t = TargetConstructor::GIVEN_SHAPE_AND_SIZE;
         tcd = new DiscreteAdaptTC(target_t);
         if (adapt_eval == 0)
         {
            tcd->SetAdaptivityEvaluator(new AdvectorCG);
         }
         else
         {
#ifdef MFEM_USE_GSLIB
            tcd->SetAdaptivityEvaluator(new InterpolatorFP);
#endif
         }
         VectorFunctionCoefficient fd_aspr3d(dim, discrete_aspr_3d);
         aspr3d.ProjectCoefficient(fd_aspr3d);

         tcd->SetSerialDiscreteTargetAspectRatio(aspr3d);
         tcd->FinalizeSerialDiscreteTargetSpec();
         target_c = tcd;
         break;
      }
      case 8: // shape/size + orientation 2D
      {
         target_t = TargetConstructor::GIVEN_SHAPE_AND_SIZE;
         tcd = new DiscreteAdaptTC(target_t);
         if (adapt_eval == 0)
         {
            tcd->SetAdaptivityEvaluator(new AdvectorCG);
         }
         else
         {
#ifdef MFEM_USE_GSLIB
            tcd->SetAdaptivityEvaluator(new InterpolatorFP);
#endif
         }

         if (metric_id == 14)
         {
            ConstantCoefficient ind_coeff(0.1*0.1);
            size.ProjectCoefficient(ind_coeff);
            tcd->SetSerialDiscreteTargetSize(size);
         }

         if (metric_id == 87)
         {
            FunctionCoefficient aspr_coeff(discrete_aspr_2d);
            aspr.ProjectCoefficient(aspr_coeff);
            DiffuseField(aspr,2);
            tcd->SetSerialDiscreteTargetAspectRatio(aspr);
         }

         FunctionCoefficient ori_coeff(discrete_ori_2d);
         ori.ProjectCoefficient(ori_coeff);
         tcd->SetSerialDiscreteTargetOrientation(ori);
         tcd->FinalizeSerialDiscreteTargetSpec();
         target_c = tcd;
         break;
      }
      default: cout << "Unknown target_id: " << target_id << endl; return 3;
   }
   if (target_c == NULL)
   {
      target_c = new TargetConstructor(target_t);
   }
   target_c->SetNodes(x0);
   TMOP_Integrator *he_nlf_integ = new TMOP_Integrator(metric, target_c);
   if (fdscheme) { he_nlf_integ->EnableFiniteDifferences(x); }

   // 12. Setup the quadrature rule for the non-linear form integrator.
   const IntegrationRule *ir = NULL;
   const int geom_type = fespace.GetFE(0)->GetGeomType();
   switch (quad_type)
   {
      case 1: ir = &IntRulesLo.Get(geom_type, quad_order); break;
      case 2: ir = &IntRules.Get(geom_type, quad_order); break;
      case 3: ir = &IntRulesCU.Get(geom_type, quad_order); break;
      default: cout << "Unknown quad_type: " << quad_type << endl;
         delete he_nlf_integ; return 3;
   }
   cout << "Quadrature points per cell: " << ir->GetNPoints() << endl;
   he_nlf_integ->SetIntegrationRule(*ir);

   if (normalization) { he_nlf_integ->EnableNormalization(x0); }

   // 13. Limit the node movement.
   // The limiting distances can be given by a general function of space.
   GridFunction dist(&fespace);
   dist = 1.0;
   // The small_phys_size is relevant only with proper normalization.
   if (normalization) { dist = small_phys_size; }
   ConstantCoefficient lim_coeff(lim_const);
   if (lim_const != 0.0) { he_nlf_integ->EnableLimiting(x0, dist, lim_coeff); }

   // 14. Setup the final NonlinearForm (which defines the integral of interest,
   //     its first and second derivatives). Here we can use a combination of
   //     metrics, i.e., optimize the sum of two integrals, where both are
   //     scaled by used-defined space-dependent weights. Note that there are no
   //     command-line options for the weights and the type of the second
   //     metric; one should update those in the code.
   NonlinearForm a(&fespace);
   ConstantCoefficient *coeff1 = NULL;
   TMOP_QualityMetric *metric2 = NULL;
   TargetConstructor *target_c2 = NULL;
   FunctionCoefficient coeff2(weight_fun);

   if (combomet == 1)
   {
      // TODO normalization of combinations.
      // We will probably drop this example and replace it with adaptivity.
      if (normalization) { MFEM_ABORT("Not implemented."); }

      // Weight of the original metric.
      coeff1 = new ConstantCoefficient(1.0);
      he_nlf_integ->SetCoefficient(*coeff1);
      a.AddDomainIntegrator(he_nlf_integ);

      // Second metric.
      metric2 = new TMOP_Metric_077;
      target_c2 = new TargetConstructor(
         TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE);
      target_c2->SetVolumeScale(0.01);
      target_c2->SetNodes(x0);
      TMOP_Integrator *he_nlf_integ2 = new TMOP_Integrator(metric2, target_c2);
      he_nlf_integ2->SetIntegrationRule(*ir);
      if (fdscheme) { he_nlf_integ2->EnableFiniteDifferences(x); }

      // Weight of metric2.
      he_nlf_integ2->SetCoefficient(coeff2);
      a.AddDomainIntegrator(he_nlf_integ2);
   }
   else { a.AddDomainIntegrator(he_nlf_integ); }

   const double init_energy = a.GetGridFunctionEnergy(x);

   // 15. Visualize the starting mesh and metric values.
   if (visualization)
   {
      char title[] = "Initial metric values";
      vis_tmop_metric_s(mesh_poly_deg, *metric, *target_c, *mesh, title, 0);
   }

   // 16. Fix all boundary nodes, or fix only a given component depending on the
   //     boundary attributes of the given mesh-> Attributes 1/2/3 correspond to
   //     fixed x/y/z components of the node. Attribute 4 corresponds to an
   //     entirely fixed node. Other boundary attributes do not affect the node
   //     movement boundary conditions.
   if (move_bnd == false)
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      a.SetEssentialBC(ess_bdr);
   }
   else
   {
      const int nd  = fespace.GetBE(0)->GetDof();
      int n = 0;
      for (int i = 0; i < mesh->GetNBE(); i++)
      {
         const int attr = mesh->GetBdrElement(i)->GetAttribute();
         MFEM_VERIFY(!(dim == 2 && attr == 3),
                     "Boundary attribute 3 must be used only for 3D meshes. "
                     "Adjust the attributes (1/2/3/4 for fixed x/y/z/all "
                     "components, rest for free nodes), or use -fix-bnd.");
         if (attr == 1 || attr == 2 || attr == 3) { n += nd; }
         if (attr == 4) { n += nd * dim; }
      }
      Array<int> ess_vdofs(n), vdofs;
      n = 0;
      for (int i = 0; i < mesh->GetNBE(); i++)
      {
         const int attr = mesh->GetBdrElement(i)->GetAttribute();
         fespace.GetBdrElementVDofs(i, vdofs);
         if (attr == 1) // Fix x components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j]; }
         }
         else if (attr == 2) // Fix y components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j+nd]; }
         }
         else if (attr == 3) // Fix z components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j+2*nd]; }
         }
         else if (attr == 4) // Fix all components.
         {
            for (int j = 0; j < vdofs.Size(); j++)
            { ess_vdofs[n++] = vdofs[j]; }
         }
      }
      a.SetEssentialVDofs(ess_vdofs);
   }

   // 17. As we use the Newton method to solve the resulting nonlinear system,
   //     here we setup the linear solver for the system's Jacobian.
   Solver *S = NULL;
   const double linsol_rtol = 1e-12;
   if (lin_solver == 0)
   {
      S = new DSmoother(1, 1.0, max_lin_iter);
   }
   else if (lin_solver == 1)
   {
      CGSolver *cg = new CGSolver;
      cg->SetMaxIter(max_lin_iter);
      cg->SetRelTol(linsol_rtol);
      cg->SetAbsTol(0.0);
      cg->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);
      S = cg;
   }
   else
   {
      MINRESSolver *minres = new MINRESSolver;
      minres->SetMaxIter(max_lin_iter);
      minres->SetRelTol(linsol_rtol);
      minres->SetAbsTol(0.0);
      minres->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);
      S = minres;
   }

   // 18. Compute the minimum det(J) of the starting mesh->
   tauval = infinity();
   const int NE = mesh->GetNE();
   for (int i = 0; i < NE; i++)
   {
      ElementTransformation *transf = mesh->GetElementTransformation(i);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir->IntPoint(j));
         tauval = min(tauval, transf->Jacobian().Det());
      }
   }
   cout << "Minimum det(J) of the original mesh is " << tauval << endl;

   // 19. Finally, perform the nonlinear optimization.
   NewtonSolver *newton = NULL;
   if (tauval > 0.0)
   {
      tauval = 0.0;
      TMOPNewtonSolver *tns = new TMOPNewtonSolver(*ir);
      newton = tns;
      cout << "TMOPNewtonSolver is used (as all det(J) > 0).\n";
   }
   else
   {
      if ( (dim == 2 && metric_id != 22 && metric_id != 252) ||
           (dim == 3 && metric_id != 352) )
      {
         cout << "The mesh is inverted. Use an untangling metric." << endl;
         return 3;
      }
      tauval -= 0.01 * h0.Min(); // Slightly below minJ0 to avoid div by 0.
      newton = new TMOPDescentNewtonSolver(*ir);
      cout << "The TMOPDescentNewtonSolver is used (as some det(J) < 0).\n";
   }
   newton->SetPreconditioner(*S);
   newton->SetMaxIter(newton_iter);
   newton->SetRelTol(newton_rtol);
   newton->SetAbsTol(0.0);
   newton->SetPrintLevel(verbosity_level >= 1 ? 1 : -1);

   // 20. AMR based size refinemenet if a size metric is used
   TMOPEstimator tmope(ind_fes, size, aspr);
   if (target_id == 4) { tmope.SetAnalyticTargetSpec(adapt_coeff); }
   TMOPRefiner tmopr(tmope, amrmetric, dim);
   int newtonstop = 0;

   if (amr_flag==1)
   {
      int ni_limit = 3; //Newton + AMR
      int nic_limit = std::max(ni_limit, 4); //Number of iterations with AMR
      int amrstop = 0;
      int nc_limit = 1; //AMR per iteration - FIXED FOR NOW

      tmopr.PreferNonconformingRefinement();
      tmopr.SetNCLimit(nc_limit);

      for (int it = 0; it<ni_limit; it++)
      {

         std::cout << it << " Begin NEWTON+AMR Iteration\n";

         newton->SetOperator(a);
         newton->Mult(b, x.GetTrueVector());
         x.SetFromTrueVector();
         if (newton->GetConverged() == false)
         {
            cout << "NewtonIteration: rtol = " << newton_rtol << " not achieved."
                 << endl;
         }
         if (amrstop==1)
         {
            newtonstop = 1;
            cout << it << " Newton and AMR have converged" << endl;
            break;
         }
         char title1[10];
         sprintf(title1, "%s %d","Newton", it);

         for (int amrit=0; amrit<nc_limit; amrit++)
         {
            // need to remap discrete functions from old mesh to new mesh here
             if (target_id > 4)
             {
                 tcd->GetSerialDiscreteTargetSize(size);
                 tcd->GetSerialDiscreteTargetAspectRatio(aspr);
                 tcd->ResetDiscreteFields();
             }

            tmopr.Reset();
            if (nc_limit!=0 && amrstop==0) {tmopr.Apply(*mesh);}
            //Update stuff
            ind_fes.Update(); fespace.Update();
            size.Update();    aspr.Update();
            x.Update();       x.SetTrueVector();
            x0.Update();      x0.SetTrueVector();
            if (target_id > 4)
            {
               tcd->SetAdaptivityEvaluator(new InterpolatorFP);
               tcd->SetSerialDiscreteTargetSize(size);
               tcd->SetSerialDiscreteTargetAspectRatio(aspr);
               tcd->FinalizeSerialDiscreteTargetSpec();
               target_c = tcd;
               he_nlf_integ->UpdateTargetConstructor(target_c);
            }
            a.Update();
            //MFEM_ABORT(" ");
            TMOPupdate(a, *mesh, fespace, move_bnd);
            if (amrstop==0)
            {
               if (tmopr.Stop())
               {
                  newtonstop = 1;
                  amrstop = 1;
                  cout << it << " " << amrit <<
                       " AMR stopping criterion satisfied. Stop." << endl;
               }
               else
               {std::cout << mesh->GetNE() << " Number of elements after AMR\n";}
            }
         }
         if (it==nic_limit-1) { amrstop=1; }

         sprintf(title1, "%s %d","AMR", it);
      } //ni_limit
   } //amr_flag==1
   if (newtonstop == 0)
   {
      newton->SetOperator(a);
      newton->Mult(b, x.GetTrueVector());
      x.SetFromTrueVector();
   }

   // 21. Save the optimized mesh to a file. This output can be viewed later
   //     using GLVis: "glvis -m optimized.mesh".
   {
      ofstream mesh_ofs("optimized.mesh");
      mesh_ofs.precision(14);
      mesh->Print(mesh_ofs);
   }
   string namefile;
   char numstr[1]; // enough to hold all numbers up to 64-bits
   sprintf(numstr, "%s%d%s", "optimized_ht_", hessiantype, ".mesh");
   {
      ofstream mesh_ofs(numstr);
      mesh_ofs.precision(14);
      mesh->Print(mesh_ofs);
   }

   // 22. Compute the amount of energy decrease.
   const double fin_energy = a.GetGridFunctionEnergy(x);
   double metric_part = fin_energy;
   if (lim_const != 0.0)
   {
      lim_coeff.constant = 0.0;
      metric_part = a.GetGridFunctionEnergy(x);
      lim_coeff.constant = lim_const;
   }
   cout << "Initial strain energy: " << init_energy
        << " = metrics: " << init_energy
        << " + limiting term: " << 0.0 << endl;
   cout << "  Final strain energy: " << fin_energy
        << " = metrics: " << metric_part
        << " + limiting term: " << fin_energy - metric_part << endl;
   cout << "The strain energy decreased by: " << setprecision(12)
        << (init_energy - fin_energy) * 100.0 / init_energy << " %." << endl;

   // 22. Visualize the final mesh and metric values.
   if (visualization)
   {
      char title[] = "Final metric values";
      vis_tmop_metric_s(mesh_poly_deg, *metric, *target_c, *mesh, title, 600);
   }

   // 23. Visualize the mesh displacement.
   if (visualization)
   {
      osockstream sock(19916, "localhost");
      sock << "solution\n";
      mesh->Print(sock);
      x0 -= x;
      x0.Save(sock);
      sock.send();
      sock << "window_title 'Displacements'\n"
           << "window_geometry "
           << 1200 << " " << 0 << " " << 600 << " " << 600 << "\n"
           << "keys jRmclA" << endl;
   }

   // 24. Free the used memory.

   delete newton;
   delete S;
   delete target_c2;
   delete metric2;
   delete coeff1;
   delete target_c;
   delete adapt_coeff;
   delete metric;

   return 0;
}

// Defined with respect to the icf mesh->
double weight_fun(const Vector &x)
{
   const double r = sqrt(x(0)*x(0) + x(1)*x(1) + 1e-12);
   const double den = 0.002;
   double l2 = 0.2 + 0.5*std::tanh((r-0.16)/den) - 0.5*std::tanh((r-0.17)/den)
               + 0.5*std::tanh((r-0.23)/den) - 0.5*std::tanh((r-0.24)/den);
   return l2;
}

void DiffuseField(GridFunction &field, int smooth_steps)
{
   //Setup the Laplacian operator
   BilinearForm *Lap = new BilinearForm(field.FESpace());
   Lap->AddDomainIntegrator(new DiffusionIntegrator());
   Lap->Assemble();
   Lap->Finalize();

   //Setup the smoothing operator
   DSmoother *S = new DSmoother(0,1.0,smooth_steps);
   S->iterative_mode = true;
   S->SetOperator(Lap->SpMat());

   Vector tmp(field.Size());
   tmp = 0.0;
   S->Mult(tmp, field);

   delete S;
   delete Lap;
}
