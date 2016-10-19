#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <complex>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <sys/time.h>

#include <face_quda.h>

#include <iostream>
#include <sstream>

namespace quda {

  BiCGstabL::BiCGstabL(DiracMatrix &mat, DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat), matSloppy(matSloppy), nKrylov(param.Nkrylov), init(false)
  {
    r.resize(nKrylov+1);
    u.resize(nKrylov+1);
    
    gamma = new Complex[nKrylov+1];
    gamma_prime = new Complex[nKrylov+1];
    gamma_prime_prime = new Complex[nKrylov+1];
    sigma = new double[nKrylov+1];
    
    tau = new Complex*[nKrylov+1];
    for (int i = 0; i < nKrylov+1; i++) { tau[i] = new Complex[nKrylov+1]; }
    
    std::stringstream ss;
    ss << "BiCGstab-" << nKrylov;
    solver_name = ss.str();
  }

  BiCGstabL::~BiCGstabL() {
    profile.TPSTART(QUDA_PROFILE_FREE);
    delete[] gamma;
    delete[] gamma_prime;
    delete[] gamma_prime_prime;
    delete[] sigma;
    
    for (int i = 0; i < nKrylov+1; i++) { delete[] tau[i]; }
    delete[] tau; 
    
    if (init) {
      delete r_sloppy_saved_p; 
      delete u[0];
      for (int i = 1; i < nKrylov+1; i++) {
        delete r[i];
        delete u[i];
      }
      
      delete x_sloppy_saved_p; 
      delete r_fullp;
      delete r0_saved_p;
      delete yp;
      delete tempp; 
      
      init = false;
    }
    
    profile.TPSTOP(QUDA_PROFILE_FREE);
    
  }
  
  // Code to check for reliable updates, copied from inv_bicgstab_quda.cpp
  // Technically, there are ways to check both 'x' and 'r' for reliable updates...
  // the current status in BiCGstab is to just look for reliable updates in 'r'.
  int BiCGstabL::reliable(double &rNorm, double &maxrx, double &maxrr, const double &r2, const double &delta) {
    // reliable updates
    rNorm = sqrt(r2);
    if (rNorm > maxrx) maxrx = rNorm;
    if (rNorm > maxrr) maxrr = rNorm;
    //int updateR = (rNorm < delta*maxrr && r0Norm <= maxrr) ? 1 : 0;
    //int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0
    int updateR = (rNorm < delta*maxrr) ? 1 : 0;
    
    //printf("reliable %d %e %e %e %e\n", updateR, rNorm, maxrx, maxrr, r2);

    return updateR;
  }

  void BiCGstabL::operator()(ColorSpinorField &x, ColorSpinorField &b) 
  {
    // BiCGstab-l is based on the algorithm outlined in
    // BICGSTAB(L) FOR LINEAR EQUATIONS INVOLVING UNSYMMETRIC MATRICES WITH COMPLEX SPECTRUM
    // G. Sleijpen, D. Fokkema, 1993.
    // My implementation is based on Kate Clark's implementation in CPS, to be found in
    // src/util/dirac_op/d_op_wilson_types/bicgstab.C
    
    // Begin profiling preamble.
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);
    
    if (!init) {
      // Initialize fields.
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      
      // Full precision variables.
      r_fullp = ColorSpinorField::Create(csParam);
      
      // Create temporary.
      yp = ColorSpinorField::Create(csParam);
      
      // Sloppy precision variables.
      csParam.setPrecision(param.precision_sloppy); 
      
      // Sloppy solution.
      x_sloppy_saved_p = ColorSpinorField::Create(csParam); // Used depending on precision.
      
      // Shadow residual.
      r0_saved_p = ColorSpinorField::Create(csParam); // Used depending on precision. 
      
      // Temporary
      tempp = ColorSpinorField::Create(csParam); 
      
      // Residual (+ extra residuals for BiCG steps), Search directions.
      // Remark: search directions are sloppy in GCR. I wonder if we can
      //           get away with that here.
      for (int i = 0; i <= nKrylov; i++) {
        r[i] = ColorSpinorField::Create(csParam);
        u[i] = ColorSpinorField::Create(csParam);
      }
      r_sloppy_saved_p = r[0]; // Used depending on precision. 
      
      init = true; 
    }
    
    double b2 = blas::norm2(b); // norm sq of source.
    double r2;                  // norm sq of residual
    
    ColorSpinorField &r_full = *r_fullp;
    ColorSpinorField &y = *yp;
    ColorSpinorField &temp = *tempp;
    
    ColorSpinorField *r0p, *x_sloppyp; // Get assigned below. 
    
    // Compute initial residual depending on whether we have an initial guess or not.
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      mat(r_full, x, y); // r[0] = Ax
      r2 = blas::xmyNorm(b, r_full); // r = b - Ax, return norm.
      blas::copy(y, x);
    } else {
      blas::copy(r_full, b); // r[0] = b
      r2 = b2;
      blas::zero(x); // defensive measure in case solution isn't already zero
      blas::zero(y);
    }
    
    // Check to see that we're not trying to invert on a zero-field source
    if (b2 == 0) {
      if (param.compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_NO) {
        warningQuda("inverting on zero-field source");
        x = b;
        param.true_res = 0.0;
        param.true_res_hq = 0.0;
	profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
        return;
      } else if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
        printfQuda("BiCGstab: Computing null vector\n");
        b2 = r2;
      } else {
        errorQuda("Null vector computing requires non-zero guess!");
      }
    }
    
    
    
    // Set field aliasing according to whether we're doing mixed precision or not.
    // There probably be bugs and headaches hiding here. 
    if (param.precision_sloppy == x.Precision()) {
      r[0] = &r_full; // r[0] \equiv r_sloppy points to the same memory location as r.
      if (param.compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_NO)
      {
        r0p = &b; // r0, b point to the same vector in memory.
      }
      else
      {
        r0p = r0_saved_p; // r0p points to the saved r0 memory.
        *r0p = r_full; // and is set equal to r.
      }
    }
    else
    {
      r0p = r0_saved_p; // r0p points to saved r0 memory.
      r[0] = r_sloppy_saved_p; // r[0] points to saved r_sloppy memory.
      *r0p = r_full; // and is set equal to r.
      *r[0] = r_full; // yup.
    }
    
    if (param.precision_sloppy == x.Precision() || !param.use_sloppy_partial_accumulator) 
    {
      x_sloppyp = &x; // x_sloppy and x point to the same vector in memory.
      blas::zero(*x_sloppyp); // x_sloppy is zeroed out (and, by extension, so is x).
    }
    else
    {
      x_sloppyp = x_sloppy_saved_p; // x_sloppy point to saved x_sloppy memory.
      blas::zero(*x_sloppyp); // and is zeroed out. 
    }
    
    // Syntatic sugar.
    ColorSpinorField &r0 = *r0p;
    ColorSpinorField &x_sloppy = *x_sloppyp;
    
    // Zero out the first search direction. 
    blas::zero(*u[0]);
    
    
    // Set some initial values.
    sigma[0] = blas::norm2(r_full);
    

    // Initialize values.
    for (int i = 1; i <= nKrylov; i++)
    {
      blas::zero(*r[i]);
    }
    
    rho0 = 1.0;
    alpha = 0.0;
    omega = 1.0;
    
    double stop = stopping(param.tol, b2, param.residual_type); // stopping condition of solver.
    
    // While I don't have heavy quark checks implemented yet, here's a start...
    //const bool use_heavy_quark_res = 
    //  (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
    
    blas::flops = 0;
    //bool l2_converge = false;
    //double r2_old = r2;
    
    // done with preamble, begin computing.
    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    
    // count iteration counts
    int k = 0; 
    
    // Various variables related to reliable updates.
    int rUpdate = 0; // count reliable updates. 
    double delta = param.delta; // delta for reliable updates. 
    double rNorm = sqrt(r2); // The current residual norm. 
    double maxrr = rNorm; // The maximum residual norm since the last reliable update.
    double maxrx = rNorm; // The same. Would be different if we did 'x' reliable updates.
    
    PrintStats(solver_name.c_str(), k, r2, b2, 0.0); // 0.0 is heavy quark residual.
    while(!convergence(r2, 0.0, stop, 0.0) && k < param.maxiter) {

      //PrintStats("BiCGstab-l", k, r2, b2, 0.0);
      
      // rho0 = -omega*rho0;
      rho0 *= -omega;
      
      // BiCG part of calculation.
      for (int j = 0; j < nKrylov; j++) {
        // rho1 = <r0, r_j>, beta = alpha*rho1/rho0, rho0 = rho1;
        rho1 = blas::cDotProduct(r0, *r[j]);
        beta = alpha*rho1/rho0;
        rho0 = rho1;
        
        // for i = 0 .. j, u[i] = r[i] - beta*u[i]
        for (int i = 0; i <= j; i++)
        {
          // could use block blas
		      blas::caxpby(1.0, *r[i], -beta, *u[i]);
        }
        
        // u[j+1] = A ( u[j] )
        matSloppy(*u[j+1], *u[j], temp);
        
        // alpha = rho0/<r0, u[j+1]>
        alpha = rho0/blas::cDotProduct(r0, *u[j+1]);
        
        // for i = 0 .. j, r[i] = r[i] - alpha u[i+1]
        for (int i = 0; i <= j; i++)
        { 
          blas::caxpy(-alpha, *u[i+1], *r[i]);
        }
        
        // r[j+1] = A r[j], x = x + alpha*u[0]
        matSloppy(*r[j+1], *r[j], temp);
        blas::caxpy(alpha, *u[0], x_sloppy);
        
      } // End BiCG part.      
      
      // MR part. Really just modified Gram-Schmidt.
      // The algorithm uses the byproducts of the Gram-Schmidt to update x
      //   and other such niceties. One day I'll read the paper more closely.
      // Can take this from 'orthoDir' in inv_gcr_quda.cpp, hard code pipelining up to l = 8.
      for (int j = 1; j <= nKrylov; j++)
      {
        for (int i = 1; i < j; i++)
        {
          // tau_ij = <r_i,r_j>/sigma_i.
          // This doesn't break on the first iteration because i < j is true.
          // (I was confused about this.)
          tau[i][j] = blas::cDotProduct(*r[i], *r[j])/sigma[i];
          
          // r_j = r_j - tau_ij r_i;
          blas::caxpy(-tau[i][j], *r[i], *r[j]);
        }
        
        // sigma_j = r_j^2, gamma'_j = <r_0, r_j>/sigma_j
        sigma[j] = blas::norm2(*r[j]);
        gamma_prime[j] = blas::cDotProduct(*r[j], *r[0])/sigma[j];
      }
      
      // gamma[nKrylov] = gamma'[nKrylov], omega = gamma[nKrylov]
      gamma[nKrylov] = gamma_prime[nKrylov];
      omega = gamma[nKrylov];
      
      // gamma = T^(-1) gamma_prime. It's in the paper, I promise.
      for (int j = nKrylov-1; j > 0; j--)
      {
        // Internal def: gamma[j] = gamma'_j - \sum_{i = j+1 to nKrylov} tau_ji gamma_i
        gamma[j] = gamma_prime[j];
        for (int i = j+1; i <= nKrylov; i++)
        {
          gamma[j] = gamma[j] - tau[j][i]*gamma[i];
        }
      }
      
      // gamma'' = T S gamma. Check paper for defn of S.
      for (int j = 1; j < nKrylov; j++)
      {
        gamma_prime_prime[j] = gamma[j+1];
        for (int i = j+1; i < nKrylov; i++)
        {
          gamma_prime_prime[j] = gamma_prime_prime[j] + tau[j][i]*gamma[i+1];
        }
      }
      
      // Update x, r, u.
      // x = x+ gamma_1 r_0, r_0 = r_0 - gamma'_l r_l, u_0 = u_0 - gamma_l u_l, where l = nKrylov.
      blas::caxpy(-gamma[nKrylov], *u[nKrylov], *u[0]);
      
      // This became a fused operator.
      //blas::caxpy(gamma[1], *r[0], x_sloppy);
      //blas::caxpy(-gamma_prime[nKrylov], *r[nKrylov], *r[0]);
      blas::caxpyBzpx(gamma[1], *r[0], x_sloppy, -gamma_prime[nKrylov], *r[nKrylov]);
      
      // for j = 1 .. nKrylov-1: u[0] -= gamma_j u[j], x += gamma''_j r[j], r[0] -= gamma'_j r[j]
      for (int j = 1; j < nKrylov; j++)
      {
        blas::caxpy(-gamma[j], *u[j], *u[0]);
        
        // This became a fused operator
        //blas::caxpy(gamma_prime_prime[j], *r[j], x_sloppy);
        //blas::caxpy(-gamma_prime[j], *r[j], *r[0]);
        blas::caxpyBxpz(gamma_prime_prime[j], *r[j], x_sloppy, -gamma_prime[j], *r[0]);
      }
      
      // sigma[0] = r_0^2
      sigma[0] = blas::norm2(*r[0]);
      r2 = sigma[0];
      
      // Check if we need to do a reliable update.
      // In inv_bicgstab_quda.cpp, there's a variable 'updateR' that holds the check.
      // That variable gets carried about because there are a few different places 'r' can get
      // updated (depending on if you're using pipelining or not). In BiCGstab-L, there's only
      // one place (for now) to get the updated residual, so we just do away with 'updateR'.
      // Further remark: "reliable" updates rNorm, maxrr, maxrx!! 
      if (reliable(rNorm, maxrx, maxrr, r2, delta))
      {
        if (x.Precision() != x_sloppy.Precision())
        {
          blas::copy(x, x_sloppy);
        }
        
        blas::xpy(x, y); // swap these around? (copied from bicgstab)
        
        // Explicitly recompute the residual.
        mat(r_full, y, x); // r[0] = Ax
        
        r2 = blas::xmyNorm(b, r_full); // r = b - Ax, return norm.
        
        sigma[0] = r2;
        
        if (x.Precision() != r[0]->Precision())
        {
          blas::copy(*r[0], r_full);
        }
        blas::zero(x_sloppy);
        
        // Update rNorm, maxrr, maxrx.
        rNorm = sqrt(r2);
        maxrr = rNorm;
        maxrx = rNorm;
        
        // Increment the reliable update count.
        rUpdate++; 
      }
      
      // Check convergence.
      k += nKrylov;
      PrintStats(solver_name.c_str(), k, r2, b2, 0.0); // last thing should be heavy quark res...
    } // Done iterating.
    
    if (x.Precision() != x_sloppy.Precision())
    {
      blas::copy(x, x_sloppy);
    }
    
    blas::xpy(y, x);
    
    // Done with compute, begin the epilogue.
    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTART(QUDA_PROFILE_EPILOGUE);
    
    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (blas::flops + mat.flops()/* + matSloppy.flops()*/)*1e-9;
    param.gflops = gflops;
    param.iter += k;
    
    if (k >= param.maxiter) // >= if nKrylov doesn't divide max iter.
      warningQuda("Exceeded maximum iterations %d", param.maxiter);

    // Print number of reliable updates.
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("%s: Reliable updates = %d\n", solver_name.c_str(), rUpdate);

    // compute the true residual
    // !param.is_preconditioner comes from bicgstab, param.compute_true_res came from gcr.
    if (!param.is_preconditioner && param.compute_true_res) { // do not do the below if this is an inner solver.
      mat(r_full, x, y);
      double true_res = blas::xmyNorm(b, r_full);
      param.true_res = sqrt(true_res / b2);
      // Probably some heavy quark stuff...
      param.true_res_hq = 0.0; //use_heavy_quark_res ? sqrt(blas::HeavyQuarkResidualNorm(x,*r[0]).z) : 0.0;
    }
    
    // Reset flops counters.
    blas::flops = 0;
    mat.flops();
    
    // copy the residual to b so we can use it outside of the solver.
    if (param.preserve_source == QUDA_PRESERVE_SOURCE_NO)
    {
      blas::copy(b, r_full);
    }
    
    // Done with epilogue, begin free.
    
    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    profile.TPSTART(QUDA_PROFILE_FREE);
    
    // ...yup...
    PrintSummary(solver_name.c_str(), k, r2, b2);
    
    // Done!
    profile.TPSTOP(QUDA_PROFILE_FREE);
    return;
  }

} // namespace quda
