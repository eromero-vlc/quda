#pragma once

#ifdef PRIMME_LIB

#include <eigensolve_quda.h>

namespace quda
{
  /**
     @brief PRIMME eigensolver.
  */
  class PRIMME : public EigenSolver
  {

public:
    const DiracMatrix &mat;
    /**
       @brief Constructor for Thick Restarted Eigensolver class
       @param eig_param The eigensolver parameters
       @param mat The operator to solve
       @param profile Time Profile
    */
    PRIMME(QudaEigParam *eig_param, const DiracMatrix &mat, TimeProfile &profile);

    /**
       @brief Compute eigenpairs
       @param[in] kSpace Krylov vector space
       @param[in] evals Computed eigenvalues
    */
    void operator()(std::vector<ColorSpinorField *> &kSpace, std::vector<Complex> &evals);

    /**
       @brief return eig_param
    */
    const QudaEigParam* get_eig_param() const { return eig_param; }
  };

} // namespace quda

#endif // PRIMME_LIB
