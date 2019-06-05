#ifdef PRIMME_LIB

#include <vector>
#include <type_traits>
#include <algorithm>
#include <complex>

#include <quda_internal.h>
#include <eigensolve_quda.h>
#include <quda_primme_interface.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

// PRIMME INTERFACE ROUTINES
//--------------------------------------------------------------------------


#include <primme.h>
#include "magma_v2.h"

static size_t lcm(size_t x, size_t y)
{
  size_t greater = std::max(x, y);
  while (greater % x != 0 || greater % y != 0) greater++;
  return greater;
}

/** @fn  int tprimme(evals_type *evals, evecs_type *evecs, resNorms_type *resNorms, primme_params *primme, QudaFieldLocation location)
   @brief Main call to PRIMME
   @param evals The returned eigenvalues; if it is not complex, the operator is assumed to be Hermitian; otherwise it is normal operator
   @param evecs The input initial guesses, and the output eigenvectors
   @param resNorms The residual vector norms of the returned eigenpairs
   @param primme The eigensolver parameters
   @param location The location (CPU/GPU) of the eigenvectors
*/
static int tprimme(float *evals, PRIMME_COMPLEX_HALF *evecs, float *resNorms, primme_params *primme, QudaFieldLocation location)
{
  if (location == QUDA_CPU_FIELD_LOCATION ) return       ksprimme       (evals, evecs, resNorms, primme);
  if (location == QUDA_CUDA_FIELD_LOCATION) return magma_ksprimme       (evals, evecs, resNorms, primme);
  errorQuda("Invalid location %d", location); return -1;
}
static int tprimme(float *evals, std::complex<float> *evecs, float *resNorms, primme_params *primme, QudaFieldLocation location)
{
  if (location == QUDA_CPU_FIELD_LOCATION ) return       cprimme        (evals, evecs, resNorms, primme);
  if (location == QUDA_CUDA_FIELD_LOCATION) return magma_cprimme        (evals, evecs, resNorms, primme);
  errorQuda("Invalid location %d", location); return -1;
}
static int tprimme(double *evals, std::complex<double> *evecs, double *resNorms, primme_params *primme, QudaFieldLocation location)
{
  if (location == QUDA_CPU_FIELD_LOCATION ) return       zprimme        (evals, evecs, resNorms, primme);
  if (location == QUDA_CUDA_FIELD_LOCATION) return magma_zprimme        (evals, evecs, resNorms, primme);
  errorQuda("Invalid location %d", location); return -1;
}
static int tprimme(std::complex<float> *evals, PRIMME_COMPLEX_HALF *evecs, float *resNorms, primme_params *primme, QudaFieldLocation location)
{
  if (location == QUDA_CPU_FIELD_LOCATION ) return       kcprimme_normal(evals, evecs, resNorms, primme);
  if (location == QUDA_CUDA_FIELD_LOCATION) return magma_kcprimme_normal(evals, evecs, resNorms, primme);
  errorQuda("Invalid location %d", location); return -1;
}
static int tprimme(std::complex<float> *evals, std::complex<float> *evecs, float *resNorms, primme_params *primme, QudaFieldLocation location)
{
  if (location == QUDA_CPU_FIELD_LOCATION ) return       cprimme_normal (evals, evecs, resNorms, primme);
  if (location == QUDA_CUDA_FIELD_LOCATION) return magma_cprimme_normal (evals, evecs, resNorms, primme);
  errorQuda("Invalid location %d", location); return -1;
}
static int tprimme(std::complex<double> *evals, std::complex<double> *evecs, double *resNorms, primme_params *primme, QudaFieldLocation location)
{
  if (location == QUDA_CPU_FIELD_LOCATION ) return       zprimme_normal (evals, evecs, resNorms, primme);
  if (location == QUDA_CUDA_FIELD_LOCATION) return magma_zprimme_normal (evals, evecs, resNorms, primme);
  errorQuda("Invalid location %d", location); return -1;
}

/** @struct QudaPrimmeType
   @brief Auxiliary class that defines the type of evals and resNorms from the tprimme
   @tparam prec Eigensolver's precision
   @typedef real_type type of Hermitian problems' eigenvalues, and residual norms.
   @typedef complex_type type of non-Hermitian problems' eigenvalues.
   @typedef evecs_type type of the eigenvectors.
*/
template<QudaPrecision prec> struct QudaPrimmeType;
template<> struct QudaPrimmeType<QUDA_HALF_PRECISION>   { typedef float  real_type; typedef std::complex<float>  complex_type; typedef PRIMME_COMPLEX_HALF  evecs_type; static constexpr const char *name = "half"  ;};
template<> struct QudaPrimmeType<QUDA_SINGLE_PRECISION> { typedef float  real_type; typedef std::complex<float>  complex_type; typedef std::complex<float>  evecs_type; static constexpr const char *name = "single";};
template<> struct QudaPrimmeType<QUDA_DOUBLE_PRECISION> { typedef double real_type; typedef std::complex<double> complex_type; typedef std::complex<double> evecs_type; static constexpr const char *name = "double";};


namespace quda
{
  //-----------------------------------------------------------------------------
  //-----------------------------------------------------------------------------

  // PRIMME constructor
  PRIMME::PRIMME(QudaEigParam *eig_param, const DiracMatrix &mat, TimeProfile &profile) :
    EigenSolver(eig_param, profile),
    mat(mat)
  {
    // PRIMME checks
    if (!(eig_param->spectrum == QUDA_SPECTRUM_SR_EIG || eig_param->spectrum == QUDA_SPECTRUM_SM_EIG)) {
      errorQuda("QUDA's PRIMME interface only supports SR and SM. Let us know if you need something else!");
    }
  }

  namespace { // Auxiliary functions

    // Auxiliary function to copy the xi-th column of x into the yi-th column of y
    // NOTE: column indices start from zero
    void copy_column(ColorSpinorField &y, int yi, ColorSpinorField &x, int xi) {

      ColorSpinorParam xparam(x), yparam(y);
      if (xi > xparam.nVec)
        errorQuda("Wrong column number, asking for column %d of x, but only having %d", xi, xparam.nVec);
      if (yi >= yparam.nVec)
        errorQuda("Wrong column number, asking for column %d of y, but only having %d", yi, yparam.nVec);
      xparam.nVec = yparam.nVec = 1;
      xparam.create = yparam.create = QUDA_REFERENCE_FIELD_CREATE;
      *(char**)&xparam.v += x.Bytes() * xi;
      *(char**)&yparam.v += y.Bytes() * yi;
      ColorSpinorField *x1 = ColorSpinorField::Create(xparam); 
      ColorSpinorField *y1 = ColorSpinorField::Create(yparam); 
      blas::copy(*y1, *x1);
      delete x1;
      delete y1;
    }

    // Auxiliary function for global sum reduction
    void globalSumDouble(void *sendBuf, void *recvBuf, int *count, 
        primme_params *primme, int *ierr)
    {
      double *x = (double*)sendBuf, *y = (double*)recvBuf;
      if (x != y) {
        for (int i=0, count_=*count; i<count_; i++) y[i] = x[i];
      }
      reduceDoubleArray(y, *count);
      *ierr = 0;
    }

    // Auxiliary function for the matvec
    template<bool use_inv> struct primmeMatvec {
      static void fun(void *x0, PRIMME_INT *ldx, void *y0, PRIMME_INT *ldy,
          int *blockSize, primme_params *primme, int *ierr)
      {
        // If this routine exits before reaching the end, notify it as an error by default
        *ierr = -1;

        // Quick return
        if (*blockSize <= 0) {*ierr = 0; return; }

        // Wrap the raw pointers into ColorSpinorField
        ColorSpinorParam *invParam = (ColorSpinorParam *)primme->commInfo;
        invParam->nVec = *blockSize;
        invParam->create = QUDA_REFERENCE_FIELD_CREATE;
        invParam->v = x0;
        ColorSpinorField *x = ColorSpinorField::Create(*invParam);
        invParam->v = y0;
        ColorSpinorField *y = ColorSpinorField::Create(*invParam);

        PRIMME *eigensolver = (PRIMME*)primme->matrix; 
        if (!use_inv) {
          // Do y = Dirac * x
          eigensolver->matVec(eigensolver->mat, *y, *x);
        } else {
          // Do y = Dirac^{-1} * x
          invertQuda(y, x, eigensolver->get_eig_param()->invert_param);
        }

        // Do y = gamma * y
        gamma5(*y, *y);

        // Clean up
        delete x;
        delete y;

        // We're good!
        *ierr = 0;
      }
    };

    template <QudaPrecision prec, bool twisted>
    void call(std::vector<ColorSpinorField *> &kSpace, std::vector<Complex> &evals_out, PRIMME *eigensolver, bool use_inv = true)
    {
      const QudaEigParam *eig_param = eigensolver->get_eig_param();

      // Place eigenvectors where the inverter is located regardless of the location of kSpace
      QudaFieldLocation location = eig_param->invert_param->input_location;

      // BLAS/LAPACK obsession with the Fortran's way of passing arrays is contagious, and eigenvectors passed to PRIMME,
      // library which relays on BLAS/LAPACK, follow the same convention. So create a multivector for evecs
      ColorSpinorParam invParam(*kSpace[0]);
      invParam.nVec = eig_param->nEv;
      invParam.create = QUDA_NULL_FIELD_CREATE;
      ColorSpinorField *evecs = ColorSpinorField::Create(invParam);

      // Copy initial guess to evecs
      int initSize = 0;
      for (size_t i=0; i<kSpace.size() && i<(size_t)std::min(eig_param->nEv, 0); i++) {
        if (kSpace[i] == nullptr) break;
        if (sqrt(blas::norm2(*kSpace[i])) <= 0) break;
        copy_column(*evecs, i, *kSpace[i], 0);
        initSize++;
      }
      if (getVerbosity() >= QUDA_SUMMARIZE && initSize > 0)
        printfQuda("Using %d initial guesses\n", initSize);

      // Initialize PRIMME configuration
      primme_params primme;
      primme_initialize(&primme);

      // Set GPU device
      magma_queue_t queue = 0;
      if (location == QUDA_CUDA_FIELD_LOCATION) {
        magma_queue_create(comm_gpuid(), &queue);
        primme.queue = &queue;
      }

      // Set global sum reduction
      primme.globalSumReal = globalSumDouble;
      primme.globalSumReal_type = primme_op_double;

      // Determine local and global matrix dimension
      size_t local_vol = 1;
      for (int i = 0; i < 4; i++) {
        local_vol *= invParam.x[i];
      }
      local_vol *= eig_param->invert_param->Ls;
      int nSpin = (eig_param->invert_param->dslash_type == QUDA_LAPLACE_DSLASH) ? 1 : 4;
      size_t nLocal = local_vol * nSpin * 3;
      double n = nLocal; reduceDouble(n);
      primme.n = n;             /* set global problem dimension */
      primme.nLocal = nLocal;   /* set local problem dimension */

      primme.numEvals = eig_param->nEv;   /* Number of wanted eigenpairs */
      primme.eps = eig_param->tol;      /* ||r|| <= eps * ||matrix|| */

      // Create evals. For twisted operators, gamma * operator is not Hermitian, but normal.
      typedef typename QudaPrimmeType<prec>::real_type real_type;
      typedef typename QudaPrimmeType<prec>::complex_type complex_type;
      typedef typename std::conditional<twisted, complex_type, real_type>::type evals_type;
      typedef typename QudaPrimmeType<prec>::evecs_type evecs_type;
      evals_type *evals = new evals_type[eig_param->nEv];

      // Create residual norms
      real_type *rnorms = new real_type[eig_param->nEv];

      // Seek for the largest eigenvalue in magnitude for using the inverse operator; and the smallest otherwise
      primme.target = (use_inv ? primme_largest_abs : primme_closest_abs);
      double zero = 0;
      primme.targetShifts = &zero;
      primme.numTargetShifts = 1;

      // Set operator
      primme.matrixMatvec = use_inv ? primmeMatvec<true>::fun : primmeMatvec<false>::fun;
      ColorSpinorParam invParam0(invParam);
      primme.commInfo = &invParam0;
      primme.matrix = (void*)eig_param;

      // Enforce leading dimension of matvec's input/output vectors to be aligned for textures
      if (location == QUDA_CUDA_FIELD_LOCATION && deviceProp.textureAlignment > 0) {
        size_t alignment = lcm(sizeof(evecs_type), deviceProp.textureAlignment);
        size_t nLocal_aligned = (nLocal * sizeof(evecs_type) + alignment - 1) / sizeof(evecs_type);
        primme.ldOPs = nLocal_aligned; 
      }

      // Set advanced options. If the operator is an inverter, configure PRIMME to minimize the
      // inverter applications. Otherwise use an strategy that minimizes orthogonalization time.
      primme_set_method(use_inv ? PRIMME_DEFAULT_MIN_MATVECS : PRIMME_DEFAULT_MIN_TIME, &primme);
      if (!use_inv) {
        primme.projectionParams.projection = primme_proj_refined;
        primme.correctionParams.maxInnerIterations = 160;
        primme.correctionParams.convTest = primme_full_LTolerance;
      }

      // Use a fast, but inaccurate orthogonalization. It seems enough for these problems
      primme.orth = primme_orth_implicit_I;

      // Display PRIMME configuration struct (optional)
      if (getVerbosity() >= QUDA_SUMMARIZE)
        primme_display_params(primme);

      if (getVerbosity() >= QUDA_SUMMARIZE) {
        printfQuda("*******************************\n");
        printfQuda("**** START PRIMME SOLUTION ****\n");
        printfQuda("*******************************\n");
      }

      if (getVerbosity() >= QUDA_SUMMARIZE)
        printfQuda("Running Eigensolver in %s precision\n", QudaPrimmeType<prec>::name);

      // Call primme
      int ret = tprimme(evals, (evecs_type*)evecs->V(), rnorms, &primme, location);

      if (ret != 0) {
        errorQuda("PRIMME returns the error code %d. Computed %d pairs.", ret, primme.initSize);
      }

      if (getVerbosity() >= QUDA_SUMMARIZE) {
        printfQuda("PRIMME computed the requested %d vectors in %g restart steps and %g OP*x operations.\n", primme.initSize,
            (double)primme.stats.numRestarts, (double)primme.stats.numMatvecs);

        // Dump all Ritz values and residua
        for (int i = 0; i < eig_param->nEv; i++) {
          printfQuda("RitzValue[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, (double)std::fabs(evals[i]), 0.0, (double)rnorms[i]);
        }
      }

      if (getVerbosity() >= QUDA_SUMMARIZE) {
        printfQuda("Time to solve problem using PRIMME  = %e\n", primme.stats.elapsedTime);
        printfQuda("Time spent in matVec                = %e  %.1f%%\n", primme.stats.timeMatvec,
            100 * primme.stats.timeMatvec / primme.stats.elapsedTime);
        printfQuda("Time spent in orthogonalization     = %e  %.1f%%\n", primme.stats.timeOrtho,
            100 * primme.stats.timeOrtho / primme.stats.elapsedTime);
        double timeComm = primme.stats.timeGlobalSum + primme.stats.timeBroadcast;
        printfQuda("Time spent in communications        = %e  %.1f%%\n", timeComm,
            100 * timeComm / primme.stats.elapsedTime);
      }

      // Copy evecs to kSpace
      kSpace.resize(primme.initSize);
      for (int i=0; i<primme.initSize; i++) {
        copy_column(*kSpace[i], 0, *evecs, i);
      }

      // Copy evals to evals_out
      evals_out.resize(primme.initSize);
      for (int i=0; i<primme.initSize; i++) {
        evals_out[i] = evals[i];
      }

      // Local clean-up
      delete [] rnorms;
      delete [] evals;
      delete evecs;
      if (location == QUDA_CUDA_FIELD_LOCATION) {
        magma_queue_destroy(queue);
      }
      primme_free(&primme);

      if (getVerbosity() >= QUDA_SUMMARIZE) {
        printfQuda("*******************************\n");
        printfQuda("***** END PRIMME SOLUTION *****\n");
        printfQuda("*******************************\n");
      }
    }

  } // namespace

  // Solver call
  void PRIMME::operator()(std::vector<ColorSpinorField *> &kSpace, std::vector<Complex> &evals)
  {
    // Quick exit
    if (eig_param->nEv <= 0) return;

    // Check to see if we are loading eigenvectors
    if (strcmp(eig_param->vec_infile, "") != 0) {
      printfQuda("Loading evecs from file name %s\n", eig_param->vec_infile);
      loadFromFile(mat, kSpace, evals);
      return;
    }

    // Set the precision of the eigensolver the same as the precision of the inverter
    QudaPrecision prec = kSpace[0]->Precision();

    // Pass precision and twisted as template arguments, because the types of vectors and values depend on them
    bool twisted = eig_param->invert_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH ? true : false;
    bool called = false;
    if (prec == QUDA_DOUBLE_PRECISION && !twisted) { call<QUDA_DOUBLE_PRECISION, false>(kSpace, evals, this); called = true; }
    if (prec == QUDA_SINGLE_PRECISION && !twisted) { call<QUDA_SINGLE_PRECISION, false>(kSpace, evals, this); called = true; }
    if (prec == QUDA_HALF_PRECISION   && !twisted) { call<QUDA_HALF_PRECISION,   false>(kSpace, evals, this); called = true; }
    if (prec == QUDA_DOUBLE_PRECISION &&  twisted) { call<QUDA_DOUBLE_PRECISION, true >(kSpace, evals, this); called = true; }
    if (prec == QUDA_SINGLE_PRECISION &&  twisted) { call<QUDA_SINGLE_PRECISION, true >(kSpace, evals, this); called = true; }
    if (prec == QUDA_HALF_PRECISION   &&  twisted) { call<QUDA_HALF_PRECISION,   true >(kSpace, evals, this); called = true; }
    if (!called) errorQuda("PRIMME: precision %d not supported", prec);

    // Only save if outfile is defined
    if (strcmp(eig_param->vec_outfile, "") != 0) {
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("saving eigenvectors\n");
      // Make an array of size nConv
      std::vector<ColorSpinorField *> vecs_ptr;
      for (size_t i = 0; i < kSpace.size(); i++) { vecs_ptr.push_back(kSpace[i]); }
      saveVectors(vecs_ptr, eig_param->vec_outfile);
    }
  }

} // namespace quda


#endif // PRIMME_LIB
