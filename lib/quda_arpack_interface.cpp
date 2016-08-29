#include <quda_arpack_interface.h>
#include <color_spinor_field_order.h>

using namespace quda ;

  struct SortEvals{
    double _val;
    int    _idx;

    SortEvals(double val, int idx) : _val(val), _idx(idx) {};

    static bool CmpEigenNrms (SortEvals v1, SortEvals v2) { return (v1._val < v2._val);}
  };

  template<typename Float> void arpack_naupd(int &ido, char &bmat, int &n, char *which, int &nev, Float &tol,  void *resid, int &ncv, void *v, int &ldv,
                    int *iparam, int *ipntr, void *workd, void *workl, int &lworkl, void *rwork, int &info)
  {
    if(sizeof(Float) == sizeof(float))
    {
       float _tol  = static_cast<float>(tol);
       ARPACK(cnaupd)(&ido, &bmat, &n, which, &nev, &_tol, static_cast<std::complex<float> *>(resid), &ncv, static_cast<std::complex<float> *>(v),
                       &ldv, iparam, ipntr, static_cast<std::complex<float> *>(workd), static_cast<std::complex<float> *>(workl), &lworkl, static_cast<float*>(rwork), &info);
    }
    else
    {
       double _tol = static_cast<double>(tol);
       ARPACK(znaupd)(&ido, &bmat, &n, which, &nev, &_tol, static_cast<std::complex<double> *>(resid), &ncv, static_cast<std::complex<double> *>(v),
                       &ldv, iparam, ipntr, static_cast<std::complex<double> *>(workd), static_cast<std::complex<double> *>(workl), &lworkl, static_cast<double*>(rwork), &info);
    }

    return;
  }

  template<typename Float> void arpack_neupd (int &comp_evecs, char howmny, int *select, void* evals, void* v, int &ldv, void* sigma, void* workev, 
		       char bmat, int &n, char *which, int &nev, Float tol,  void* resid, int &ncv, void* v1, int &ldv1, int *iparam, int *ipntr, 
                       void* workd, void* workl, int &lworkl, void* rwork, int &info)
  {
    if(sizeof(Float) == sizeof(float))
    {   
       float _tol = static_cast<float>(tol);
       ARPACK(cneupd)(&comp_evecs, &howmny, select, static_cast<std::complex<float> *>(evals),
                     static_cast<std::complex<float> *>(v), &ldv, static_cast<std::complex<float> *>(sigma), static_cast<std::complex<float> *>(workev), &bmat, &n, which,
                     &nev, &_tol, static_cast<std::complex<float> *>(resid), &ncv, static_cast<std::complex<float> *>(v1),
                     &ldv1, iparam, ipntr, static_cast<std::complex<float> *>(workd), static_cast<std::complex<float> *>(workl),
                     &lworkl, static_cast<float *>(rwork), &info); 
    }
    else
    {
       double _tol = static_cast<double>(tol);
       ARPACK(zneupd)(&comp_evecs, &howmny, select, static_cast<std::complex<double> *>(evals),
                     static_cast<std::complex<double> *>(v), &ldv, static_cast<std::complex<double> *>(sigma), static_cast<std::complex<double> *>(workev), &bmat, &n, which,
                     &nev, &_tol, static_cast<std::complex<double> *>(resid), &ncv, static_cast<std::complex<double> *>(v1),
                     &ldv1, iparam, ipntr, static_cast<std::complex<double> *>(workd), static_cast<std::complex<double> *>(workl),
                     &lworkl, static_cast<double *>(rwork), &info);
    }

    return;
  }


  template<typename Float, int fineSpin, int fineColor, int reducedColor>
  void convertFrom2DVector(cpuColorSpinorField &out, std::complex<Float> *in) {
     if(out.FieldOrder() != QUDA_SPACE_SPIN_COLOR_FIELD_ORDER ) errorQuda("\nIncorrect feild order (%d).\n", out.FieldOrder() );
     quda::colorspinor::FieldOrderCB<Float,fineSpin,fineColor,1,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER> outOrder(static_cast<ColorSpinorField&>(out));//fineColor =3 here!

     blas::zero(out);

     for (int parity = 0; parity < 2; parity++) {
       for(int x_cb = 0; x_cb < out.VolumeCB(); x_cb++) {

         int i = parity*out.VolumeCB() + x_cb;
         int xx[4] = {0};
         out.LatticeIndex(xx, i);

         int _2d_idx = (xx[0] + xx[1]*out.X(0))*fineSpin*reducedColor;

         if( xx[2] == 0 && xx[3] == 0 ) for(int s = 0; s < fineSpin; s++) for(int c = 0; c < reducedColor; c++) outOrder(parity, x_cb, s, c) = in[_2d_idx+s*reducedColor+c];
       }
     }

     return;
  }

  template<typename Float, int fineSpin, int fineColor, int reducedColor>
  void convertTo2DVector(std::complex<Float> *out, cpuColorSpinorField &in) {
     if(in.FieldOrder() != QUDA_SPACE_SPIN_COLOR_FIELD_ORDER ) errorQuda("\nIncorrect feild order (%d).\n", in.FieldOrder() );
     quda::colorspinor::FieldOrderCB<Float,fineSpin,fineColor,1,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER> inOrder(static_cast<ColorSpinorField&>(in));

     for (int parity = 0; parity < 2; parity++) {
       for(int x_cb = 0; x_cb < in.VolumeCB(); x_cb++) {

         int i = parity*in.VolumeCB() + x_cb;
         int xx[4] = {0};
         in.LatticeIndex(xx, i);

         int _2d_idx = (xx[0] + xx[1]*in.X(0))*fineSpin*reducedColor;

         if( xx[2] == 0 && xx[3] == 0 ) for(int s = 0; s < fineSpin; s++) for(int c = 0; c < reducedColor; c++) out[_2d_idx+s*reducedColor+c] = inOrder(parity, x_cb, s, c);
       }
     }

     return;
  }

  template<typename Float, int fineSpin, int fineColor, int reducedColor, bool do_2d_emulation>
  void arpack_matvec(std::complex<Float> *out, std::complex<Float> *in,  DiracMatrix &matEigen, QudaPrecision matPrecision, ColorSpinorField &meta)
  {
    ColorSpinorParam csParam(meta);

    csParam.create = QUDA_ZERO_FIELD_CREATE;  
    //cpuParam.extendDimensionality();5-dim field
    csParam.location = QUDA_CPU_FIELD_LOCATION;
    csParam.setPrecision(sizeof(Float) == sizeof(float) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION);
    csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;

    if(!do_2d_emulation) 
    {
      csParam.create = QUDA_REFERENCE_FIELD_CREATE;
      csParam.v      = static_cast<void*>(in);
    }

    cpuColorSpinorField *cpu_tmp1 = static_cast<cpuColorSpinorField*>(ColorSpinorField::Create(csParam));
    
    if(!do_2d_emulation)
    {
      csParam.v      = static_cast<void*>(out);
    }
    else
    {
      convertFrom2DVector<Float, fineSpin, fineColor, reducedColor>(*cpu_tmp1, in);
    }
    
    cpuColorSpinorField *cpu_tmp2 = static_cast<cpuColorSpinorField*>(ColorSpinorField::Create(csParam));

    csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    csParam.location   = QUDA_CUDA_FIELD_LOCATION; // hard code to GPU location for null-space generation for now
    csParam.create     = QUDA_COPY_FIELD_CREATE;
    csParam.setPrecision(matPrecision);

    ColorSpinorField *cuda_tmp1 = static_cast<ColorSpinorField*>(new cudaColorSpinorField(*cpu_tmp1, csParam));
    //
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    ColorSpinorField *cuda_tmp2 = static_cast<ColorSpinorField*>(new cudaColorSpinorField(csParam));

    matEigen(*cuda_tmp2, *cuda_tmp1);

    *cpu_tmp2 = *cuda_tmp2;
    if(do_2d_emulation) convertTo2DVector<Float, fineSpin, fineColor, reducedColor>(out, *cpu_tmp2);

    delete cpu_tmp1;
    delete cpu_tmp2;

    delete cuda_tmp1;
    delete cuda_tmp2;

    return;
  }

//copy fields:
  template<typename Float, int fineSpin, int fineColor, int reducedColor, bool do_2d_emulation> 
  void copy_eigenvectors(std::vector<ColorSpinorField*> &B, std::complex<Float> *arpack_evecs, std::complex<Float> *arpack_evals, const int cldn, const int nev)
  {
    std::vector<SortEvals> sorted_evals_cntr;
    sorted_evals_cntr.reserve(nev);

    ColorSpinorParam csParam(*B[0]);

    csParam.create = do_2d_emulation ? QUDA_ZERO_FIELD_CREATE : QUDA_REFERENCE_FIELD_CREATE;  
    //cpuParam.extendDimensionality();5-dim field
    csParam.location = QUDA_CPU_FIELD_LOCATION;
    csParam.setPrecision(sizeof(Float) == sizeof(float) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION);
    csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;

    for(int e = 0; e < nev; e++) sorted_evals_cntr.push_back( SortEvals(arpack_evals[e].imag(), e ));
    std::stable_sort(sorted_evals_cntr.begin(), sorted_evals_cntr.end(), SortEvals::CmpEigenNrms);

    cpuColorSpinorField *cpu_tmp = nullptr;
    int ev_id = 0;

    for(std::vector<ColorSpinorField*>::iterator vec = B.begin() ; vec != B.end(); ++vec) {
      int sorted_id =  sorted_evals_cntr[ev_id++]._idx;

      std::complex<Float>* tmp_buffer =  &arpack_evecs[sorted_id*cldn];
      cpuColorSpinorField *curr_nullvec = static_cast<cpuColorSpinorField*> (*vec);
      if(do_2d_emulation)
      {
        static_cast<void*>(tmp_buffer);
        cpu_tmp = static_cast<cpuColorSpinorField*>(ColorSpinorField::Create(csParam));
        convertFrom2DVector<Float, fineSpin, fineColor, reducedColor>(*cpu_tmp, tmp_buffer);
      }
      else
      {
        csParam.v = static_cast<void*>(tmp_buffer);
        cpu_tmp = static_cast<cpuColorSpinorField*>(ColorSpinorField::Create(csParam));
      }

      *curr_nullvec = *cpu_tmp;

      delete cpu_tmp;
    }
  }

  template<typename Float, int fineSpin, int fineColor, int reducedColor, bool do_2d_emulation>
  int arpack_solve( char *lanczos_which, std::vector<ColorSpinorField*> &B, void *evals, DiracMatrix &mat,  QudaPrecision matPrecision, Float tol, int nev, int ncv)
  {
    int   max_iter = 240000;

    size_t clen = B[0]->X(0)*B[0]->X(1)*( do_2d_emulation ? (B[0]->X(2)*B[0]->X(3)) : 1 )*B[0]->Nspin()*B[0]->Ncolor();
    size_t cldn = clen;
    const size_t wbytes = cldn*sizeof(std::complex<Float>);
    void *arpack_evecs = malloc(wbytes*ncv);     /* workspace for evectors (BLAS-matrix), [ld_evec, >=ncv] */

    /* all FORTRAN communication uses underscored variables */
    int ido_; 
    int info_;
    int iparam_[11];
    int ipntr_[14];
    int n_      = clen,
        nev_    = nev,
        ncv_    = ncv,
        ldv_    = cldn,
        lworkl_ = (3 * ncv_ * ncv_ + 5 * ncv_) * 2,
        rvec_   = 1;
    void *sigma_ = malloc(sizeof(std::complex<Float>));
    Float tol_ = tol;

    void *w_d_         = evals;
    void *w_v_         = arpack_evecs;

    void *resid_      = malloc(wbytes);
    void *w_workd_    = malloc(wbytes * 3);
    void *w_workl_    = malloc(sizeof(std::complex<Float>) * lworkl_);
    void *w_rwork_    = malloc(sizeof(Float) *ncv_);
    
    /* __neupd-only workspace */
    void *w_workev_   = malloc(sizeof(std::complex<Float>) * 2 * ncv_);
    int *select_      = (int*)malloc(sizeof(int) * ncv_);

    if(resid_ == nullptr||
           w_workd_ == nullptr||
           w_workl_ == nullptr||
           w_rwork_ == nullptr||
           w_workev_ == nullptr||
           select_ == nullptr)    errorQuda("Could not allocate memory..");

    memset(sigma_, 0, sizeof(std::complex<Float>));
    memset(resid_, 0, wbytes);
    memset(w_workd_, 0, wbytes * 3);

    /* cnaupd cycle */
    ido_        = 0;
    info_       = 0;
    iparam_[0]  = 1;
    iparam_[2]  = max_iter;
    iparam_[3]  = 1;
    iparam_[6]  = 1;

    char howmny='P';
    char bmat = 'I';

    int iter_cnt= 0;

    do {
      //interface to arpack routines
      arpack_naupd<Float>(ido_, bmat, n_, lanczos_which, nev_, tol, resid_, ncv_, w_v_, ldv_, iparam_, ipntr_, w_workd_, w_workl_, lworkl_, w_rwork_, info_);
  
      if (info_ != 0) errorQuda("\nError in ARPACK CNAUPD (error code %d) , exit.\n", info_);

      iter_cnt++;
        
      if (ido_ == -1 || ido_ == 1) {
         //apply matrix vector here:
         std::complex<Float> *in   = &(static_cast<std::complex<Float>*> (w_workd_))[(ipntr_[0]-1)];
         std::complex<Float> *out  = &(static_cast<std::complex<Float>*> (w_workd_))[(ipntr_[1]-1)];
         //
         arpack_matvec<Float, fineSpin, fineColor, reducedColor, do_2d_emulation> (out, in,  mat, matPrecision, *B[0]) ;
         if(iter_cnt % 50 == 0) printfQuda("\nIteration : %d\n", iter_cnt);
      } 

    } while (99 != ido_ && iter_cnt < max_iter);

    printfQuda("Finish: iter=%04d  info=%d  ido=%d\n", iter_cnt, info_, ido_);

    //int conv_cnt = iparam_[4];

    /* for howmny="P", no additional space is required */
    arpack_neupd<Float>(rvec_, howmny, select_, w_d_, w_v_, ldv_, sigma_, w_workev_, bmat, n_, lanczos_which,
                        nev_, tol_, resid_, ncv_, w_v_, ldv_, iparam_, ipntr_, w_workd_, w_workl_, lworkl_, w_rwork_, info_);

    if (info_ != 0) errorQuda("\nError in ARPACK CNEUPD (error code %d) , exit.\n", info_);    
//copy fields:
    copy_eigenvectors<Float, fineSpin, fineColor, reducedColor, do_2d_emulation>(B, static_cast<std::complex<Float>* > (arpack_evecs), static_cast<std::complex<Float>* > (w_d_), cldn, nev);

    printfQuda("\ndone..\n");

    /* cleanup */
    if (w_workl_ != nullptr)   free(w_workl_);
    if (w_rwork_ != nullptr)   free(w_rwork_);
    if (w_workev_ != nullptr)  free(w_workev_);
    if (select_ != nullptr)    free(select_);

    //n_iters    = iter_cnt;
    //nconv      = conv_cnt;
    free(arpack_evecs);

    if (w_workd_ != nullptr)   free(w_workd_);
    if (resid_   != nullptr)   free(resid_);

    return 0;
  }



///////////////////////////////////////////////////ARPACK SOLVER////////////////////////////////////////////////////////


 void ArpackArgs::operator()( std::vector<ColorSpinorField*> &B, std::complex<double> *evals )
 {
   const int fineSpin  = 1;
   const int fineColor = 3;

   if(_2d_field)
   {
     warningQuda("\nSolving 2d eigen-problem\n");
     if(reducedColors == 1)
     {
        if(use_full_prec_arpack)   arpack_solve<double, fineSpin, fineColor, 1, true>( lanczos_which, B, (void*)evals, matEigen, mat_precision, tol, nev , ncv );
        else                       arpack_solve<float, fineSpin, fineColor, 1, true>( lanczos_which, B, (void*)evals, matEigen, mat_precision, (float)tol, nev , ncv  );
     }
     else errorQuda("\nUnsupported colors.\n");
   }
   else
   {
     if(use_full_prec_arpack)   arpack_solve<double, fineSpin, fineColor, fineColor, false>( lanczos_which, B, (void*)evals, matEigen, mat_precision, tol, nev , ncv );
     else                       arpack_solve<float, fineSpin, fineColor, fineColor, false>( lanczos_which, B, (void*)evals, matEigen, mat_precision, (float)tol, nev , ncv  );
   }
 
   return;
 }
