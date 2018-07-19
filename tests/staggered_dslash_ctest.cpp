#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <blas_quda.h>

#include <misc.h>
#include <test_util.h>
#include <dslash_util.h>
#include <staggered_dslash_reference.h>
#include "llfat_reference.h"
#include <gauge_field.h>
#include <unitarization_links.h>

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

#include <qio_field.h>

#include <assert.h>
#include <gtest.h>

using namespace quda;

#define MAX(a,b) ((a)>(b)?(a):(b))
#define staggeredSpinorSiteSize 6
// What test are we doing (0 = dslash, 1 = MatPC, 2 = Mat)

extern void usage(char** argv );

extern QudaDslashType dslash_type;

extern int test_type;

QudaGaugeParam gaugeParam;
QudaInvertParam inv_param;

cpuGaugeField *cpuFat = NULL;
cpuGaugeField *cpuLong = NULL;

cpuColorSpinorField *spinor, *spinorOut, *spinorRef, *tmpCpu;
cudaColorSpinorField *cudaSpinor, *cudaSpinorOut;

cudaColorSpinorField* tmp;

void *hostGauge[4];

// In the HISQ case, we include building fat/long links in this unit test
void *fatlink_gpu[4], *longlink_gpu[4];
void *fatlink_gpu_milc, *longlink_gpu_milc;

void *fatlink_cpu[4], *longlink_cpu[4];
void *fatlink_cpu_milc, *longlink_cpu_milc;
#ifdef MULTI_GPU
void **ghost_fatlink_cpu, **ghost_longlink_cpu;
#endif

QudaParity parity = QUDA_EVEN_PARITY;
extern QudaDagType dagger;
int transfer = 0; // include transfer time in the benchmark?
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern int gridsize_from_cmdline[];

extern int device;
extern bool verify_results;
extern int niter;

extern bool kernel_pack_t;

extern double mass; // the mass of the Dirac operator

// relativistic correction for naik term
extern double eps_naik;
// Number of naiks. If eps_naik is 0.0, we only need
// to construct one naik.
static int n_naiks = 1;

extern char latfile[];


int X[4];
extern int Nsrc; // number of spinors to apply to simultaneously

Dirac* dirac;

//const char *prec_str[] = {"half", "single", "double"};
const char *prec_str[] = {"double"};
const char *recon_str[] = {"r18", "r13", "r9"};

// Unitarization coefficients
static double unitarize_eps  = 1e-6;
static bool reunit_allow_svd = true;
static bool reunit_svd_only  = false;
static double svd_rel_error  = 1e-4;
static double svd_abs_error  = 1e-4;
static double max_allowed_error = 1e-11;

// For loading the gauge fields
int argc_copy;
char** argv_copy;

// matrix element debugging function
int getPrintVectorIndex(const int X[4], const int coord[4])
{
  //x[4] = cb_index/(X[3]*X[2]*X[1]*X[0]/2);
  //x[3] = (cb_index/(X[2]*X[1]*X[0]/2) % X[3];
  //x[2] = (cb_index/(X[1]*X[0]/2)) % X[2];
  //x[1] = (cb_index/(X[0]/2)) % X[1];
  //x[0] = 2*(cb_index%(X[0]/2)) + ((x[3]+x[2]+x[1]+parity)&1);
  int idx = ((((coord[3]*X[2]+coord[2])*X[1]+coord[1])*X[0])+coord[0]) >> 1;
  int phase = (coord[0]+coord[1]+coord[2]+coord[3])%2;
  return 2*idx+phase;
}


void init(int precision, QudaReconstructType link_recon) {

  //auto prec = precision == 2 ? QUDA_DOUBLE_PRECISION : precision  == 1 ? QUDA_SINGLE_PRECISION : QUDA_HALF_PRECISION;
  auto prec = QUDA_DOUBLE_PRECISION;

  setKernelPackT(kernel_pack_t);

  setVerbosity(QUDA_SUMMARIZE);

  gaugeParam = newQudaGaugeParam();
  inv_param = newQudaInvertParam();

  gaugeParam.X[0] = X[0] = xdim;
  gaugeParam.X[1] = X[1] = ydim;
  gaugeParam.X[2] = X[2] = zdim;
  gaugeParam.X[3] = X[3] = tdim;

  setDims(gaugeParam.X);
  dw_setDims(gaugeParam.X,Nsrc); // so we can use 5-d indexing from dwf
  setSpinorSiteSize(6);

  gaugeParam.cpu_prec = QUDA_DOUBLE_PRECISION;
  gaugeParam.cuda_prec = prec;
  gaugeParam.reconstruct = link_recon;
  gaugeParam.reconstruct_sloppy = gaugeParam.reconstruct;
  gaugeParam.cuda_prec_sloppy = gaugeParam.cuda_prec;

    // ensure that the default is improved staggered
  if (dslash_type != QUDA_STAGGERED_DSLASH &&
    dslash_type != QUDA_ASQTAD_DSLASH)
    dslash_type = QUDA_ASQTAD_DSLASH;

  gaugeParam.anisotropy = 1.0;
  gaugeParam.tadpole_coeff = 0.8;
  gaugeParam.scale = (dslash_type == QUDA_ASQTAD_DSLASH) ? -1.0/(24.0*gaugeParam.tadpole_coeff*gaugeParam.tadpole_coeff) : 1.0;
  gaugeParam.gauge_order = QUDA_MILC_GAUGE_ORDER;
  gaugeParam.t_boundary = QUDA_ANTI_PERIODIC_T;
  //gaugeParam.staggered_phase_type = QUDA_STAGGERED_PHASE_MILC;
  gaugeParam.gauge_fix = QUDA_GAUGE_FIXED_NO;
  gaugeParam.type = QUDA_WILSON_LINKS;

  inv_param.cpu_prec = QUDA_DOUBLE_PRECISION;
  inv_param.cuda_prec = prec;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dagger = dagger;
  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  inv_param.dslash_type = dslash_type;
  inv_param.mass = mass;
  inv_param.mass_normalization = QUDA_MASS_NORMALIZATION;

  // ensure that the default is improved staggered
  if (inv_param.dslash_type != QUDA_STAGGERED_DSLASH &&
    inv_param.dslash_type != QUDA_ASQTAD_DSLASH)
    inv_param.dslash_type = QUDA_ASQTAD_DSLASH;

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  int tmpint = MAX(X[1]*X[2]*X[3], X[0]*X[2]*X[3]);
  tmpint = MAX(tmpint, X[0]*X[1]*X[3]);
  tmpint = MAX(tmpint, X[0]*X[1]*X[2]);


  gaugeParam.ga_pad = tmpint;
  inv_param.sp_pad = tmpint;

  ColorSpinorParam csParam;
  csParam.nColor=3;
  csParam.nSpin=1;
  csParam.nDim=5;
  for(int d = 0; d < 4; d++) {
    csParam.x[d] = gaugeParam.X[d];
  }
  csParam.x[4] = Nsrc; // number of sources becomes the fifth dimension

  csParam.setPrecision(inv_param.cpu_prec);
  csParam.pad = 0;
  if (test_type < 2) {
    inv_param.solution_type = QUDA_MATPC_SOLUTION;
    csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    csParam.x[0] /= 2;
  } else {
    inv_param.solution_type = QUDA_MAT_SOLUTION;
    csParam.siteSubset = QUDA_FULL_SITE_SUBSET;	
  }

  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder  = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = inv_param.gamma_basis; // this parameter is meaningless for staggered
  csParam.create = QUDA_ZERO_FIELD_CREATE;    

  spinor = new cpuColorSpinorField(csParam);
  spinorOut = new cpuColorSpinorField(csParam);
  spinorRef = new cpuColorSpinorField(csParam);
  tmpCpu = new cpuColorSpinorField(csParam);

  csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  csParam.x[0] = gaugeParam.X[0];

  // printfQuda("Randomizing fields ...\n");

  spinor->Source(QUDA_RANDOM_SOURCE);
  /*int latDim[4] = {xdim,ydim,zdim,tdim};
  int coord[4] = {1,1,1,1};
  spinor->zero(); // zero before dropping a point source
  spinor->Source(QUDA_POINT_SOURCE, getPrintVectorIndex(latDim, coord), 0, 0);
  */

  size_t gSize = (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  // Allocate a lot of memory because I'm very confused
  fatlink_cpu_milc = malloc(4*V*gaugeSiteSize*gSize);
  longlink_cpu_milc = malloc(4*V*gaugeSiteSize*gSize);

  fatlink_gpu_milc = malloc(4*V*gaugeSiteSize*gSize);
  longlink_gpu_milc = malloc(4*V*gaugeSiteSize*gSize);

  for (int dir = 0; dir < 4; dir++) {
    fatlink_gpu[dir] = malloc(V*gaugeSiteSize*gSize);
    longlink_gpu[dir] = malloc(V*gaugeSiteSize*gSize);

    fatlink_cpu[dir] = malloc(V*gaugeSiteSize*gSize);
    longlink_cpu[dir] = malloc(V*gaugeSiteSize*gSize);

    if (fatlink_gpu[dir] == NULL || longlink_gpu[dir] == NULL ||
          fatlink_cpu[dir] == NULL || longlink_cpu[dir] == NULL) {
      errorQuda("ERROR: malloc failed for fatlink/longlink");
    }  
  }

  // create a base field
  void *inlink[4];
  for (int dir = 0; dir < 4; dir++) {
    inlink[dir] = malloc(V*gaugeSiteSize*gSize);
  }

  // load a field WITHOUT PHASES
  if (strcmp(latfile,"")) {
    read_gauge_field(latfile, inlink, gaugeParam.cpu_prec, gaugeParam.X, argc_copy, argv_copy);
  } else {
    //construct_fat_long_gauge_field(inlink, longlink_cpu, 1, gaugeParam.cpu_prec,&gaugeParam,dslash_type);
    createSiteLinkCPU(inlink, gaugeParam.cpu_prec, 0); // 0 for no phases
  }

  // If we're doing HISQ fields, we build links both on the CPU and the GPU.
  if (dslash_type == QUDA_ASQTAD_DSLASH) {

    ///////////////////////////
    // Set path coefficients //
    ///////////////////////////

    // Reference: "generic_ks/imp_actions/hisq/hisq_action.h"

    // First path: create V, W links 
    double act_path_coeff_1[6] = {
      ( 1.0/8.0),                 /* one link */
        0.0,                      /* Naik */
      (-1.0/8.0)*0.5,             /* simple staple */
      ( 1.0/8.0)*0.25*0.5,        /* displace link in two directions */
      (-1.0/8.0)*0.125*(1.0/6.0), /* displace link in three directions */
        0.0                       /* Lepage term */
    };

    // Second path: create X, long links
    double act_path_coeff_2[6] = {
      (( 1.0/8.0)+(2.0*6.0/16.0)+(1.0/8.0)),   /* one link */
          /* One link is 1/8 as in fat7 + 2*3/8 for Lepage + 1/8 for Naik */
      (-1.0/24.0),                             /* Naik */
      (-1.0/8.0)*0.5,                          /* simple staple */
      ( 1.0/8.0)*0.25*0.5,                     /* displace link in two directions */
      (-1.0/8.0)*0.125*(1.0/6.0),              /* displace link in three directions */
      (-2.0/16.0)                              /* Lepage term, correct O(a^2) 2x ASQTAD */
    };

    // Paths for epsilon corrections. Not used if n_naiks = 1.
    double act_path_coeff_3[6] = {
      ( 1.0/8.0),    /* one link b/c of Naik */
      (-1.0/24.0),   /* Naik */
        0.0,         /* simple staple */
        0.0,         /* displace link in two directions */
        0.0,         /* displace link in three directions */
        0.0          /* Lepage term */
    };

    // silence some Naik complaining
    (void)n_naiks;


    ////////////////////////////////////
    // Set unitarization coefficients //
    ////////////////////////////////////

    setUnitarizeLinksConstants(unitarize_eps,
             max_allowed_error,
             reunit_allow_svd,
             reunit_svd_only,
             svd_rel_error,
             svd_abs_error);

    //////////////////////////
    // Create the CPU links //
    //////////////////////////

    double* act_paths[3] = { act_path_coeff_1, act_path_coeff_2, act_path_coeff_3 };

    computeHISQLinksCPU(fatlink_cpu, longlink_cpu, 
                        nullptr, nullptr,
                        inlink, &gaugeParam, act_paths, 0.0 /*eps_naik*/);

    //////////////////////////
    // Create the GPU links //
    //////////////////////////

    // Skip eps field for now

    // GPU link creation only works for single and double precision

    if (prec == QUDA_SINGLE_PRECISION || prec == QUDA_DOUBLE_PRECISION) {
      

      // inlink in different format
      void *inlink_milc = pinned_malloc(4*V*gaugeSiteSize*gSize);
      for(int i=0; i<V; ++i){
        for(int dir=0; dir<4; ++dir){
          char* src = (char*)inlink[dir];
          memcpy((char*)inlink_milc + (i*4 + dir)*gaugeSiteSize*gSize, src+i*gaugeSiteSize*gSize, gaugeSiteSize*gSize);
        } 
      }

      // Paths for step 1:
      void* vlink_milc  = pinned_malloc(4*V*gaugeSiteSize*gSize); // V links
      void* wlink_milc  = pinned_malloc(4*V*gaugeSiteSize*gSize); // W links
      
      // Paths for step 2:
      void* fatlink_milc = pinned_malloc(4*V*gaugeSiteSize*gSize); // final fat ("X") links
      void* longlink_milc = pinned_malloc(4*V*gaugeSiteSize*gSize); // final long links
      
      // Create V links (fat7 links) and W links (unitarized V links), 1st path table set
      computeKSLinkQuda(vlink_milc, nullptr, wlink_milc, inlink_milc, act_path_coeff_1, &gaugeParam);

      // Create X and long links, 2nd path table set
      computeKSLinkQuda(fatlink_milc, longlink_milc, nullptr, wlink_milc, act_path_coeff_2, &gaugeParam);

      // Copy back
      for(int i=0; i < V; i++){
        for(int dir=0; dir< 4; dir++){
          char* src = ((char*)fatlink_milc )+ (4*i+dir)*gaugeSiteSize*gSize;
          char* dst = ((char*)fatlink_gpu [dir]) + i*gaugeSiteSize*gSize;
          memcpy(dst, src, gaugeSiteSize*gSize);

          src = ((char*)longlink_milc)+ (4*i+dir)*gaugeSiteSize*gSize;
          dst = ((char*)longlink_gpu[dir]) + i*gaugeSiteSize*gSize;
          memcpy(dst, src, gaugeSiteSize*gSize);
        }
      }

      // Clean up GPU compute links
      host_free(inlink_milc);
      host_free(vlink_milc);
      host_free(wlink_milc);
      host_free(fatlink_milc);
      host_free(longlink_milc);
    } else { // prec == QUDA_HALF_PRECISION

      for (int dir = 0; dir < 4; dir++) {
        memcpy(fatlink_gpu[dir],fatlink_cpu[dir], V*gaugeSiteSize*gSize);
        memcpy(longlink_gpu[dir],longlink_cpu[dir], V*gaugeSiteSize*gSize);
      }
    }

  } else {
    // we apply phases then copy it over
    applyGaugeFieldScaling_long(inlink, Vh, &gaugeParam, QUDA_STAGGERED_DSLASH, gaugeParam.cpu_prec);

    for (int dir = 0; dir < 4; dir++) {
      memcpy(fatlink_gpu[dir],inlink[dir], V*gaugeSiteSize*gSize);
      memcpy(fatlink_cpu[dir],inlink[dir], V*gaugeSiteSize*gSize);
      memset(longlink_gpu[dir],0,V*gaugeSiteSize*gSize);
      memset(longlink_cpu[dir],0,V*gaugeSiteSize*gSize);
    }
  }

  printfQuda("Ready to copy into milc pointers\n");

  // Alright, we've created all the void** links.
  // Create the void* pointers
  for(int dir=0; dir<4; ++dir){
    for(int i=0; i<V; ++i){
      for(int j=0; j<gaugeSiteSize; ++j){
        if(gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION){
          ((double*)fatlink_gpu_milc)[(i*4 + dir)*gaugeSiteSize + j] = ((double*)fatlink_gpu[dir])[i*gaugeSiteSize + j];
          ((double*)fatlink_cpu_milc)[(i*4 + dir)*gaugeSiteSize + j] = ((double*)fatlink_cpu[dir])[i*gaugeSiteSize + j];
          ((double*)longlink_gpu_milc)[(i*4 + dir)*gaugeSiteSize + j] = ((double*)longlink_gpu[dir])[i*gaugeSiteSize + j];
          ((double*)longlink_cpu_milc)[(i*4 + dir)*gaugeSiteSize + j] = ((double*)longlink_cpu[dir])[i*gaugeSiteSize + j];
        }else{
          ((float*)fatlink_gpu_milc)[(i*4 + dir)*gaugeSiteSize + j] = ((float*)fatlink_gpu[dir])[i*gaugeSiteSize + j];
          ((float*)fatlink_cpu_milc)[(i*4 + dir)*gaugeSiteSize + j] = ((float*)fatlink_cpu[dir])[i*gaugeSiteSize + j];
          ((double*)longlink_gpu_milc)[(i*4 + dir)*gaugeSiteSize + j] = ((double*)longlink_gpu[dir])[i*gaugeSiteSize + j];
          ((double*)longlink_cpu_milc)[(i*4 + dir)*gaugeSiteSize + j] = ((double*)longlink_cpu[dir])[i*gaugeSiteSize + j];
        }
      }
    }
  }

  printfQuda("Copied into milc pointers\n");

  // Create ghost zones for CPU fields,
  // prepare and load the GPU fields

#ifdef MULTI_GPU
  printf("About to create fatlink ghost\n");

  gaugeParam.type = (dslash_type == QUDA_ASQTAD_DSLASH) ? QUDA_ASQTAD_FAT_LINKS : QUDA_SU3_LINKS;
  gaugeParam.reconstruct = QUDA_RECONSTRUCT_NO;
  GaugeFieldParam cpuFatParam(fatlink_cpu_milc, gaugeParam);
  cpuFatParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuFat = new cpuGaugeField(cpuFatParam);
  ghost_fatlink_cpu = cpuFat->Ghost();

  printf("Created fatlink ghost\n");

  printf("About to create longlink ghost\n");
  gaugeParam.type = QUDA_ASQTAD_LONG_LINKS;
  GaugeFieldParam cpuLongParam(longlink_cpu_milc, gaugeParam);
  cpuLongParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuLong = new cpuGaugeField(cpuLongParam);
  ghost_longlink_cpu = cpuLong->Ghost();
  printf("Created longlink ghost\n");

  int x_face_size = X[1]*X[2]*X[3]/2;
  int y_face_size = X[0]*X[2]*X[3]/2;
  int z_face_size = X[0]*X[1]*X[3]/2;
  int t_face_size = X[0]*X[1]*X[2]/2;
  int pad_size = MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gaugeParam.ga_pad = pad_size;    
#endif

  gaugeParam.type = (dslash_type == QUDA_ASQTAD_DSLASH) ? QUDA_ASQTAD_FAT_LINKS : QUDA_SU3_LINKS;
  if (dslash_type == QUDA_STAGGERED_DSLASH) {
    gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = link_recon;
  } else {
    gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  }

  
  
  // printfQuda("Fat links sending..."); 
  loadGaugeQuda(fatlink_gpu_milc, &gaugeParam);
  // printfQuda("Fat links sent\n"); 

  gaugeParam.type = QUDA_ASQTAD_LONG_LINKS;  

#ifdef MULTI_GPU
  gaugeParam.ga_pad = 3*pad_size;
#endif

  if (dslash_type == QUDA_ASQTAD_DSLASH) {

    gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = (link_recon==QUDA_RECONSTRUCT_12) ? QUDA_RECONSTRUCT_13 : (link_recon==QUDA_RECONSTRUCT_8) ? QUDA_RECONSTRUCT_13 : link_recon;
    // printfQuda("Long links sending..."); 
    loadGaugeQuda(longlink_gpu_milc, &gaugeParam);
    // printfQuda("Long links sent...\n");
  }

  // printfQuda("Sending fields to GPU..."); 

  if (!transfer) {

    csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    csParam.pad = inv_param.sp_pad;
    csParam.setPrecision(inv_param.cuda_prec);
    if (test_type < 2){
      csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
      csParam.x[0] /=2;
    }

    // printfQuda("Creating cudaSpinor\n");
    cudaSpinor = new cudaColorSpinorField(csParam);

    // printfQuda("Creating cudaSpinorOut\n");
    cudaSpinorOut = new cudaColorSpinorField(csParam);

    // printfQuda("Sending spinor field to GPU\n");
    *cudaSpinor = *spinor;

    cudaDeviceSynchronize();
    checkCudaError();

    // double spinor_norm2 = blas::norm2(*spinor);
    // double cuda_spinor_norm2=  blas::norm2(*cudaSpinor);
    // printfQuda("Source CPU = %f, CUDA=%f\n", spinor_norm2, cuda_spinor_norm2);

    if(test_type == 2) csParam.x[0] /=2;

    csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    tmp = new cudaColorSpinorField(csParam);

    bool pc = (test_type != 2);
    DiracParam diracParam;
    setDiracParam(diracParam, &inv_param, pc);

    diracParam.tmp1=tmp;

    dirac = Dirac::create(diracParam);

  } else {
    errorQuda("Error not suppported");
  }

  return;
}

void end(void) 
{
  for (int dir = 0; dir < 4; dir++) {
    free(fatlink_gpu[dir]);
    free(longlink_gpu[dir]);
    free(fatlink_cpu[dir]);
    free(longlink_cpu[dir]);
  }
  free(fatlink_gpu_milc);
  free(longlink_gpu_milc);
  free(fatlink_cpu_milc);
  free(longlink_cpu_milc);

  if (!transfer){
    delete dirac;
    delete cudaSpinor;
    delete cudaSpinorOut;
    delete tmp;
  }

  delete spinor;
  delete spinorOut;
  delete spinorRef;
  delete tmpCpu;

  freeGaugeQuda();

  if (cpuFat) delete cpuFat;
  if (cpuLong) delete cpuLong;
  commDimPartitionedReset();
  
}

struct DslashTime {
  double event_time;
  double cpu_time;
  double cpu_min;
  double cpu_max;

  DslashTime() : event_time(0.0), cpu_time(0.0), cpu_min(DBL_MAX), cpu_max(0.0) {}
};

DslashTime dslashCUDA(int niter) {

  DslashTime dslash_time;
  timeval tstart, tstop;

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  cudaEventSynchronize(start);

  comm_barrier();
  cudaEventRecord(start, 0);

  for (int i = 0; i < niter; i++) {

    gettimeofday(&tstart, NULL);

    switch (test_type) {
      case 0:
      if (transfer){
          //dslashQuda(spinorOdd, spinorEven, &inv_param, parity);
      } else {
        dirac->Dslash(*cudaSpinorOut, *cudaSpinor, parity);
      }	   
      break;
      case 1:
      if (transfer){
          //MatPCDagMatPcQuda(spinorOdd, spinorEven, &inv_param);
      } else {
        dirac->MdagM(*cudaSpinorOut, *cudaSpinor);
      }
      break;
      case 2:
      errorQuda("Staggered operator acting on full-site not supported");
      if (transfer){
          //MatQuda(spinorGPU, spinor, &inv_param);
      } else {
        dirac->M(*cudaSpinorOut, *cudaSpinor);
      }
    }

    gettimeofday(&tstop, NULL);
    long ds = tstop.tv_sec - tstart.tv_sec;
    long dus = tstop.tv_usec - tstart.tv_usec;
    double elapsed = ds + 0.000001*dus;

    dslash_time.cpu_time += elapsed;
    // skip first and last iterations since they may skew these metrics if comms are not synchronous
    if (i>0 && i<niter) {
      if (elapsed < dslash_time.cpu_min) dslash_time.cpu_min = elapsed;
      if (elapsed > dslash_time.cpu_max) dslash_time.cpu_max = elapsed;
    }
  }

  

  cudaEventCreate(&end);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float runTime;
  cudaEventElapsedTime(&runTime, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  dslash_time.event_time = runTime / 1000;

  // check for errors
  cudaError_t stat = cudaGetLastError();
  if (stat != cudaSuccess)
    errorQuda("with ERROR: %s\n", cudaGetErrorString(stat));

  //printfQuda("CUDA Test\n");
  /*int latDim[4] = {xdim,ydim,zdim,tdim};
  int coord[4];
  for (int t = 0; t < 2; t++)
    for (int z = 0; z < 2; z++)
      for (int y = 0; y < 2; y++)
        for (int x = 0; x < 2; x++) {
          coord[0] = x, coord[1] = y; coord[2] = z; coord[3] = t;
          cudaSpinorOut->PrintVector(getPrintVectorIndex(latDim,coord));
        }*/
  //for (int i = 0; i < Vh; i++)    
  //  cudaSpinorOut->PrintVector(i);


  return dslash_time;
}

void staggeredDslashRef()
{

  // compare to dslash reference implementation
  // printfQuda("Calculating reference implementation...");
  fflush(stdout);
  switch (test_type) {
    case 0:
#ifdef MULTI_GPU
    staggered_dslash_mg4dir(spinorRef, fatlink_cpu, longlink_cpu, ghost_fatlink_cpu, ghost_longlink_cpu,
     spinor, parity, dagger, inv_param.cpu_prec, gaugeParam.cpu_prec);
#else
    staggered_dslash(spinorRef->V(), fatlink_cpu, longlink_cpu, spinor->V(), parity, dagger, inv_param.cpu_prec, gaugeParam.cpu_prec);
#endif    
    break;
    case 1:
#ifdef MULTI_GPU
    matdagmat_mg4dir(spinorRef, fatlink_cpu, longlink_cpu, ghost_fatlink_cpu, ghost_longlink_cpu,
     spinor, mass, 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmpCpu, parity);
#else
    matdagmat(spinorRef->V(), fatlink_cpu, longlink_cpu, spinor->V(), mass, 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmpCpu->V(), parity);
#endif
    break;
    case 2:
      //mat(spinorRef->V(), fatlink_cpu, longlink_cpu, spinor->V(), kappa, dagger, 
      //inv_param.cpu_prec, gaugeParam.cpu_prec);
    break;
    default:
    errorQuda("Test type not defined");
  }

  //printfQuda("CPU Test\n");
  //int latDim[4] = {xdim,ydim,zdim,tdim};
  /*int coord[4];
  for (int t = 0; t < 2; t++)
    for (int z = 0; z < 2; z++)
      for (int y = 0; y < 2; y++)
        for (int x = 0; x < 2; x++) {
          coord[0] = x, coord[1] = y; coord[2] = z; coord[3] = t;
          spinorRef->PrintVector(getPrintVectorIndex(latDim,coord));
        }
  */
  //for (int i = 0; i < Vh; i++)    
  //  spinorRef->PrintVector(i);

  // printfQuda("done.\n");
  //errorQuda("meh");

}


void display_test_info(int precision, QudaReconstructType link_recon)
{
  //auto prec = precision == 2 ? QUDA_DOUBLE_PRECISION : precision  == 1 ? QUDA_SINGLE_PRECISION : QUDA_HALF_PRECISION;
  auto prec = QUDA_DOUBLE_PRECISION;
  // printfQuda("running the following test:\n");
  // auto linkrecon = dslash_type == QUDA_ASQTAD_DSLASH ? (link_recon == QUDA_RECONSTRUCT_12 ?  QUDA_RECONSTRUCT_13 : (link_recon == QUDA_RECONSTRUCT_8 ? QUDA_RECONSTRUCT_9: link_recon)) : link_recon;
  printfQuda("prec recon   test_type     dagger   S_dim         T_dimension\n");
  printfQuda("%s   %s       %d           %d       %d/%d/%d        %d \n", 
    get_prec_str(prec), get_recon_str(link_recon), 
    test_type, dagger, xdim, ydim, zdim, tdim);
  // printfQuda("Grid partition info:     X  Y  Z  T\n"); 
  // printfQuda("                         %d  %d  %d  %d\n", 
  //     dimPartitioned(0),
  //     dimPartitioned(1),
  //     dimPartitioned(2),
  //     dimPartitioned(3));

  return ;

}

using ::testing::TestWithParam;
using ::testing::Bool;
using ::testing::Values;
using ::testing::Range;
using ::testing::Combine;


void usage_extra(char** argv )
{
  printfQuda("Extra options:\n");
  printfQuda("    --test <0/1>                             # Test method\n");
  printfQuda("                                                0: Even destination spinor\n");
  printfQuda("                                                1: Odd destination spinor\n");
  return ;
}

using ::testing::TestWithParam;
using ::testing::Bool;
using ::testing::Values;
using ::testing::Range;
using ::testing::Combine;

class StaggeredDslashTest : public ::testing::TestWithParam<::testing::tuple<int, int, int>> {
protected:
  ::testing::tuple<int, int, int> param;

public:
  virtual ~StaggeredDslashTest() { }
  virtual void SetUp() {
    int prec = ::testing::get<0>(GetParam());
    QudaReconstructType recon = static_cast<QudaReconstructType>(::testing::get<1>(GetParam()));


    int value = ::testing::get<2>(GetParam());
    for(int j=0; j < 4;j++){
      if (value &  (1 << j)){
        commDimPartitionedSet(j);
      }

    }
    updateR();
    init(prec, recon);
    display_test_info(prec, recon);
  }
  virtual void TearDown() { end(); }

  static void SetUpTestCase() {
    initQuda(device);
  }

  // Per-test-case tear-down.
  // Called after the last test in this test case.
  // Can be omitted if not needed.
  static void TearDownTestCase() {
    endQuda();
  }

};

 TEST_P(StaggeredDslashTest, verify) {
    { // warm-up run
      // printfQuda("Tuning...\n");
      dslashCUDA(1);
    }

    dslashCUDA(2);

    if (!transfer) *spinorOut = *cudaSpinorOut;

    staggeredDslashRef();
    double spinor_ref_norm2 = blas::norm2(*spinorRef);
    double spinor_out_norm2 =  blas::norm2(*spinorOut);

    if (!transfer) {
      double cuda_spinor_out_norm2 =  blas::norm2(*cudaSpinorOut);
      printfQuda("Results: CPU=%f, CUDA=%f, CPU-CUDA=%f\n",  spinor_ref_norm2, cuda_spinor_out_norm2,
       spinor_out_norm2);
    } else {
      printfQuda("Result: CPU=%f , CPU-CUDA=%f", spinor_ref_norm2, spinor_out_norm2);
    }

    double deviation = pow(10, -(double)(cpuColorSpinorField::Compare(*spinorRef, *spinorOut)));
    double tol = (inv_param.cuda_prec == QUDA_DOUBLE_PRECISION ? 1e-12 :
      (inv_param.cuda_prec == QUDA_SINGLE_PRECISION ? 1e-3 : 1e-1));
    ASSERT_LE(deviation, tol) << "CPU and CUDA implementations do not agree";
  }

TEST_P(StaggeredDslashTest, benchmark) {
    { // warm-up run
      // printfQuda("Tuning...\n");
      dslashCUDA(1);
    }

    // reset flop counter
    dirac->Flops();

    DslashTime dslash_time = dslashCUDA(niter);

    if (!transfer) *spinorOut = *cudaSpinorOut;

    printfQuda("%fus per kernel call\n", 1e6*dslash_time.event_time / niter);

    unsigned long long flops = dirac->Flops();
    double gflops=1.0e-9*flops/dslash_time.event_time;
    printfQuda("GFLOPS = %f\n", gflops );
    RecordProperty("Gflops", std::to_string(gflops));

    RecordProperty("Halo_bidirectitonal_BW_GPU", 1.0e-9*2*cudaSpinor->GhostBytes()*niter/dslash_time.event_time);
    RecordProperty("Halo_bidirectitonal_BW_CPU", 1.0e-9*2*cudaSpinor->GhostBytes()*niter/dslash_time.cpu_time);
    RecordProperty("Halo_bidirectitonal_BW_CPU_min", 1.0e-9*2*cudaSpinor->GhostBytes()/dslash_time.cpu_max);
    RecordProperty("Halo_bidirectitonal_BW_CPU_max", 1.0e-9*2*cudaSpinor->GhostBytes()/dslash_time.cpu_min);
    RecordProperty("Halo_message_size_bytes",2*cudaSpinor->GhostBytes());

    printfQuda("Effective halo bi-directional bandwidth (GB/s) GPU = %f ( CPU = %f, min = %f , max = %f ) for aggregate message size %lu bytes\n",
     1.0e-9*2*cudaSpinor->GhostBytes()*niter/dslash_time.event_time, 1.0e-9*2*cudaSpinor->GhostBytes()*niter/dslash_time.cpu_time,
     1.0e-9*2*cudaSpinor->GhostBytes()/dslash_time.cpu_max, 1.0e-9*2*cudaSpinor->GhostBytes()/dslash_time.cpu_min,
     2*cudaSpinor->GhostBytes());

  }

  int main(int argc, char **argv) 
  {
    // hack for loading gauge fields
    argc_copy = argc;
    argv_copy = argv;

  // initalize google test
    ::testing::InitGoogleTest(&argc, argv);
    for (int i=1 ;i < argc; i++){

      if(process_command_line_option(argc, argv, &i) == 0){
        continue;
      }    

      fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
      usage(argv);
    }

    initComms(argc, argv, gridsize_from_cmdline);


  // return result of RUN_ALL_TESTS
    int test_rc = RUN_ALL_TESTS();

    finalizeComms();

    return test_rc;
  }

  std::string getstaggereddslashtestname(testing::TestParamInfo<::testing::tuple<int, int, int>> param){
   const int prec = ::testing::get<0>(param.param);
   const int recon = ::testing::get<1>(param.param);
   const int part = ::testing::get<2>(param.param);
   std::stringstream ss;
   // ss << get_dslash_str(dslash_type) << "_";
   ss << prec_str[prec];
   ss << "_r" << recon;
   ss << "_partition" << part;
   return ss.str();
 }


#ifdef MULTI_GPU
 INSTANTIATE_TEST_CASE_P(QUDA, StaggeredDslashTest, Combine( Range(0,1)/*Range(0,3)*/, ::testing::Values(QUDA_RECONSTRUCT_NO,QUDA_RECONSTRUCT_12,QUDA_RECONSTRUCT_8), Range(0,16)),getstaggereddslashtestname);
#else
 INSTANTIATE_TEST_CASE_P(QUDA, StaggeredDslashTest, Combine( /*Range(0,3)*/Range(0,1), ::testing::Values(QUDA_RECONSTRUCT_NO,QUDA_RECONSTRUCT_12,QUDA_RECONSTRUCT_8), ::testing::Values(0) ),getstaggereddslashtestname);
#endif

