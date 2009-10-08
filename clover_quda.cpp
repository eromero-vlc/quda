#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <quda.h>
#include <spinor_quda.h>
#include <util_quda.h>


ParityClover allocateParityClover(int *X, Precision precision)
{
  ParityClover ret;

  ret.precision = precision;
  ret.volume = 1;
  for (int d=0; d<4; d++) {
    ret.X[d] = X[d];
    ret.volume *= X[d];
  }
  ret.Nc = 3;
  ret.Ns = 4;
  ret.length = ret.volume*ret.Nc*ret.Nc*ret.Ns*ret.Ns/2; // block-diagonal Hermitian (72 reals)

  if (precision == QUDA_DOUBLE_PRECISION) ret.bytes = ret.length*sizeof(double);
  else if (precision == QUDA_SINGLE_PRECISION) ret.bytes = ret.length*sizeof(float);
  else ret.bytes = ret.length*sizeof(short);

  if (cudaMalloc((void**)&(ret.clover), ret.bytes) == cudaErrorMemoryAllocation) {
    printf("Error allocating clover term\n");
    exit(0);
  }   

  if (precision == QUDA_HALF_PRECISION) {
    if (cudaMalloc((void**)&ret.cloverNorm, ret.bytes/18) == cudaErrorMemoryAllocation) {
      printf("Error allocating cloverNorm\n");
      exit(0);
    }
  }

  return ret;
}

FullClover allocateCloverField(int *X, Precision precision)
{
  FullClover ret;
  ret.even = allocateParityClover(X, precision);
  ret.odd = allocateParityClover(X, precision);
  return ret;
}

void freeParityClover(ParityClover *clover)
{
  cudaFree(clover->clover);
  clover->clover = NULL;
}

void freeCloverField(FullClover *clover)
{
  freeParityClover(&clover->even);
  freeParityClover(&clover->odd);
}

template <typename Float>
static inline void packCloverMatrix(float4* a, Float *b, int Vh)
{
  const Float half = 0.5; // pre-include factor of 1/2 introduced by basis change

  for (int i=0; i<18; i++) {
    a[i*Vh].x = half * b[4*i+0];
    a[i*Vh].y = half * b[4*i+1];
    a[i*Vh].z = half * b[4*i+2];
    a[i*Vh].w = half * b[4*i+3];
  }
}

template <typename Float>
static inline void packCloverMatrix(double2* a, Float *b, int Vh)
{
  const Float half = 0.5; // pre-include factor of 1/2 introduced by basis change

  for (int i=0; i<36; i++) {
    a[i*Vh].x = half * b[2*i+0];
    a[i*Vh].y = half * b[2*i+1];
  }
}

template <typename Float, typename FloatN>
static void packParityClover(FloatN *res, Float *clover, int Vh)
{
  for (int i = 0; i < Vh; i++) {
    packCloverMatrix(res+i, clover+72*i, Vh);
  }
}

template <typename Float, typename FloatN>
static void packFullClover(FloatN *even, FloatN *odd, Float *clover, int *X)
{
  int Vh = X[0]*X[1]*X[2]*X[3];
  X[0] *= 2; // X now contains dimensions of the full lattice

  for (int i=0; i<Vh; i++) {

    int boundaryCrossings = i/X[0] + i/(X[1]*X[0]) + i/(X[2]*X[1]*X[0]);

    { // even sites
      int k = 2*i + boundaryCrossings%2; 
      packCloverMatrix(even+i, clover+72*k, Vh);
    }
    
    { // odd sites
      int k = 2*i + (boundaryCrossings+1)%2;
      packCloverMatrix(odd+i, clover+72*k, Vh);
    }
  }
}

template<typename Float>
static inline void packCloverMatrixHalf(short4 *res, float *norm, Float *clover, int Vh)
{
  const Float half = 0.5; // pre-include factor of 1/2 introduced by basis change
  Float max, a, c;

  // treat the two chiral blocks separately
  for (int chi=0; chi<2; chi++) {
    max = fabs(clover[0]);
    for (int i=1; i<36; i++) {
      if ((a = fabs(clover[i])) > max) max = a;
    }
    c = MAX_SHORT/max;
    for (int i=0; i<9; i++) {
      res[i*Vh].x = (short) (c * clover[4*i+0]);
      res[i*Vh].y = (short) (c * clover[4*i+1]);
      res[i*Vh].z = (short) (c * clover[4*i+2]);
      res[i*Vh].w = (short) (c * clover[4*i+3]);
    }
    norm[chi*Vh] = half*max;
    res += 9*Vh;
    clover += 36;
  }
}

template <typename Float>
static void packParityCloverHalf(short4 *res, float *norm, Float *clover, int Vh)
{
  for (int i = 0; i < Vh; i++) {
    packCloverMatrixHalf(res+i, norm+i, clover+72*i, Vh);
  }
}

template <typename Float>
static void packFullCloverHalf(short4 *even, float *evenNorm, short4 *odd, float *oddNorm,
			       Float *clover, int *X)
{
  int Vh = X[0]*X[1]*X[2]*X[3];
  X[0] *= 2; // X now contains dimensions of the full lattice

  for (int i=0; i<Vh; i++) {

    int boundaryCrossings = i/X[0] + i/(X[1]*X[0]) + i/(X[2]*X[1]*X[0]);

    { // even sites
      int k = 2*i + boundaryCrossings%2; 
      packCloverMatrixHalf(even+i, evenNorm+i, clover+72*k, Vh);
    }
    
    { // odd sites
      int k = 2*i + (boundaryCrossings+1)%2;
      packCloverMatrixHalf(odd+i, oddNorm+i, clover+72*k, Vh);
    }
  }
}

void loadParityClover(ParityClover ret, void *clover, Precision cpu_prec, 
		      CloverFieldOrder clover_order)
{
  // use pinned memory                                                                                           
  void *packedClover, *packedCloverNorm;

  if (ret.precision == QUDA_DOUBLE_PRECISION && cpu_prec != QUDA_DOUBLE_PRECISION) {
    printf("QUDA error: cannot have CUDA double precision without double CPU precision\n");
    exit(-1);
  }
  if (clover_order != QUDA_PACKED_CLOVER_ORDER) {
    printf("QUDA error: invalid clover order\n");
    exit(-1);
  }

#ifndef __DEVICE_EMULATION__
  if (cudaMallocHost(&packedClover, ret.bytes) == cudaErrorMemoryAllocation) {
    printf("Error allocating clover pinned memory\n");
    exit(0);
  }  
  if (ret.precision == QUDA_HALF_PRECISION) 
    if (cudaMallocHost(&packedCloverNorm, ret.bytes/18) == cudaErrorMemoryAllocation) {
      printf("Error allocating clover pinned memory\n");
      exit(0);
    } 
#else
  packedClover = malloc(ret.bytes);
  if (ret.precision == QUDA_HALF_PRECISION) packedCloverNorm = malloc(ret.bytes/18);
#endif
    
  if (ret.precision == QUDA_DOUBLE_PRECISION) {
    packParityClover((double2 *)packedClover, (double *)clover, ret.volume);
  } else if (ret.precision == QUDA_SINGLE_PRECISION) {
    if (cpu_prec == QUDA_DOUBLE_PRECISION) {
      packParityClover((float4 *)packedClover, (double *)clover, ret.volume);
    } else {
      packParityClover((float4 *)packedClover, (float *)clover, ret.volume);
    }
  } else {
    if (cpu_prec == QUDA_DOUBLE_PRECISION) {
      packParityCloverHalf((short4 *)packedClover, (float *)packedCloverNorm, (double *)clover, ret.volume);
    } else {
      packParityCloverHalf((short4 *)packedClover, (float *)packedCloverNorm, (float *)clover, ret.volume);
    }
  }
  
  cudaMemcpy(ret.clover, packedClover, ret.bytes, cudaMemcpyHostToDevice);
  if (ret.precision == QUDA_HALF_PRECISION) {
    cudaMemcpy(ret.cloverNorm, packedCloverNorm, ret.bytes/18, cudaMemcpyHostToDevice);
  }

#ifndef __DEVICE_EMULATION__
  cudaFreeHost(packedClover);
  if (ret.precision == QUDA_HALF_PRECISION) cudaFreeHost(packedCloverNorm);
#else
  free(packedClover);
  if (ret.precision == QUDA_HALF_PRECISION) free(packedCloverNorm);
#endif

}

void loadFullClover(FullClover ret, void *clover, Precision cpu_prec,
		    CloverFieldOrder clover_order)
{
  // use pinned memory                                                                                           
  void *packedEven, *packedEvenNorm, *packedOdd, *packedOddNorm;

  if (ret.even.precision == QUDA_DOUBLE_PRECISION && cpu_prec != QUDA_DOUBLE_PRECISION) {
    printf("QUDA error: cannot have CUDA double precision without double CPU precision\n");
    exit(-1);
  }
  if (clover_order != QUDA_LEX_PACKED_CLOVER_ORDER) {
    printf("QUDA error: invalid clover order\n");
    exit(-1);
  }

#ifndef __DEVICE_EMULATION__
  cudaMallocHost(&packedEven, ret.even.bytes);
  cudaMallocHost(&packedOdd, ret.even.bytes);
  if (ret.even.precision == QUDA_HALF_PRECISION) {
    cudaMallocHost(&packedEvenNorm, ret.even.bytes/18);
    cudaMallocHost(&packedOddNorm, ret.even.bytes/18);
  }
#else
  packedEven = malloc(ret.even.bytes);
  packedOdd = malloc(ret.even.bytes);
  if (ret.even.precision == QUDA_HALF_PRECISION) {
    packedEvenNorm = malloc(ret.even.bytes/18);
    packedOddNorm = malloc(ret.even.bytes/18);
  }
#endif
    
  if (ret.even.precision == QUDA_DOUBLE_PRECISION) {
    packFullClover((double2 *)packedEven, (double2 *)packedOdd, (double *)clover, ret.even.X);
  } else if (ret.even.precision == QUDA_SINGLE_PRECISION) {
    if (cpu_prec == QUDA_DOUBLE_PRECISION) {
      packFullClover((float4 *)packedEven, (float4 *)packedOdd, (double *)clover, ret.even.X);
    } else {
      packFullClover((float4 *)packedEven, (float4 *)packedOdd, (float *)clover, ret.even.X);    
    }
  } else {
    if (cpu_prec == QUDA_DOUBLE_PRECISION) {
      packFullCloverHalf((short4 *)packedEven, (float *)packedEvenNorm, (short4 *)packedOdd,
			 (float *) packedOddNorm, (double *)clover, ret.even.X);
    } else {
      packFullCloverHalf((short4 *)packedEven, (float *)packedEvenNorm, (short4 *)packedOdd,
			 (float * )packedOddNorm, (float *)clover, ret.even.X);    
    }
  }

  cudaMemcpy(ret.even.clover, packedEven, ret.even.bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(ret.odd.clover, packedOdd, ret.even.bytes, cudaMemcpyHostToDevice);
  if (ret.even.precision == QUDA_HALF_PRECISION) {
    cudaMemcpy(ret.even.cloverNorm, packedEvenNorm, ret.even.bytes/18, cudaMemcpyHostToDevice);
    cudaMemcpy(ret.odd.cloverNorm, packedOddNorm, ret.even.bytes/18, cudaMemcpyHostToDevice);
  }

#ifndef __DEVICE_EMULATION__
  cudaFreeHost(packedEven);
  cudaFreeHost(packedOdd);
  if (ret.even.precision == QUDA_HALF_PRECISION) {
    cudaFreeHost(packedEvenNorm);
    cudaFreeHost(packedOddNorm);
  }
#else
  free(packedEven);
  free(packedOdd);
  if (ret.even.precision == QUDA_HALF_PRECISION) {
    free(packedEvenNorm);
    free(packedOddNorm);
  }
#endif

}

void loadCloverField(FullClover ret, void *clover, Precision cpu_prec, CloverFieldOrder clover_order)
{
  void *clover_odd;

  if (cpu_prec == QUDA_SINGLE_PRECISION) clover_odd = (float *)clover + ret.even.length;
  else clover_odd = (double *)clover + ret.even.length;

  if (clover_order == QUDA_LEX_PACKED_CLOVER_ORDER) {
    loadFullClover(ret, clover, cpu_prec, clover_order);
  } else if (clover_order == QUDA_PACKED_CLOVER_ORDER) {
    loadParityClover(ret.even, clover, cpu_prec, clover_order);
    loadParityClover(ret.odd, clover_odd, cpu_prec, clover_order);
  } else {
    printf("QUDA error: CloverFieldOrder %d not supported\n", clover_order);
    exit(-1);
  }
}

/*
void createCloverField(FullClover *cudaClover, void *cpuClover, int *X, Precision precision)
{
  if (invert_param->clover_cpu_prec == QUDA_HALF_PRECISION) {
    printf("QUDA error: half precision not supported on cpu\n");
    exit(-1);
  }

  // X should contain the dimensions of the even/odd sublattice
  *cudaClover = allocateCloverField(X, precision);
  loadCloverField(*cudaClover, cpuClover, precision, invert_param->clover_order);
}
*/
