/* 
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
 
CC = cc
OPT = -O3
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/

unsigned int interleave(unsigned int x, unsigned int y); //x is row, y is column
const char* dgemm_desc = "Simple blocked dgemm.";
#include <emmintrin.h>
#define Z_SIZE 1000000
void SIMD_do_block_4 (double *M, double *N, double *K);
double Aprime[Z_SIZE], Bprime[Z_SIZE], Cprime[Z_SIZE];

#define BLOCK_SIZE 4
#define BLOCK_SIZE1 64

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each row i of A */
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j) 
  for (int i = 0; i < M; ++i)
    {
      /* Compute C(i,j) */
      double cij = C[i+j*lda];
      for (int k = 0; k < K; ++k)
	cij += A[i+k*lda] * B[k+j*lda];
      C[i+j*lda] = cij;
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void do_block_1 (int lda, int M, int N, int K, double* A, double* B, double* C, double* Ap, double* Bp, double* Cp)
{
  /* For each block-row of A */ 
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
  for (int i = 0; i < lda; i += BLOCK_SIZE)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	int r = min (BLOCK_SIZE, M-i);
	int s = min (BLOCK_SIZE, N-j);
	int t = min (BLOCK_SIZE, K-k);

	/* Perform individual block dgemm */
	if (r==4 && s==4 && t==4)
		SIMD_do_block_4((Ap+interleave(i,k)),(Bp+interleave(k,j)),(Cp+interleave(i,j)));
	else
		do_block(lda, r, s, t, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
}
/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
	int zmax=lda-lda%4;
	for (unsigned int j=0; j<zmax; ++j)
		for (unsigned int i=0; i<zmax; ++i)
			{
			*(Aprime + interleave(i,j))=A[i+j*lda];
			*(Bprime + interleave(i,j))=B[i+j*lda];
			*(Cprime + interleave(i,j))=0;
			}
  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += BLOCK_SIZE1)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE1)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE1)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	int M = min (BLOCK_SIZE1, lda-i);
	int N = min (BLOCK_SIZE1, lda-j);
	int K = min (BLOCK_SIZE1, lda-k);

	/* Perform individual block dgemm */
		do_block_1(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda, Aprime + interleave(i,k), Bprime + interleave(k,j), Cprime + interleave(i,j));
      }
	for (unsigned int i=0; i<lda; ++i)
		for (unsigned int j=0; j<zmax; ++j)
			C[i+j*lda]+=*(Cprime + interleave(i,j));
}

unsigned int interleave(unsigned int x, unsigned int y) //x is row, y is column
{
        unsigned int z = 0;
        static const unsigned int B[] = {0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF};
        static const unsigned int S[] = {1, 2, 4, 8};

        x = (x | (x << S[3])) & B[3];
        x = (x | (x << S[2])) & B[2];
        x = (x | (x << S[1])) & B[1];
        x = (x | (x << S[0])) & B[0];

        y = (y | (y << S[3])) & B[3];
        y = (y | (y << S[2])) & B[2];
        y = (y | (y << S[1])) & B[1];
        y = (y | (y << S[0])) & B[0];

        z = x | (y << 1);

        return z;
}

void SIMD_do_block_4 (register double *M,register double *N, register double *K)
/*this function multiplies 4x4 matricies stored in column-major-Z order*/
{
	__m128d c0 = _mm_load_pd(K);
	__m128d c2 = _mm_load_pd(K+2);
	__m128d c4 = _mm_load_pd(K+4);
	__m128d c6 = _mm_load_pd(K+6);
	__m128d c8 = _mm_load_pd(K+8);
	__m128d c10 = _mm_load_pd(K+10);
	__m128d c12 = _mm_load_pd(K+12);
	__m128d c14 = _mm_load_pd(K+14);

	__m128d a1 = _mm_load_pd(M);
	__m128d a2 = _mm_load_pd(M+4);
	__m128d b1 = _mm_load1_pd(N);
	__m128d b2 = _mm_load1_pd(N+2);
	__m128d b3 = _mm_load1_pd(N+8);
	__m128d b4 = _mm_load1_pd(N+10);

	c0 = _mm_add_pd( c0, _mm_mul_pd(a1,b1));
	c2 = _mm_add_pd( c2, _mm_mul_pd(a1,b2));
	c4 = _mm_add_pd( c4, _mm_mul_pd(a2,b1));
	c6 = _mm_add_pd( c6, _mm_mul_pd(a2,b2));
	c8 = _mm_add_pd( c8, _mm_mul_pd(a1,b3));
	c10 = _mm_add_pd( c10, _mm_mul_pd(a1,b4));
	c12 = _mm_add_pd( c12, _mm_mul_pd(a2,b3));
	c14 = _mm_add_pd( c14, _mm_mul_pd(a2,b4));

	a1 = _mm_load_pd(M+2);
	a2 = _mm_load_pd(M+6);
	b1 = _mm_load1_pd(N+1);
	b2 = _mm_load1_pd(N+3);
	b3 = _mm_load1_pd(N+9);
	b4 = _mm_load1_pd(N+11);

	c0 = _mm_add_pd( c0, _mm_mul_pd(a1,b1));
	c2 = _mm_add_pd( c2, _mm_mul_pd(a1,b2));
	c4 = _mm_add_pd( c4, _mm_mul_pd(a2,b1));
	c6 = _mm_add_pd( c6, _mm_mul_pd(a2,b2));
	c8 = _mm_add_pd( c8, _mm_mul_pd(a1,b3));
	c10 = _mm_add_pd( c10, _mm_mul_pd(a1,b4));
	c12 = _mm_add_pd( c12, _mm_mul_pd(a2,b3));
	c14 = _mm_add_pd( c14, _mm_mul_pd(a2,b4));

	a1 = _mm_load_pd(M+8);
	a2 = _mm_load_pd(M+12);
	b1 = _mm_load1_pd(N+4);
	b2 = _mm_load1_pd(N+6);
	b3 = _mm_load1_pd(N+12);
	b4 = _mm_load1_pd(N+14);

	c0 = _mm_add_pd( c0, _mm_mul_pd(a1,b1));
	c2 = _mm_add_pd( c2, _mm_mul_pd(a1,b2));
	c4 = _mm_add_pd( c4, _mm_mul_pd(a2,b1));
	c6 = _mm_add_pd( c6, _mm_mul_pd(a2,b2));
	c8 = _mm_add_pd( c8, _mm_mul_pd(a1,b3));
	c10 = _mm_add_pd( c10, _mm_mul_pd(a1,b4));
	c12 = _mm_add_pd( c12, _mm_mul_pd(a2,b3));
	c14 = _mm_add_pd( c14, _mm_mul_pd(a2,b4));

	a1 = _mm_load_pd(M+10);
	a2 = _mm_load_pd(M+14);
	b1 = _mm_load1_pd(N+5);
	b2 = _mm_load1_pd(N+7);
	b3 = _mm_load1_pd(N+13);
	b4 = _mm_load1_pd(N+15);

	c0 = _mm_add_pd( c0, _mm_mul_pd(a1,b1));
	c2 = _mm_add_pd( c2, _mm_mul_pd(a1,b2));
	c4 = _mm_add_pd( c4, _mm_mul_pd(a2,b1));
	c6 = _mm_add_pd( c6, _mm_mul_pd(a2,b2));
	c8 = _mm_add_pd( c8, _mm_mul_pd(a1,b3));
	c10 = _mm_add_pd( c10, _mm_mul_pd(a1,b4));
	c12 = _mm_add_pd( c12, _mm_mul_pd(a2,b3));
	c14 = _mm_add_pd( c14, _mm_mul_pd(a2,b4));

	_mm_storeu_pd(K,c0);
	_mm_storeu_pd(K+2,c2);
	_mm_storeu_pd(K+4,c4);
	_mm_storeu_pd(K+6,c6);
	_mm_storeu_pd(K+8,c8);
	_mm_storeu_pd(K+10,c10);
	_mm_storeu_pd(K+12,c12);
	_mm_storeu_pd(K+14,c14);
}
