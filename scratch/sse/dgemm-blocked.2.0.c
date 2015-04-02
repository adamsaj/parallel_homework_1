/* 
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
 
CC = cc
OPT = -O3
CFLAGS = -Wall -std=gnu99 -unroll-loops $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/

#include <emmintrin.h>
#define Z_SIZE 10000
const char* dgemm_desc = "Simple blocked dgemm.";
void SIMD_do_block_4 (double *M, double *N, double *K);
double Aprime[Z_SIZE], Bprime[Z_SIZE], Cprime[Z_SIZE];

#define BLOCK_SIZE_l1 4
//#define BLOCK_SIZE_l2 30
#define BLOCK_SIZE_l3 512

#define min(a,b) (((a)<(b))?(a):(b))

unsigned int interleave(unsigned int x, unsigned int y); //x is row, y is column

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block_l1 (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j) 
    {
      /* Compute C(i,j) */
      double cij = 0, cij_1=0, cij_2=0, cij_3=0;//, cij_4=0, cij_5=0, cij_6=0, cij_7=0;
	int k = 0;
	for (; k < K - 3; k += 4)
	{
		cij += A[i+k*lda] * B[k+j*lda];
		cij_1 += A[i+(k+1)*lda] * B[(k+1)+j*lda];
		cij_2 += A[i+(k+2)*lda] * B[(k+2)+j*lda];
		cij_3 += A[i+(k+3)*lda] * B[(k+3)+j*lda];
//		cij_4 += A[i+(k+4)*lda] * B[(k+4)+j*lda];
//		cij_5 += A[i+(k+5)*lda] * B[(k+5)+j*lda];
//		cij_6 += A[i+(k+6)*lda] * B[(k+6)+j*lda];
//		cij_7 += A[i+(k+7)*lda] * B[(k+7)+j*lda];
	}
//      for (; k < K - 1; k += 2)
//	{
//		cij += (A[i+k*lda] * B[k+j*lda]) + (A[i+(k+1)*lda] * B[(k+1)+j*lda]);
//	}
      for (; k < K; ++k)
	cij += A[i+k*lda] * B[k+j*lda];
//	cij += cij_1 + cij_2 + cij_3;
//	cij_4 += cij_5 + cij_6 + cij_7;
	cij += cij_1;
	cij_2 += cij_3;
      C[i+j*lda] += cij + cij_2;
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
//void do_block_l2 (int lda, int l, int m, int n, double* A, double* B, double* C)
//{
//  /* For each block-row of A */ 
//  for (int i = 0; i < l; i += BLOCK_SIZE_l1)
//    /* For each block-column of B */
//    for (int j = 0; j < m; j += BLOCK_SIZE_l1)
//      /* Accumulate block dgemms into block of C */
//      for (int k = 0; k < n; k += BLOCK_SIZE_l1)
//      {
//	/* Correct block dimensions if block "goes off edge of" the matrix */
//	int M = min (BLOCK_SIZE_l1, l-i);
//	int N = min (BLOCK_SIZE_l1, m-j);
//	int K = min (BLOCK_SIZE_l1, n-k);
//
//	/* Perform individual block dgemm */
//	do_block_l1(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
//      }
//}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void do_block_l3 (int lda, int l, int m, int n, double* A, double* B, double* C)
{
  /* For each block-row of A */ 
  for (int i = 0; i < l; i += BLOCK_SIZE_l1)
    /* For each block-column of B */
    for (int j = 0; j < m; j += BLOCK_SIZE_l1)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < n; k += BLOCK_SIZE_l1)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	int M = min (BLOCK_SIZE_l1, l-i);
	int N = min (BLOCK_SIZE_l1, m-j);
	int K = min (BLOCK_SIZE_l1, n-k);

	/* Perform individual block dgemm */
	if (M==4 && N==4 && K==4)
		SIMD_do_block_4((A+interleave(i,k)),(B+interleave(k,j)),(C+interleave(i,j)));
	else
		do_block_l1(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
	for (unsigned int i=0; i<lda; ++i)
		for (unsigned int j=0; j<lda; ++j)
			{
			*(Aprime + interleave(i,j))=A[i+j*lda];
			*(Bprime + interleave(i,j))=B[i+j*lda];
			*(Cprime + interleave(i,j))=0;
			}
			
  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += BLOCK_SIZE_l3)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE_l3)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE_l3)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	int M = min (BLOCK_SIZE_l3, lda-i);
	int N = min (BLOCK_SIZE_l3, lda-j);
	int K = min (BLOCK_SIZE_l3, lda-k);

	/* Perform individual block dgemm */
	do_block_l3(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
	for (unsigned int i=0; i<lda; ++i)
		for (unsigned int j=0; j<lda; ++j)
			C[i+j*lda]=*(Cprime + interleave(i,j));
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

void SIMD_do_block_4 (double *M, double *N, double *K)
/*this function multiplies 4x4 matricies stored in column-major-Z order*/
{
	__m128d c0 = _mm_loadu_pd(K);
	__m128d c2 = _mm_loadu_pd(K+2);
	__m128d c4 = _mm_loadu_pd(K+4);
	__m128d c6 = _mm_loadu_pd(K+6);
	__m128d c8 = _mm_loadu_pd(K+8);
	__m128d c10 = _mm_loadu_pd(K+10);
	__m128d c12 = _mm_loadu_pd(K+12);
	__m128d c14 = _mm_loadu_pd(K+14);

	__m128d a1 = _mm_loadu_pd(M);
	__m128d a2 = _mm_loadu_pd(M+4);
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

	a1 = _mm_loadu_pd(M+2);
	a2 = _mm_loadu_pd(M+6);
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

	a1 = _mm_loadu_pd(M+8);
	a2 = _mm_loadu_pd(M+12);
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

	a1 = _mm_loadu_pd(M+10);
	a2 = _mm_loadu_pd(M+14);
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
