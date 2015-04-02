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
const char* dgemm_desc = "Simple blocked dgemm.";
static void simd_2x2(int lda, int M, int N, int K,double* A, double* B, double* C);

#define BLOCK_SIZE_l1 4
#define BLOCK_SIZE_l2 32
#define BLOCK_SIZE_l3 512

#define min(a,b) (((a)<(b))?(a):(b))

//unsigned int interleave(unsigned int x, unsigned int y); //x is row, y is column
//double Aprime[10000], Bprime[10000];

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
void do_block_l2 (int lda, int l, int m, int n, double* A, double* B, double* C)
{
  /* For each block-row of A */ 
    /* For each block-column of B */
    for (int j = 0; j < m; j += BLOCK_SIZE_l1)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < n; k += BLOCK_SIZE_l1)
  for (int i = 0; i < l; i += BLOCK_SIZE_l1)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	int M = min (BLOCK_SIZE_l1, l-i);
	int N = min (BLOCK_SIZE_l1, m-j);
	int K = min (BLOCK_SIZE_l1, n-k);

	/* Perform individual block dgemm */
				if (M == 4 && N==4 && K==4)
					simd_2x2(lda, i + k*lda, k+j*lda , i+j*lda, A , B , C);
				else
	do_block_l1(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void do_block_l3 (int lda, int l, int m, int n, double* A, double* B, double* C)
{
  /* For each block-row of A */ 
  for (int i = 0; i < l; i += BLOCK_SIZE_l2)
    /* For each block-column of B */
    for (int j = 0; j < m; j += BLOCK_SIZE_l2)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < n; k += BLOCK_SIZE_l2)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	int M = min (BLOCK_SIZE_l2, l-i);
	int N = min (BLOCK_SIZE_l2, m-j);
	int K = min (BLOCK_SIZE_l2, n-k);

	/* Perform individual block dgemm */
	do_block_l2(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
//	for (unsigned int i=0; i<lda; ++i)
//		for (unsigned int j=0; j<lda; ++j)
//			{
//			*(Aprime + interleave(i,j))=A[i+j*lda];
//			*(Bprime + interleave(j,i))=B[i+j*lda];
//			}
			
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
}


//unsigned int interleave(unsigned int x, unsigned int y) //x is row, y is column
//{
//        unsigned int z = 0;
//        static const unsigned int B[] = {0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF};
//        static const unsigned int S[] = {1, 2, 4, 8};
//
//        x = (x | (x << S[3])) & B[3];
//        x = (x | (x << S[2])) & B[2];
//        x = (x | (x << S[1])) & B[1];
//        x = (x | (x << S[0])) & B[0];
//
//        y = (y | (y << S[3])) & B[3];
//        y = (y | (y << S[2])) & B[2];
//        y = (y | (y << S[1])) & B[1];
//        y = (y | (y << S[0])) & B[0];
//
//        z = x | (y << 1);
//
//        return z;
//}

static void simd_2x2(int lda, int M, int N, int K,double* A, double* B, double* C)
{
	__m128d c1 = _mm_loadu_pd(C+K+0+lda*0);
	__m128d c2 = _mm_loadu_pd(C+K+2+lda*0);
	__m128d c3 = _mm_loadu_pd(C+K+0+lda*1);
	__m128d c4 = _mm_loadu_pd(C+K+2+lda*1);
	__m128d c5 = _mm_loadu_pd(C+K+0+lda*2);
	__m128d c6 = _mm_loadu_pd(C+K+2+lda*2);
	__m128d c7 = _mm_loadu_pd(C+K+0+lda*3);
	__m128d c8 = _mm_loadu_pd(C+K+2+lda*3);

	for( int i = 0; i < 4; ++i )
	{
		__m128d a01 = _mm_loadu_pd(A+M+0+lda*i);
		__m128d a23 = _mm_loadu_pd(A+M+2+lda*i);
		__m128d b0  = _mm_load1_pd(B+N+i+lda*0);
		__m128d b4  = _mm_load1_pd(B+N+i+lda*1);
		__m128d b8  = _mm_load1_pd(B+N+i+lda*2);
		__m128d b12 = _mm_load1_pd(B+N+i+lda*3);

		c1 = _mm_add_pd( c1, _mm_mul_pd( a01, b0 ));
		c2 = _mm_add_pd( c2, _mm_mul_pd( a23, b0 ));
		c3 = _mm_add_pd( c3, _mm_mul_pd( a01, b4 ));
		c4 = _mm_add_pd( c4, _mm_mul_pd( a23, b4 ));
		c5 = _mm_add_pd( c5, _mm_mul_pd( a01, b8 ));
		c6 = _mm_add_pd( c6, _mm_mul_pd( a23, b8 ));
		c7 = _mm_add_pd( c7, _mm_mul_pd( a01, b12));
		c8 = _mm_add_pd( c8, _mm_mul_pd( a23, b12));

	}
	_mm_storeu_pd(C+K+0+lda*0, c1 );
	_mm_storeu_pd(C+K+2+lda*0, c2 );
	_mm_storeu_pd(C+K+0+lda*1, c3 );
	_mm_storeu_pd(C+K+2+lda*1, c4 );
	_mm_storeu_pd(C+K+0+lda*2, c5 );
	_mm_storeu_pd(C+K+2+lda*2, c6 );
	_mm_storeu_pd(C+K+0+lda*3, c7 );
	_mm_storeu_pd(C+K+2+lda*3, c8 );
}
