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

const char* dgemm_desc = "Simple blocked dgemm.";

#include <emmintrin.h>

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 4
#endif

#define min(a,b) (((a)<(b))?(a):(b))

static void simd_2x2(int lda, int M, int N, int K,double* A, double* B, double* C);

static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
	for (int i = 0; i < M; ++i)
		for (int j = 0; j < N; ++j) 
		{
			double cij = C[i+j*lda];

			for (int k = 0; k < K; ++k)
			{
				cij += A[i+k*lda] * B[k+j*lda];
			}
			C[i+j*lda] = cij;
		}
}

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



void square_dgemm (int lda, double* A, double* B, double* C)
{
		for (int j = 0; j < lda; j += BLOCK_SIZE)
			for (int k = 0; k < lda; k += BLOCK_SIZE)
	for (int i = 0; i < lda; i += BLOCK_SIZE)
			{
				int M = min (BLOCK_SIZE, lda-i);
				int N = min (BLOCK_SIZE, lda-j);
				int K = min (BLOCK_SIZE, lda-k);
				if (M == 4 && N==4 && K==4)
					simd_2x2(lda, i + k*lda, k+j*lda , i+j*lda, A , B , C);
				else
					do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
			}
	
}

