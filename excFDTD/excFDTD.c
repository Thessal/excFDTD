//VS2015
//ipsxe2016 Cluster Edition
//Konstantinos, 2013


#define _DimX (10)
#define _DimY (20)
#define _DimZ (30)

#define _S_factor (2.0f)
//#define _dx (1.0f)
#define _dx (1e-7)


//#define _c0 (1.0f)
#define _c0 299792458.0f
#define _USE_MATH_DEFINES
#define _mu0_ ( 4e-7 * M_PI )
#define _eps0_ ( 1.0f / _c0 / _c0 / _mu0_ )
#define _dt_ (_dx / _c0 / _S_factor)
#define _cdt (_dx / _S_factor)
float stability_factor_inv = 1.0f / _S_factor; //not good

// doi:10.1088/0022-3727/40/22/043, Au
#define _eps_inf (1.0300 )
#define _gamma_D (1.1274e14)
#define _omega_D (1.3064e16 )
#define _Omega1_L (4.0812e15)
#define _Gamma1_L (7.3277e14)
#define _Delta_eps1_L (0.86822)
#define _phi1_L (-0.60756 )
#define _Omega2_L (6.4269e15)
#define _Gamma2_L (6.7371e14)
#define _Delta_eps2_L (1.3700)
#define _phi2_L (-0.08734)

#define _a01_div_eps0 (2.0f * _Delta_eps1_L * _Omega1_L * (_Omega1_L * cos(_phi1_L) - _Gamma1_L * sin(_phi1_L)))
#define _a02_div_eps0 (2.0f * _Delta_eps2_L * _Omega2_L * (_Omega2_L * cos(_phi2_L) - _Gamma2_L * sin(_phi2_L)))
#define _a11_div_eps0 (-2.0f * _Delta_eps1_L * _Omega1_L * sin(_phi1_L))
#define _a12_div_eps0 (-2.0f * _Delta_eps2_L * _Omega2_L * sin(_phi2_L))
#define _a01 ( _a01_div_eps0 * _eps0_ )
#define _a02 ( _a02_div_eps0 * _eps0_ )
#define _a11 ( _a11_div_eps0 * _eps0_ )
#define _a12 ( _a12_div_eps0 * _eps0_ )
#define _b01 (_Omega1_L * _Omega1_L + _Gamma1_L * _Gamma1_L)
#define _b02 (_Omega2_L * _Omega2_L + _Gamma2_L * _Gamma2_L)
#define _b11 (2.0f * _Gamma1_L)
#define _b12 (2.0f * _Gamma2_L)
#define _b21 (1.0f)
#define _b22 (1.0f)

#define _C1_ ( (_b21 / _dt_ / _dt_) + (_b11 * 0.5f / _dt_) + (_b01 * 0.25f) )
#define _C2_ ( (_b22 / _dt_ / _dt_) + (_b12 * 0.5f / _dt_) + (_b02 * 0.25f) )
#define _C11 ( ( ( 2.0f*_b21/_dt_/_dt_ ) - (_b01*0.5f) ) / _C1_ )
#define _C12 ( ( ( 2.0f*_b22/_dt_/_dt_ ) - (_b02*0.5f) ) / _C2_ )
#define _C21 ( ( ( _b11 * 0.5f / _dt_ ) - ( _b21 / _dt_ / _dt_ ) - ( _b01 * 0.25f ) ) / _C1_ )
#define _C22 ( ( ( _b12 * 0.5f / _dt_ ) - ( _b22 / _dt_ / _dt_ ) - ( _b02 * 0.25f ) ) / _C2_ )
#define _C31 ( ( ( _a01 * 0.25f ) + ( _a11 * 0.5f / _dt_ ) ) / _C1_ )
#define _C32 ( ( ( _a02 * 0.25f ) + ( _a12 * 0.5f / _dt_ ) ) / _C2_ )
#define _C41 ( _a01 * 0.5f / _C1_ )
#define _C42 ( _a02 * 0.5f / _C2_ )
#define _C51 ( ( _a01 * 0.25f ) - ( _a11 * 0.5f / _dt_ ) ) / _C1_
#define _C52 ( ( _a02 * 0.25f ) - ( _a12 * 0.5f / _dt_ ) ) / _C2_
#define _C4p ( _C41 + _C42 )
#define _C5p ( _C51 + _C52 )

//#define _dq_ ( ( 1.0f / _dt_ / _dt_ ) + ( _gamma_D  * 0.5f / _dt_ ) )
//#define _d1q ( 2.0f / _dq_ / _dt_ / _dt_ )
//#define _d2q ( ( _gamma_D * 0.5f / _dt_ ) - ( 1 / _dt_ / _dt_ ) ) / _dq_
//#define _d3q_div_eps0 ( _omega_D * _omega_D * 0.25f / _dq_ )
//#define _d4q_div_eps0 ( _omega_D * _omega_D * 0.5f / _dq_ )
//#define _d5q_div_eps0 _d3q_div_eps0
//single drude pole
//check eq24 --> slightly different from drude eq
#define _gamma__ (_gamma_D)
//FIXME : gamma = gamma_D ?  check
#define _d1 ( ( 2.0f - _gamma__ * _dt_ ) / ( 2 + _gamma__ * _dt_) )
#define _d2 (_eps0_ * _omega_D * _omega_D  * _dt_ / (2 + _gamma__ * _dt_) / _gamma__ )
#define _sigma_ (_eps0_ * _omega_D * _omega_D / _gamma__ )

#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "immintrin.h"
#include "time.h"

#define _blockDimX (8)
#define _blockDimY (8)
#define _blockDimZ (8)
#define _blockDimXYZ (_blockDimX * _blockDimY * _blockDimZ)

#define _sizeofFloat (32)

//3D loop tiling with 1px padding, for L1 cache or cuda SHMEM utilization
//1px padding required for easy L1 cache optimization
#define _gridDimX ((_DimX + (_blockDimX-2) - 1) / (_blockDimX-2))
#define _gridDimY ((_DimY + (_blockDimY-2) - 1) / (_blockDimY-2))
#define _gridDimZ ((_DimZ + (_blockDimZ-2) - 1) / (_blockDimZ-2))

#define _threadPerGrid (_blockDimX*_gridDimX*_blockDimY*_gridDimY*_blockDimZ*_gridDimZ)
#define _offsetX (1)
#define _offsetY (_blockDimX)
#define _offsetZ (_blockDimX*_blockDimY)
#define _offsetX_byte (4)
#define _offsetY_byte (_blockDimX*4)
#define _offsetZ_byte (_blockDimX*_blockDimY*4)
#define _offsetPadding (_offsetX+_offsetY+_offsetZ)
#define _offsetPadding_byte ((_offsetX+_offsetY+_offsetZ)*4)

#define _INDEX_BLOCK(blockIdxX, blockIdxY, blockIdxZ) \
		((blockIdxX) + _gridDimX * (blockIdxY) + _gridDimX * _gridDimY * (blockIdxZ))
#define _INDEX_THREAD(blockIdxX, blockIdxY, blockIdxZ, threadIdxX, threadIdxY, threadIdxZ) \
		( \
		_INDEX_BLOCK(blockIdxX, blockIdxY, blockIdxZ) * _blockDimXYZ\
		+ (threadIdxX) + _blockDimX * (threadIdxY) + _blockDimX * _blockDimY * (threadIdxZ) \
		)
#define _INDEX_XYZ(x,y,z) \
		( \
		_INDEX_THREAD( ((x))/(_blockDimX-2), ((y))/(_blockDimY-2), ((z))/(_blockDimZ-2), ((x))%(_blockDimX-2)+1 , ((y))%(_blockDimY-2)+1 , ((z))%(_blockDimZ-2)+1 ) \
		)

//#define SWAP(x, y) do { typeof(x) SWAPVAR = x; x = y; y = SWAPVAR; } while (0)
#include <string.h>
#define SWAP(x,y) do \
{ unsigned char swap_temp[sizeof(x) == sizeof(y) ? (signed)sizeof(x) : -1]; \
memcpy(swap_temp, &y, sizeof(x)); \
memcpy(&y, &x, sizeof(x)); \
memcpy(&x, swap_temp, sizeof(x)); \
} while (0)

__declspec(align(32)) float eps0_c_Ex[_threadPerGrid] = { 0 };
__declspec(align(32)) float eps0_c_Ey[_threadPerGrid] = { 0 };
__declspec(align(32)) float eps0_c_Ez[_threadPerGrid] = { 0 };
__declspec(align(32)) float eps0_c_Ex_old[_threadPerGrid] = { 0 };
__declspec(align(32)) float eps0_c_Ey_old[_threadPerGrid] = { 0 };
__declspec(align(32)) float eps0_c_Ez_old[_threadPerGrid] = { 0 };
__declspec(align(32)) float eps0_c_Pdx[_threadPerGrid] = { 0 };
__declspec(align(32)) float eps0_c_Pdy[_threadPerGrid] = { 0 };
__declspec(align(32)) float eps0_c_Pdz[_threadPerGrid] = { 0 };
__declspec(align(32)) float eps0_c_Pcp1x[_threadPerGrid] = { 0 };
__declspec(align(32)) float eps0_c_Pcp1y[_threadPerGrid] = { 0 };
__declspec(align(32)) float eps0_c_Pcp1z[_threadPerGrid] = { 0 };
__declspec(align(32)) float eps0_c_Pcp2x[_threadPerGrid] = { 0 };
__declspec(align(32)) float eps0_c_Pcp2y[_threadPerGrid] = { 0 };
__declspec(align(32)) float eps0_c_Pcp2z[_threadPerGrid] = { 0 };
__declspec(align(32)) float eps0_c_Pcp1x_old[_threadPerGrid] = { 0 };
__declspec(align(32)) float eps0_c_Pcp1y_old[_threadPerGrid] = { 0 };
__declspec(align(32)) float eps0_c_Pcp1z_old[_threadPerGrid] = { 0 };
__declspec(align(32)) float eps0_c_Pcp2x_old[_threadPerGrid] = { 0 };
__declspec(align(32)) float eps0_c_Pcp2y_old[_threadPerGrid] = { 0 };
__declspec(align(32)) float eps0_c_Pcp2z_old[_threadPerGrid] = { 0 };
__declspec(align(32)) float Hx[_threadPerGrid] = { 0 };
__declspec(align(32)) float Hy[_threadPerGrid] = { 0 };
__declspec(align(32)) float Hz[_threadPerGrid] = { 0 };
__declspec(align(32)) float eps_r_inv[_threadPerGrid] = { 0 };
__declspec(align(32)) float tempx[_threadPerGrid] = { 0 };
__declspec(align(32)) float tempy[_threadPerGrid] = { 0 };
__declspec(align(32)) float tempz[_threadPerGrid] = { 0 };

__m256* mEx = (__m256*)eps0_c_Ex;
__m256* mEy = (__m256*)eps0_c_Ey;
__m256* mEz = (__m256*)eps0_c_Ez;
__m256* mHx = (__m256*)Hx;
__m256* mHy = (__m256*)Hy;
__m256* mHz = (__m256*)Hz;
__m256* meps = (__m256*)eps_r_inv;

int init(void);
void Dielectric_HE(void);
void Dielectric_HE_C(void);
void DCP_HE_C(void);
void syncPadding(void);
int snapshot(void);
int main(int argc, char* argv[])
{
	printf("total area : block(%d,%d,%d) * grid(%d,%d,%d) = %d(%d,%d,%d)\n",_blockDimX,_blockDimY,_blockDimZ,_gridDimX,_gridDimY,_gridDimZ, _threadPerGrid, _blockDimX*_gridDimX, _blockDimY*_gridDimY, _blockDimZ*_gridDimZ);
	printf("thread space (padding overhead) : %d(%d,%d,%d) \n", (_blockDimX - 2)*_gridDimX* (_blockDimY - 2)*_gridDimY* (_blockDimZ - 2)*_gridDimZ,(_blockDimX-2)*_gridDimX , (_blockDimY - 2)*_gridDimY, (_blockDimZ - 2)*_gridDimZ);
	printf("xyz space (block overhead) : %d(%d,%d,%d) \n", _DimX*_DimY*_DimZ, _DimX, _DimY, _DimZ);
	printf("unit : sizeof(float)\n");
	printf("\n");

	printf("S       \t%01.2e\n", _S_factor);
	printf("c0      \t%01.2e\n", _c0);
	printf("dt      \t%01.2e\n", _dt_);
	printf("dx      \t%01.2e\n", _dx);
	printf("\n");

	printf("check:\n");
	printf("a01/eps0\t%01.2e\n", _a01_div_eps0);
	printf("a11/eps0\t%01.2e\n", _a11_div_eps0);
	printf("b01     \t%01.2e\n", _b01);
	printf("b11     \t%01.2e\n", _b11);
	printf("b21     \t%01.2e\n", _b21);

	init();
	
	time_t start;

	start = clock();
	for (int i = 0; i < 5; i++) { syncPadding(); DCP_HE_C(); }
	printf("time : %f\n", (double)(clock() - start) / CLK_TCK);

	//start = clock();
	//for (int i = 0; i < 1; i++) { syncPadding(); Dielectric_HE(); }
	//printf("time : %f\n", (double)(clock() - start) / CLK_TCK);


	syncPadding();
	
	snapshot();

	return 0;
}

void Dielectric_HE_C(void)
{
	for (unsigned __int64 offset = 0; offset < _threadPerGrid; offset += 1) {
		Hx[offset] -= (eps0_c_Ey[offset] - eps0_c_Ey[offset + _offsetZ] + eps0_c_Ez[offset + _offsetX] - eps0_c_Ez[offset + _offsetX]) * _cdt;
		Hy[offset] -= (eps0_c_Ez[offset] - eps0_c_Ez[offset + _offsetX] + eps0_c_Ex[offset + _offsetZ] - eps0_c_Ex[offset]) *_cdt;
		Hz[offset] -= (eps0_c_Ex[offset + _offsetZ] - eps0_c_Ex[offset + _offsetY + _offsetZ] + eps0_c_Ey[offset + _offsetZ] - eps0_c_Ey[offset - _offsetX + _offsetZ]) * _cdt;
		eps0_c_Ex[offset] += (Hy[offset - _offsetZ] - Hy[offset] + Hz[offset - _offsetZ] - Hz[offset - _offsetY - _offsetZ]) * eps_r_inv[offset] * _cdt;
		eps0_c_Ey[offset] += (Hz[offset - _offsetZ] - Hz[offset + _offsetX - _offsetZ] + Hx[offset] - Hx[offset - _offsetZ]) * eps_r_inv[offset] * _cdt;
		eps0_c_Ez[offset] += (Hx[offset - _offsetX - _offsetY] - Hx[offset - _offsetX] + Hy[offset] - Hy[offset - _offsetX]) * eps_r_inv[offset] * _cdt;
	}
}

void Dielectric_HE(void)
{
	for (unsigned __int64 offset = 0; offset < _threadPerGrid / 8; offset += 1) {
		_mm256_store_ps(Hx + offset * 8, // 1 * ymm(32byte) = 8 * float(4byte)
			_mm256_sub_ps(
				mHx[offset],
				_mm256_mul_ps(
					_mm256_sub_ps(
						_mm256_add_ps(
							_mm256_sub_ps(
								mEy[offset],
								mEy[offset + _offsetZ / 8]),
							_mm256_loadu_ps(eps0_c_Ez + offset + _offsetX + _offsetY)),
						_mm256_loadu_ps(eps0_c_Ez + offset + _offsetX)),
					_mm256_broadcast_ss(&stability_factor_inv))
			)
		);
		_mm256_store_ps(Hy + offset * 8, // 1 * ymm(32byte) = 8 * float(4byte)
			_mm256_sub_ps(
				mHy[offset],
				_mm256_mul_ps(
					_mm256_sub_ps(
						_mm256_add_ps(
							_mm256_sub_ps(
								mEz[offset],
								_mm256_loadu_ps(eps0_c_Ez + offset + _offsetX)),
							mEx[offset + _offsetZ / 8]),
						mEx[offset]),
					_mm256_broadcast_ss(&stability_factor_inv))
			)
		);
		_mm256_store_ps(Hz + offset * 8, // 1 * ymm(32byte) = 8 * float(4byte)
			_mm256_sub_ps(
				mHz[offset],
				_mm256_mul_ps(
					_mm256_sub_ps(
						_mm256_add_ps(
							_mm256_sub_ps(
								mEx[offset + _offsetZ / 8],
								mEx[offset + _offsetY / 8 + _offsetZ / 8]),
							mEy[offset + _offsetZ / 8]),
						_mm256_loadu_ps(eps0_c_Ey + offset - _offsetX + _offsetZ)),
					_mm256_broadcast_ss(&stability_factor_inv))
			)
		);

		_mm256_store_ps(eps0_c_Ex + offset * 8, // 1 * ymm(32byte) = 8 * float(4byte)
			_mm256_add_ps(
				_mm256_mul_ps(
					_mm256_mul_ps(
						_mm256_sub_ps(
							_mm256_add_ps(
								_mm256_sub_ps(
									mHy[offset - _offsetZ/8], //strange
									mHy[offset]),
								mHz[offset - _offsetZ / 8]),
							mHz[offset - _offsetY / 8 - _offsetZ / 8]),
						meps[offset]),
					_mm256_broadcast_ss(&stability_factor_inv)), //not good
				mEx[offset])
		);
		_mm256_store_ps(eps0_c_Ey + offset * 8, // 1 * ymm(32byte) = 8 * float(4byte)
			_mm256_add_ps(
				_mm256_mul_ps(
					_mm256_mul_ps(
						_mm256_sub_ps(
							_mm256_add_ps(
								_mm256_sub_ps(
									mHz[offset - (_offsetZ / 8)],
									_mm256_loadu_ps(Hz + offset * 8 + _offsetX - _offsetZ )),//SSE2 : _mm256_permutevar8x32_ps(mHy+offset*8,-1)									 
								mHx[offset]),
							mHx[offset - _offsetZ / 8]),
						meps[offset]),
					_mm256_broadcast_ss(&stability_factor_inv)),
				mEy[offset])
		);
		_mm256_store_ps(eps0_c_Ez + offset * 8, // 1 * ymm(32byte) = 8 * float(4byte)
			_mm256_add_ps(
				_mm256_mul_ps(
					_mm256_mul_ps(
						_mm256_sub_ps(
							_mm256_add_ps(
								_mm256_sub_ps(
									_mm256_loadu_ps(Hx + offset * 8 - _offsetX - _offsetY),
									_mm256_loadu_ps(Hx + offset * 8 - _offsetX)),
								mHy[offset]),
							_mm256_loadu_ps(Hy + offset * 8 - _offsetX)),
						meps[offset]),
					_mm256_broadcast_ss(&stability_factor_inv)),
				mEz[offset])
		);
	}
}

void DCP_HE_C(void)
{
	// before --> after
	// temp --> E
	// E --> E_old
	// pcp_old <--> pcp

	for (unsigned __int64 offset = 0; offset < _threadPerGrid; offset += 1) {
		// H
		Hx[offset] -= (eps0_c_Ey[offset] - eps0_c_Ey[offset + _offsetZ] + eps0_c_Ez[offset + _offsetX] - eps0_c_Ez[offset + _offsetX]) * _cdt;
		Hy[offset] -= (eps0_c_Ez[offset] - eps0_c_Ez[offset + _offsetX] + eps0_c_Ex[offset + _offsetZ] - eps0_c_Ex[offset]) * _cdt;
		Hz[offset] -= (eps0_c_Ex[offset + _offsetZ] - eps0_c_Ex[offset + _offsetY + _offsetZ] + eps0_c_Ey[offset + _offsetZ] - eps0_c_Ey[offset - _offsetX + _offsetZ]) * _cdt;

		// E
		tempx[offset] = (Hy[offset - _offsetZ] - Hy[offset] + Hz[offset - _offsetZ] - Hz[offset - _offsetY - _offsetZ]) * eps_r_inv[offset] * _cdt;//eq30term1
		tempy[offset] = (Hz[offset - _offsetZ] - Hz[offset + _offsetX - _offsetZ] + Hx[offset] - Hx[offset - _offsetZ]) * eps_r_inv[offset] * _cdt;
		tempz[offset] = (Hx[offset - _offsetX - _offsetY] - Hx[offset - _offsetX] + Hy[offset] - Hy[offset - _offsetX]) * eps_r_inv[offset] * _cdt;
		tempx[offset] += eps0_c_Ex[offset] * (_eps0_ * _eps_inf - 0.5f * _sigma_ * _dt_ + _d2 - _C4p); //eq30term2
		tempy[offset] += eps0_c_Ey[offset] * (_eps0_ * _eps_inf - 0.5f * _sigma_ * _dt_ + _d2 - _C4p);
		tempz[offset] += eps0_c_Ez[offset] * (_eps0_ * _eps_inf - 0.5f * _sigma_ * _dt_ + _d2 - _C4p);
		tempx[offset] -= eps0_c_Ex_old[offset] * _C5p; //eq30term3
		tempy[offset] -= eps0_c_Ey_old[offset] * _C5p;
		tempz[offset] -= eps0_c_Ez_old[offset] * _C5p;
		tempx[offset] -= eps0_c_Pdx[offset] * (_d1 - 1); //eq30term4
		tempy[offset] -= eps0_c_Pdy[offset] * (_d1 - 1);
		tempz[offset] -= eps0_c_Pdz[offset] * (_d1 - 1);
		tempz[offset] -= eps0_c_Pdz[offset] * (_d1 - 1);
		tempx[offset] -= eps0_c_Pcp1x[offset] * (_C11 - 1); //eq30term5
		tempy[offset] -= eps0_c_Pcp1y[offset] * (_C11 - 1);
		tempz[offset] -= eps0_c_Pcp1z[offset] * (_C11 - 1);
		tempx[offset] -= eps0_c_Pcp2x[offset] * (_C12 - 1);
		tempy[offset] -= eps0_c_Pcp2y[offset] * (_C12 - 1);
		tempz[offset] -= eps0_c_Pcp2z[offset] * (_C12 - 1);
		tempx[offset] -= eps0_c_Pcp2x[offset] * (_C12 - 1);
		tempy[offset] -= eps0_c_Pcp2y[offset] * (_C12 - 1);
		tempz[offset] -= eps0_c_Pcp2z[offset] * (_C12 - 1);
		tempx[offset] -= eps0_c_Pcp1x_old[offset] * _C21; //eq30term6
		tempy[offset] -= eps0_c_Pcp1y_old[offset] * _C21;
		tempz[offset] -= eps0_c_Pcp1z_old[offset] * _C21;
		tempx[offset] -= eps0_c_Pcp2x_old[offset] * _C22;
		tempy[offset] -= eps0_c_Pcp2y_old[offset] * _C22;
		tempz[offset] -= eps0_c_Pcp2z_old[offset] * _C22;

		//PCP
		eps0_c_Pcp1x_old[offset] *= _C21; //eq14term2
		eps0_c_Pcp1y_old[offset] *= _C21;
		eps0_c_Pcp1z_old[offset] *= _C21;
		eps0_c_Pcp2x_old[offset] *= _C22;
		eps0_c_Pcp2y_old[offset] *= _C22;
		eps0_c_Pcp2z_old[offset] *= _C22;
		eps0_c_Pcp1x_old[offset] += eps0_c_Pcp1x[offset] * _C11; //eq14term1
		eps0_c_Pcp1y_old[offset] += eps0_c_Pcp1y[offset] * _C11;
		eps0_c_Pcp1z_old[offset] += eps0_c_Pcp1z[offset] * _C11;
		eps0_c_Pcp2x_old[offset] += eps0_c_Pcp2x[offset] * _C12;
		eps0_c_Pcp2y_old[offset] += eps0_c_Pcp2y[offset] * _C12;
		eps0_c_Pcp2z_old[offset] += eps0_c_Pcp2z[offset] * _C12;
		eps0_c_Pcp1x_old[offset] += tempx[offset] * _C31; //eq14term3
		eps0_c_Pcp1y_old[offset] += tempy[offset] * _C31;
		eps0_c_Pcp1z_old[offset] += tempz[offset] * _C31;
		eps0_c_Pcp2x_old[offset] += tempx[offset] * _C32;
		eps0_c_Pcp2y_old[offset] += tempy[offset] * _C32;
		eps0_c_Pcp2z_old[offset] += tempz[offset] * _C32;
		eps0_c_Pcp1x_old[offset] += eps0_c_Ex[offset] * _C41; //eq14term4
		eps0_c_Pcp1y_old[offset] += eps0_c_Ey[offset] * _C41;
		eps0_c_Pcp1z_old[offset] += eps0_c_Ez[offset] * _C41;
		eps0_c_Pcp2x_old[offset] += eps0_c_Ex[offset] * _C42;
		eps0_c_Pcp2y_old[offset] += eps0_c_Ey[offset] * _C42;
		eps0_c_Pcp2z_old[offset] += eps0_c_Ez[offset] * _C42;
		eps0_c_Pcp1x_old[offset] += eps0_c_Ex_old[offset] * _C51; //eq14term5
		eps0_c_Pcp1y_old[offset] += eps0_c_Ey_old[offset] * _C51;
		eps0_c_Pcp1z_old[offset] += eps0_c_Ez_old[offset] * _C51;
		eps0_c_Pcp2x_old[offset] += eps0_c_Ex_old[offset] * _C52;
		eps0_c_Pcp2y_old[offset] += eps0_c_Ey_old[offset] * _C52;
		eps0_c_Pcp2z_old[offset] += eps0_c_Ez_old[offset] * _C52;
		//eps0_c_Ex_old can be used as temp var here now

		//PD
		eps0_c_Pdx[offset] *= _d1; //eq27term1
		eps0_c_Pdy[offset] *= _d1; 
		eps0_c_Pdz[offset] *= _d1; 
		eps0_c_Pdx[offset] -= tempx[offset] * _d2; //eq27term2
		eps0_c_Pdy[offset] -= tempy[offset] * _d2;
		eps0_c_Pdz[offset] -= tempz[offset] * _d2;
		eps0_c_Pdx[offset] -= eps0_c_Ex[offset] * _d2; //eq27term3
		eps0_c_Pdy[offset] -= eps0_c_Ey[offset] * _d2;
		eps0_c_Pdz[offset] -= eps0_c_Ez[offset] * _d2;

	}

	SWAP(tempx, eps0_c_Ex);
	SWAP(tempy, eps0_c_Ey);
	SWAP(tempz, eps0_c_Ez);

	SWAP(tempx, eps0_c_Ex_old);
	SWAP(tempy, eps0_c_Ey_old);
	SWAP(tempz, eps0_c_Ez_old);

	SWAP(eps0_c_Pcp1x_old, eps0_c_Pcp1x);
	SWAP(eps0_c_Pcp1y_old, eps0_c_Pcp1y);
	SWAP(eps0_c_Pcp1z_old, eps0_c_Pcp1z);
	SWAP(eps0_c_Pcp2x_old, eps0_c_Pcp2x);
	SWAP(eps0_c_Pcp2y_old, eps0_c_Pcp2y);
	SWAP(eps0_c_Pcp2z_old, eps0_c_Pcp2z);

}

#define _syncX_(FIELD) \
FIELD[_INDEX_THREAD(X - 1, Y, Z, _blockDimX - 1, yy, zz)] = FIELD[_INDEX_THREAD(X, Y, Z, 1, yy, zz)]; \
FIELD[_INDEX_THREAD(X, Y, Z, 0, yy, zz)] = FIELD[_INDEX_THREAD(X - 1, Y, Z, _blockDimX - 2, yy, zz)]; 
#define _syncY_(FIELD) \
FIELD[_INDEX_THREAD(X, Y - 1, Z, xx, _blockDimY - 1, zz)] = FIELD[_INDEX_THREAD(X, Y, Z, xx, 1, zz)]; \
FIELD[_INDEX_THREAD(X, Y, Z, xx, 0, zz)] = FIELD[_INDEX_THREAD(X, Y - 1, Z, xx, _blockDimY - 2, zz)]; 
#define _syncZ_(FIELD) \
FIELD[_INDEX_THREAD(X, Y, Z - 1, xx, yy, _blockDimZ - 1)] = FIELD[_INDEX_THREAD(X, Y, Z, xx, yy, 1)]; \
FIELD[_INDEX_THREAD(X, Y, Z, xx, yy, 0)] = FIELD[_INDEX_THREAD(X, Y, Z - 1, xx, yy, _blockDimZ - 2)];
#define _syncXall _syncX_(eps0_c_Ex) _syncX_(eps0_c_Ey) _syncX_(eps0_c_Ez)  _syncX_(Hx)  _syncX_(Hy) _syncX_(Hz) 
#define _syncYall _syncY_(eps0_c_Ex) _syncY_(eps0_c_Ey) _syncY_(eps0_c_Ez)  _syncY_(Hx)  _syncY_(Hy) _syncY_(Hz) 
#define _syncZall _syncZ_(eps0_c_Ex) _syncZ_(eps0_c_Ey) _syncZ_(eps0_c_Ez)  _syncZ_(Hx)  _syncZ_(Hy) _syncZ_(Hz) 

//이거 안넣으면 boundary에서 index 1 shift 있음
//#define _syncBoundaryX(FIELD) \
//FIELD[_INDEX_THREAD(_gridDimX-1, Y, Z, _blockDimX - 1, yy, zz)] = FIELD[_INDEX_THREAD(0, Y, Z, 1, yy, zz)]; \
//FIELD[_INDEX_THREAD(0, Y, Z, 0, yy, zz)] = FIELD[_INDEX_THREAD(_gridDimX-1, Y, Z, _blockDimX - 2, yy, zz)]; \
//#define _syncBoundaryall _syncBoundary(eps0_c_Ex) _syncBoundary(eps0_c_Ey) _syncBoundary(eps0_c_Ez)  _syncBoundary(Hx)  _syncBoundary(Hy) _syncBoundary(Hz) 

void syncPadding(void) {
	for (int X = 1; X < _gridDimX; X++)
		for (int Y = 1; Y < _gridDimY; Y++)
			for (int Z = 1; Z < _gridDimZ; Z++) {
				for (int yy = 0; yy<_blockDimY; yy++)
					for (int zz = 0; zz < _blockDimZ; zz++) { _syncXall }
				for (int zz = 0; zz<_blockDimY; zz++)
					for (int xx = 0; xx < _blockDimZ; xx++) { _syncYall }
				for (int xx = 0; xx<_blockDimY; xx++)
					for (int yy = 0; yy < _blockDimZ; yy++) { _syncZall }
			}
}


int init(void)
{
	for (unsigned __int64 i = 0; i < _threadPerGrid; i++)
	{
		//eps0_c_Ey[i] = 0.0f;

		unsigned __int64 tmp = i;
		int X = 0, Y = 0, Z = 0;
		X += tmp % _blockDimX - 1; tmp /= _blockDimX;
		Y += tmp % _blockDimY - 1; tmp /= _blockDimY;
		Z += tmp % _blockDimZ - 1; tmp /= _blockDimZ;
		X += (tmp % _gridDimX) * (_blockDimX-2); tmp /= _gridDimX;
		Y += (tmp % _gridDimY) * (_blockDimY-2); tmp /= _gridDimY;
		Z += (tmp % _gridDimZ) * (_blockDimZ-2); tmp /= _gridDimZ;

		eps0_c_Ey[i] = (float)(0.0f);
		eps0_c_Ez[i] = (float)(0.0f);
		Hx[i] = (float)(0.0f);
		Hy[i] = (float)(0.0f);
		Hz[i] = (float)(0.0f);
		//Hy[i] = (float)(X * 10000 + Y * 100 + Z);
		//Hz[i] = (float)(X * 10000 + Y * 100 + Z);
		eps_r_inv[i] = 1.0f;


		if (X < 0 || _DimX-1 < X || Y < 0 || _DimY-1 < Y || Z < 0 || _DimZ-1 < Z) { continue; }
		else if (X==5 && Y ==2 && Z ==5) { eps0_c_Ex[i] = 1.0f; }
		//else if (0 <= X) { eps0_c_Ex[i] = 8.2f; }
		//else if (0 <= X) { 	eps0_c_Ex[i] = (float)(X*10000 + Y*100 + Z);}
		else { eps0_c_Ex[i] = 0.0f; }


	}
	return 0;
}
#define TEMP eps0_c_Ex
//#define TEMP Hy
int snapshot(void)
{
	printf("Z=5\n");
	int Z = 5;
	for (int Y = 0; Y < _DimY; Y++) {
		//if (Y%_blockDimY == 0) { for (int X = 0; X < _DimX; X++) { printf("%07.3f ", eps0_c_Ex[_INDEX_XYZ(X, -1, Z)]);  } 	printf("\n");}
		for (int X = 0; X < _DimX; X++) {
			//if (X%_blockDimX == 0) { printf("%07.1f ", eps0_c_Ex[_INDEX_XYZ(-1, Y, Z)]);}
			printf("%07.3f ", TEMP[_INDEX_XYZ(X, Y, Z) ]);
			//printf("%06.0f \t", eps0_c_Ex[_INDEX_XYZ(X, Y, Z)]);
			//printf("%d \t", _INDEX_BLOCK(X/(_blockDimX-2), Y/(_blockDimY-2), Z/(_blockDimZ-2)));
			//printf("%d \t",Y/_blockDimY);
			//printf("%d \t",_INDEX_XYZ(X,Y,Z));
			//if (X%_blockDimX == _blockDimX-1) { printf("%07.3f ", eps0_c_Ex[_INDEX_XYZ(_blockDimX, Y, Z)]); }
		}
		printf("\n");
		//if (Y%_blockDimY == _blockDimY -1 ){			for (int X = 0; X < _DimX; X++) 	{				printf("%07.3f ", eps0_c_Ex[_INDEX_XYZ(X, _blockDimY, Z)]);			}printf("\n");		}
	}
	return (0) ;
	printf("\n");
	printf("Y=2\n");
	for (int Z = 0; Z < _DimZ; Z++) {
		for (int X = 0; X < _DimX; X++) {
			int Y = 2;
			printf("%02.1f \t", eps0_c_Ex[_INDEX_XYZ(X, Y, Z)]);
			//printf("%06.0f \t", eps0_c_Ex[_INDEX_XYZ(X, Y, Z)]);
		}
		printf("\n");
	}
	return 0;
}