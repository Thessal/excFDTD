//ADE DCP FDTD AVX ver
//Jongkook Choi
//VS2015, ipsxe2016 Cluster Edition, Parallel MKL, IPP (multithread)
//reference : K.P.Prokopidis, 2013
//reference : uFDTD  (eecs.wsu.edu/~schneidj/ufdtd/)
//reference : Torok et al, 2006 (doi: 10.1364/JOSAA.23.000713) formulation used for NTFF


#define _DimX (100)
#define _DimY (100)
#define _DimZ (100)

#define _STEP (1000)

//eq35
//consider using simple PML for NTFF calculation
#define _PML_PX_X_ (16)
#define _PML_PX_Y_ (16)
#define _PML_PX_Z_ (16)
#define _PML_ALPHA_TUNING_ 0.1f
int pml_n = 3; //consider using macro
float pml_R = 10e-4;
float pml_kappa_max = 8.0f;
#define _NTFF_Margin_ (10)

#define _S_factor (2.0f)
#define _dx (5e-9)

#define _c0 299792458.0f
#define _USE_MATH_DEFINES
#define _mu0_ ( 4e-7 * M_PI )
#define _eps0_ ( 1.0f / _c0 / _c0 / _mu0_ )
#define _dt_ (_dx / _c0 / _S_factor)
#define _cdt__ (_dx / _S_factor)
#define _cdt_div_dx (1 / _S_factor)
float stability_factor_inv = 1.0f / _S_factor;

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

// RFT NTFF frequency
#define RFT_WINDOW _STEP
//#if RFT_WINDOW > _STEP
//	#define RFT_WINDOW _STEP
//#endif
#define FREQ_N 1
float FREQ_LIST_DESIRED[FREQ_N] = {_c0 / 500e-9};
//#define FREQ_N 3
//float FREQ_LIST_DESIRED[FREQ_N] = { _c0 / 300e-9, _c0 / 500e-9, _c0 / 800e-9 };
float RFT_K_LIST_CALCULATED[FREQ_N];


// == loop tiling ==

#define _blockDimX (20)
#define _blockDimY (20)
#define _blockDimZ (20)
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

// === surface memory for runningDFT and NTFF
#define _SURF_Margin_ _NTFF_Margin_
#define _SURF_StartX_ ((_PML_PX_X_)+_SURF_Margin_)
#define _SURF_StartY_ ((_PML_PX_Y_)+_SURF_Margin_)
#define _SURF_StartZ_ ((_PML_PX_Z_)+_SURF_Margin_)
#define _SURF_EndX_ ((_DimX-1)-(_PML_PX_X_)-_SURF_Margin_) 
#define _SURF_EndY_ ((_DimY-1)-(_PML_PX_Y_)-_SURF_Margin_)
#define _SURF_EndZ_ ((_DimZ-1)-(_PML_PX_Z_)-_SURF_Margin_)
#define _SURF_DX_ ((_SURF_EndX_)-(_SURF_StartX_)+1) 
#define _SURF_DY_ ((_SURF_EndY_)-(_SURF_StartY_)+1)
#define _SURF_DZ_ ((_SURF_EndZ_)-(_SURF_StartZ_)+1)
#define _SURF_SIZE_ (2*((_SURF_DX_)*(_SURF_DY_)+(_SURF_DZ_-2)*(_SURF_DX_)+(_SURF_DY_-2)*(_SURF_DZ_-2)))

//using this function inside for loop is not efficient
#define _SURF_INDEX_XYZ(x,y,z) \
( \
(x < _SURF_StartX_ || _SURF_EndX_ < x || y < _SURF_StartY_ || _SURF_EndY_ < y || z < _SURF_StartZ_ || _SURF_EndZ_ < z) ? -1 : ( \
z == _SURF_StartZ_ ? _SURF_DX_*(y-_SURF_StartY_)+(x-_SURF_StartX_) : ( \
z == _SURF_EndZ_   ? _SURF_DX_*(y-_SURF_StartY_)+(x-_SURF_StartX_) + (_SURF_DX_*_SURF_DY_) : ( \
y == _SURF_StartY_ ? (_SURF_DZ_-2)*(x-_SURF_StartX_)+(z-_SURF_StartZ_-1) + 2*(_SURF_DX_*_SURF_DY_) : ( \
y == _SURF_EndY_   ? (_SURF_DZ_-2)*(x-_SURF_StartX_)+(z-_SURF_StartZ_-1) + 2*(_SURF_DX_*_SURF_DY_) + (_SURF_DZ_-2)*(_SURF_DX_) : ( \
x == _SURF_StartX_ ? (_SURF_DY_-2)*(z-_SURF_StartZ_-1)+(y-_SURF_StartY_-1) + 2*(_SURF_DX_*_SURF_DY_+(_SURF_DZ_-2)*(_SURF_DX_)) : ( \
x == _SURF_EndX_   ? (_SURF_DY_-2)*(z-_SURF_StartZ_-1)+(y-_SURF_StartY_-1) + 2*(_SURF_DX_*_SURF_DY_+(_SURF_DZ_-2)*(_SURF_DX_)) + (_SURF_DY_-2)*(_SURF_DZ_-2) : ( \
-1))))))) \
)  

//#define _SET_SURF_XYZ_INDEX(surf_index) \
//do{ \
//unsigned __int64 temp_idx = surf_index; \
//if ( temp_idx                         <  (_SURF_DX_*_SURF_DY_) ) { \
//	surf_z = _SURF_StartZ_; surf_x = _SURF_StartX_ + temp_idx % (_SURF_DX_ - 2); surf_y = _SURF_StartY_ + temp_idx / (_SURF_DX_ - 2); break; } \
//if ((temp_idx-=(_SURF_DX_*_SURF_DY_)) <  (_SURF_DX_*_SURF_DY_) ) { \
//	surf_z = _SURF_EndZ_; surf_x = _SURF_StartX_ + temp_idx % (_SURF_DX_ - 2); surf_y = _SURF_StartY_ + temp_idx / (_SURF_DX_ - 2); break; } \
//if ((temp_idx-=(_SURF_DX_*_SURF_DY_)) <  (_SURF_DZ_-2)*(_SURF_DX_) ) { \
//	surf_y = _SURF_StartY_; surf_z = 1 + _SURF_StartZ_ + temp_idx % (_SURF_DZ_ - 2); surf_x = _SURF_StartX_ + temp_idx / (_SURF_DZ_ - 2); break; }\
//if ((temp_idx-=(_SURF_DZ_-2)*(_SURF_DX_)) <  (_SURF_DZ_-2)*(_SURF_DX_) ) { \
//	surf_y = _SURF_EndY_; surf_z = 1 + _SURF_StartZ_ + temp_idx % (_SURF_DZ_ - 2); surf_x = _SURF_StartX_ + temp_idx / (_SURF_DZ_ - 2); break; }\
//if ((temp_idx-=(_SURF_DZ_-2)*(_SURF_DX_)) <  (_SURF_DY_-2)*(_SURF_DZ_-2) ) { \
//	surf_x = _SURF_StartX_; surf_y = 1 + _SURF_StartY_ + temp_idx % (_SURF_DY_ - 2); surf_z = 1 + _SURF_StartZ_ + temp_idx / (_SURF_DY_ - 2); break; }\
//if ((temp_idx-=(_SURF_DY_-2)*(_SURF_DZ_-2)) <  (_SURF_DY_-2)*(_SURF_DZ_-2) ) { \
//	surf_x = _SURF_EndX_; surf_y = 1 + _SURF_StartY_ + temp_idx % (_SURF_DY_ - 2); surf_z = 1 + _SURF_StartZ_ + temp_idx / (_SURF_DY_ - 2); break; }\
//surf_x = -1; surf_y = -1; surf_z = -1; \
//} while(0)
#define _SET_SURF_XYZ_INDEX(surf_index) \
do{ \
unsigned __int64 temp_idx = surf_index; \
if ( temp_idx                         <  (_SURF_DX_*_SURF_DY_) ) { \
	surf_z = _SURF_StartZ_; surf_x = _SURF_StartX_ + temp_idx % (_SURF_DX_); surf_y = _SURF_StartY_ + temp_idx / (_SURF_DX_); break; } \
if ((temp_idx-=(_SURF_DX_*_SURF_DY_)) <  (_SURF_DX_*_SURF_DY_) ) { \
	surf_z = _SURF_EndZ_; surf_x = _SURF_StartX_ + temp_idx % (_SURF_DX_); surf_y = _SURF_StartY_ + temp_idx / (_SURF_DX_); break; } \
if ((temp_idx-=(_SURF_DX_*_SURF_DY_)) <  (_SURF_DZ_-2)*(_SURF_DX_) ) { \
	surf_y = _SURF_StartY_; surf_z = 1 + _SURF_StartZ_ + temp_idx % (_SURF_DZ_ - 2); surf_x = _SURF_StartX_ + temp_idx / (_SURF_DZ_ - 2); break; }\
if ((temp_idx-=(_SURF_DZ_-2)*(_SURF_DX_)) <  (_SURF_DZ_-2)*(_SURF_DX_) ) { \
	surf_y = _SURF_EndY_; surf_z = 1 + _SURF_StartZ_ + temp_idx % (_SURF_DZ_ - 2); surf_x = _SURF_StartX_ + temp_idx / (_SURF_DZ_ - 2); break; }\
if ((temp_idx-=(_SURF_DZ_-2)*(_SURF_DX_)) <  (_SURF_DY_-2)*(_SURF_DZ_-2) ) { \
	surf_x = _SURF_StartX_; surf_y = 1 + _SURF_StartY_ + temp_idx % (_SURF_DY_ - 2); surf_z = 1 + _SURF_StartZ_ + temp_idx / (_SURF_DY_ - 2); break; }\
if ((temp_idx-=(_SURF_DY_-2)*(_SURF_DZ_-2)) <  (_SURF_DY_-2)*(_SURF_DZ_-2) ) { \
	surf_x = _SURF_EndX_; surf_y = 1 + _SURF_StartY_ + temp_idx % (_SURF_DY_ - 2); surf_z = 1 + _SURF_StartZ_ + temp_idx / (_SURF_DY_ - 2); break; }\
surf_x = -1; surf_y = -1; surf_z = -1; \
} while(0)

static double FT_eps0cE[_SURF_SIZE_][FREQ_N][3][2]; //XYZ,Re,Im
static double FT_H[_SURF_SIZE_][FREQ_N][3][2]; //XYZ,Re,Im


// === DCP coefficients

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
#define _C3p ( _C31 + _C32 )
#define _C4p ( _C41 + _C42 )
#define _C5p ( _C51 + _C52 )

#define _gamma__ (_gamma_D)
//FIXME : gamma = gamma_D ?  check
#define _d1 ( ( 2.0f - _gamma__ * _dt_ ) / ( 2 + _gamma__ * _dt_) )
#define _d2 (_eps0_ * _omega_D * _omega_D  * _dt_ / (2 + _gamma__ * _dt_) / _gamma__ )
#define _sigma_ (_eps0_ * _omega_D * _omega_D / _gamma__ )




// == Maxwell memory ==
__declspec(align(32)) static float eps0_c_Ex[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float eps0_c_Ey[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float eps0_c_Ez[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float eps0_c_Ex_old[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float eps0_c_Ey_old[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float eps0_c_Ez_old[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float eps0_c_Pdx[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float eps0_c_Pdy[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float eps0_c_Pdz[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float eps0_c_Pcp1x[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float eps0_c_Pcp1y[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float eps0_c_Pcp1z[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float eps0_c_Pcp2x[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float eps0_c_Pcp2y[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float eps0_c_Pcp2z[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float eps0_c_Pcp1x_old[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float eps0_c_Pcp1y_old[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float eps0_c_Pcp1z_old[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float eps0_c_Pcp2x_old[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float eps0_c_Pcp2y_old[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float eps0_c_Pcp2z_old[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float Hx[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float Hy[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float Hz[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float eps_r_inv[_threadPerGrid] = { 1.0f }; //1.0f not filled
__declspec(align(32)) static float tempx[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float tempy[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float tempz[_threadPerGrid] = { 0.0f };

// == PML memory ==
__declspec(align(32)) static float psiXY_dx[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float psiYZ_dx[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float psiZX_dx[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float psiXZ_dx[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float psiYX_dx[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float psiZY_dx[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float b_X[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float b_Y[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float b_Z[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float C_X[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float C_Y[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float C_Z[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float sigmaX_dt_div_eps0[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float sigmaY_dt_div_eps0[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float sigmaZ_dt_div_eps0[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float kappaX[_threadPerGrid] = { 1.0f }; //1.0f not filled
__declspec(align(32)) static float kappaY[_threadPerGrid] = { 1.0f };
__declspec(align(32)) static float kappaZ[_threadPerGrid] = { 1.0f };
__declspec(align(32)) static float alpha_dt_div_eps0[_threadPerGrid] = { 0.0f };

__declspec(align(32)) static unsigned __int32 mask[_threadPerGrid] = { 0 };





#include "immintrin.h"
// == Assembly setup ==
__m256* mEx = (__m256*)eps0_c_Ex;
__m256* mEy = (__m256*)eps0_c_Ey;
__m256* mEz = (__m256*)eps0_c_Ez;
__m256* mHx = (__m256*)Hx;
__m256* mHy = (__m256*)Hy;
__m256* mHz = (__m256*)Hz;
__m256* meps = (__m256*)eps_r_inv;




#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "time.h"
#include "lodepng.h"
int init(void);
void Dielectric_HE(void);
void Dielectric_HE_C(void);
void DCP_HE_C(void);
void syncPadding(void);
void RFT(void);
void NTFF(void);
int snapshot(void);

#include "mkl.h"
int main(int argc, char* argv[])
{
	init();
	time_t start;
	start = clock();
	printf("\nCalculating field \n");
	for (int i = 0; i <= _STEP; i++) {
		printf("%f%%\r", 100.0f*(float)i / _STEP);
		float addval = 1.0*sin(2.0f * M_PI * _c0 / 500e-9 * (float)i * _dt_) * exp(-((float)i - 500.0f)*((float)i - 500.0f) / 250.0f / 250.0f);
		//float addval = sin(2.0f * M_PI * _c0 / 500e-9 * (float)i * _dt_) ;
		eps0_c_Ex[_INDEX_XYZ(50, 50, 50)] += addval;
		//Hy[_INDEX_XYZ(50, 50, 50)] += addval;
		DCP_HE_C();
		RFT(); //FIXME : print warning message if RFT would not reach its final step
	}
	printf("\ntime : %f\n", (double)(clock() - start) / CLK_TCK);
	snapshot();
	NTFF();
	return 0;
}

int dielectric_flag = 0;
void Dielectric_HE_C(void)
{
	dielectric_flag = 1;
	DCP_HE_C();
}

void DCP_HE_C(void)
{
	syncPadding();
	for (unsigned __int64 offset = 0; offset < _threadPerGrid; offset += 1) {
		if (((mask[offset] & (1 << 0)) >> 0) == 1) { continue; } // skip padding

		Hx[offset] -= (eps0_c_Ey[offset] - eps0_c_Ey[offset + _offsetZ] + eps0_c_Ez[offset + _offsetX + _offsetY] - eps0_c_Ez[offset + _offsetX]) * _cdt_div_dx;
		Hy[offset] -= (eps0_c_Ez[offset] - eps0_c_Ez[offset + _offsetX] + eps0_c_Ex[offset + _offsetZ] - eps0_c_Ex[offset]) *_cdt_div_dx;
		Hz[offset] -= (eps0_c_Ex[offset + _offsetZ] - eps0_c_Ex[offset + _offsetY + _offsetZ] + eps0_c_Ey[offset + _offsetZ] - eps0_c_Ey[offset - _offsetX + _offsetZ]) * _cdt_div_dx;
	}
	syncPadding();

	for (unsigned __int64 offset = 0; offset < _threadPerGrid; offset += 1) {
		if (((mask[offset] & (1 << 0)) >> 0) == 1) { continue; } // skip padding
		if (((mask[offset] & (1 << 1)) >> 1) == 0)
		{// NON PML // can be merged to PML
			if (dielectric_flag== 0 && ((mask[offset] & (0b1111 << 4)) >> 4) > 0) // metal
			{	
				//// E

				tempx[offset] = (Hy[offset - _offsetZ] - Hy[offset] + Hz[offset - _offsetZ] - Hz[offset - _offsetY - _offsetZ]) * eps_r_inv[offset] * _cdt_div_dx; //eq30term1
				tempy[offset] = (Hz[offset - _offsetZ] - Hz[offset + _offsetX - _offsetZ] + Hx[offset] - Hx[offset - _offsetZ]) * eps_r_inv[offset] * _cdt_div_dx;
				tempz[offset] = (Hx[offset - _offsetX - _offsetY] - Hx[offset - _offsetX] + Hy[offset] - Hy[offset - _offsetX]) * eps_r_inv[offset] * _cdt_div_dx;
				tempx[offset] += eps0_c_Ex[offset] * (_eps0_ * _eps_inf - 0.5f * _sigma_ * _dt_ + _d2 - _C4p) / _eps0_; //eq30term2
				tempy[offset] += eps0_c_Ey[offset] * (_eps0_ * _eps_inf - 0.5f * _sigma_ * _dt_ + _d2 - _C4p) / _eps0_; //FIXME : div_eps0 coeffs cleanup
				tempz[offset] += eps0_c_Ez[offset] * (_eps0_ * _eps_inf - 0.5f * _sigma_ * _dt_ + _d2 - _C4p) / _eps0_;
				tempx[offset] -= eps0_c_Ex_old[offset] * _C5p / _eps0_; //eq30term3
				tempy[offset] -= eps0_c_Ey_old[offset] * _C5p / _eps0_;
				tempz[offset] -= eps0_c_Ez_old[offset] * _C5p / _eps0_;
				tempx[offset] -= eps0_c_Pdx[offset] * (_d1 - 1.0f) / _eps0_; //eq30term4
				tempy[offset] -= eps0_c_Pdy[offset] * (_d1 - 1.0f) / _eps0_;
				tempz[offset] -= eps0_c_Pdz[offset] * (_d1 - 1.0f) / _eps0_;
				tempx[offset] -= eps0_c_Pcp1x[offset] * (_C11 - 1.0f) / _eps0_; //eq30term5
				tempy[offset] -= eps0_c_Pcp1y[offset] * (_C11 - 1.0f) / _eps0_;
				tempz[offset] -= eps0_c_Pcp1z[offset] * (_C11 - 1.0f) / _eps0_;
				tempx[offset] -= eps0_c_Pcp2x[offset] * (_C12 - 1.0f) / _eps0_;
				tempy[offset] -= eps0_c_Pcp2y[offset] * (_C12 - 1.0f) / _eps0_;
				tempz[offset] -= eps0_c_Pcp2z[offset] * (_C12 - 1.0f) / _eps0_;
				tempx[offset] -= eps0_c_Pcp1x_old[offset] * _C21 / _eps0_; //eq30term6
				tempy[offset] -= eps0_c_Pcp1y_old[offset] * _C21 / _eps0_;
				tempz[offset] -= eps0_c_Pcp1z_old[offset] * _C21 / _eps0_;
				tempx[offset] -= eps0_c_Pcp2x_old[offset] * _C22 / _eps0_;
				tempy[offset] -= eps0_c_Pcp2y_old[offset] * _C22 / _eps0_;
				tempz[offset] -= eps0_c_Pcp2z_old[offset] * _C22 / _eps0_;
				tempx[offset] /= (_eps0_ * _eps_inf + 0.5f * _sigma_ * _dt_ - _d2 + _C3p) / _eps0_; //eq30term0
				tempy[offset] /= (_eps0_ * _eps_inf + 0.5f * _sigma_ * _dt_ - _d2 + _C3p) / _eps0_;
				tempz[offset] /= (_eps0_ * _eps_inf + 0.5f * _sigma_ * _dt_ - _d2 + _C3p) / _eps0_;

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


				//FIXME : can be simpler than this?
				eps0_c_Ex_old[offset] = eps0_c_Ex[offset];
				eps0_c_Ey_old[offset] = eps0_c_Ey[offset];
				eps0_c_Ez_old[offset] = eps0_c_Ez[offset];

				eps0_c_Ex[offset] = tempx[offset]; //FIXME : tempx size can be reduced: conider CUDA
				eps0_c_Ey[offset] = tempy[offset];
				eps0_c_Ez[offset] = tempz[offset];

				eps0_c_Pcp1x_old[offset] = eps0_c_Pcp1x[offset];
				eps0_c_Pcp1y_old[offset] = eps0_c_Pcp1y[offset];
				eps0_c_Pcp1z_old[offset] = eps0_c_Pcp1z[offset];

				eps0_c_Pcp2x_old[offset] = eps0_c_Pcp2x[offset];
				eps0_c_Pcp2y_old[offset] = eps0_c_Pcp2y[offset];
				eps0_c_Pcp2z_old[offset] = eps0_c_Pcp2z[offset];

				continue;
			}

			// non metal
			eps0_c_Ex[offset] += (Hy[offset - _offsetZ] - Hy[offset] + Hz[offset - _offsetZ] - Hz[offset - _offsetY - _offsetZ]) * eps_r_inv[offset] * _cdt_div_dx;
			eps0_c_Ey[offset] += (Hz[offset - _offsetZ] - Hz[offset + _offsetX - _offsetZ] + Hx[offset] - Hx[offset - _offsetZ]) * eps_r_inv[offset] * _cdt_div_dx;
			eps0_c_Ez[offset] += (Hx[offset - _offsetX - _offsetY] - Hx[offset - _offsetX] + Hy[offset] - Hy[offset - _offsetX]) * eps_r_inv[offset] * _cdt_div_dx;
			continue;
		}
		//PML //chap11.pdf
		psiXY_dx[offset] *= b_Y[offset];
		psiXY_dx[offset] += C_Y[offset] * (Hz[offset - _offsetZ] - Hz[offset - _offsetY - _offsetZ]) ;
		psiXZ_dx[offset] *= b_Z[offset];
		psiXZ_dx[offset] += C_Z[offset] * (-Hy[offset - _offsetZ] + Hy[offset]);
		psiYZ_dx[offset] *= b_Z[offset];
		psiYZ_dx[offset] += C_Z[offset] * (Hx[offset] - Hx[offset - _offsetZ]) ;
		psiYX_dx[offset] *= b_X[offset];
		psiYX_dx[offset] += C_X[offset] * (-Hz[offset - _offsetZ] + Hz[offset + _offsetX - _offsetZ]);
		psiZX_dx[offset] *= b_X[offset];
		psiZX_dx[offset] += C_X[offset] * (Hy[offset] - Hy[offset - _offsetX]) ;
		psiZY_dx[offset] *= b_Y[offset];
		psiZY_dx[offset] += C_Y[offset] * (-Hx[offset - _offsetX - _offsetY] + Hx[offset - _offsetX]);
		
		eps0_c_Ex[offset] += ((Hy[offset - _offsetZ] - Hy[offset]) / kappaZ[offset] + (Hz[offset - _offsetZ] - Hz[offset - _offsetY - _offsetZ]) / kappaY[offset]) * eps_r_inv[offset] * _cdt_div_dx; // constant can be merged;
		eps0_c_Ey[offset] += ((Hz[offset - _offsetZ] - Hz[offset + _offsetX - _offsetZ]) / kappaX[offset] + (Hx[offset] - Hx[offset - _offsetZ]) / kappaZ[offset]) * eps_r_inv[offset] * _cdt_div_dx; 
		eps0_c_Ez[offset] += ((Hx[offset - _offsetX - _offsetY] - Hx[offset - _offsetX]) / kappaY[offset] + (Hy[offset] - Hy[offset - _offsetX]) / kappaX[offset]) * eps_r_inv[offset] * _cdt_div_dx; 

		float tryError = 1.0f;// FIXME;
		eps0_c_Ex[offset] += (psiXY_dx[offset] - psiXZ_dx[offset]) * eps_r_inv[offset] * tryError; // constant can be merged;
		eps0_c_Ey[offset] += (psiYZ_dx[offset] - psiYX_dx[offset]) * eps_r_inv[offset] * tryError;
		eps0_c_Ez[offset] += (psiZX_dx[offset] - psiZY_dx[offset]) * eps_r_inv[offset] * tryError;
	}
}

int RFT_counter = 0;
void RFT(void) {
	if (RFT_counter < _STEP - RFT_WINDOW) {	RFT_counter++; return; }
	unsigned __int64 surf_x, surf_y, surf_z, offset;
	for (int i = 0; i < _SURF_SIZE_; i++) {
		for (int j = 0; j < FREQ_N; j++) {
			_SET_SURF_XYZ_INDEX(i);
			offset = _INDEX_XYZ(surf_x, surf_y, surf_z);
			//Yee cell mismatch consideration //FIXME : see 14.79
			//imag
			FT_eps0cE[i][j][0][1] -= 0.5 * (eps0_c_Ex[offset - _offsetX] + eps0_c_Ex[offset]) / (RFT_WINDOW - 1) * sinf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			FT_eps0cE[i][j][1][1] -= 0.5 * (eps0_c_Ey[offset - _offsetX - _offsetY] + eps0_c_Ey[offset - _offsetX]) / (RFT_WINDOW - 1) * sinf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			FT_eps0cE[i][j][2][1] -= 0.5 * (eps0_c_Ez[offset - _offsetZ] + eps0_c_Ez[offset]) / (RFT_WINDOW - 1) * sinf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			FT_H[i][j][0][1] -= 0.25 * (Hx[offset - _offsetX - _offsetY - _offsetZ] + Hx[offset - _offsetX - _offsetY] + Hx[offset - _offsetX - _offsetZ] + Hx[offset - _offsetX]) / (RFT_WINDOW - 1) * sinf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			FT_H[i][j][1][1] -= 0.25 * (Hy[offset - _offsetX - _offsetZ] + Hy[offset - _offsetX] + Hy[offset - _offsetZ] + Hy[offset]) / (RFT_WINDOW - 1) * sinf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			FT_H[i][j][2][1] -= 0.25 * (Hz[offset - _offsetZ - _offsetX - _offsetY] + Hz[offset - _offsetZ - _offsetX] + Hz[offset - _offsetZ - _offsetY] + Hz[offset - _offsetZ]) / (RFT_WINDOW - 1) * sinf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			if (RFT_counter == _STEP) { continue; }
			//real
			FT_eps0cE[i][j][0][0] += 0.5 * (eps0_c_Ex[offset - _offsetX] + eps0_c_Ex[offset]) / RFT_WINDOW * cosf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter); //RFT_counter is not q but it shoud not matter
			FT_eps0cE[i][j][1][0] += 0.5 * (eps0_c_Ey[offset - _offsetX - _offsetY] + eps0_c_Ey[offset - _offsetX]) / RFT_WINDOW * cosf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			FT_eps0cE[i][j][2][0] += 0.5 * (eps0_c_Ez[offset - _offsetZ] + eps0_c_Ez[offset]) / RFT_WINDOW * cosf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			FT_H[i][j][0][0] += 0.25 * (Hx[offset - _offsetX - _offsetY - _offsetZ] + Hx[offset - _offsetX - _offsetY] + Hx[offset - _offsetX - _offsetZ] + Hx[offset - _offsetX]) / RFT_WINDOW * cosf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter); //RFT_counter is not q but it shoud not matter
			FT_H[i][j][1][0] += 0.25 * (Hy[offset - _offsetX - _offsetZ] + Hy[offset - _offsetX] + Hy[offset - _offsetZ] + Hy[offset]) / RFT_WINDOW * cosf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			FT_H[i][j][2][0] += 0.25 * (Hz[offset - _offsetZ - _offsetX - _offsetY] + Hz[offset - _offsetZ - _offsetX] + Hz[offset - _offsetZ - _offsetY] + Hz[offset - _offsetZ]) / RFT_WINDOW * cosf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
		} //FIXME : use ipp functions!
	}
	if (RFT_counter == _STEP) { printf("RFT finish!\n"); }
	RFT_counter++;
}

////imag
//FT_eps0cE[i][j][0][1] -= eps0_c_Ex[offset] / (RFT_WINDOW - 1) * sinf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
//FT_eps0cE[i][j][1][1] -= eps0_c_Ey[offset] / (RFT_WINDOW - 1) * sinf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
//FT_eps0cE[i][j][2][1] -= eps0_c_Ez[offset] / (RFT_WINDOW - 1) * sinf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
//FT_H[i][j][0][1] -= Hx[offset] / (RFT_WINDOW - 1) * sinf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
//FT_H[i][j][1][1] -= Hy[offset] / (RFT_WINDOW - 1) * sinf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
//FT_H[i][j][2][1] -= Hz[offset] / (RFT_WINDOW - 1) * sinf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
//if (RFT_counter == _STEP) { break; }
////real
//FT_eps0cE[i][j][0][0] += eps0_c_Ex[offset] / RFT_WINDOW * cosf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter); //RFT_counter is not q but it shoud not matter
//FT_eps0cE[i][j][1][0] += eps0_c_Ey[offset] / RFT_WINDOW * cosf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
//FT_eps0cE[i][j][2][0] += eps0_c_Ez[offset] / RFT_WINDOW * cosf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
//FT_H[i][j][0][0] += Hx[offset] / RFT_WINDOW * cosf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
//FT_H[i][j][1][0] += Hy[offset] / RFT_WINDOW * cosf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
//FT_H[i][j][2][0] += Hz[offset] / RFT_WINDOW * cosf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);


#define NTFF_IMG_SIZE 200
#define _SURF_MidX_ ((_SURF_StartX_+_SURF_EndX_)/2.0f)
#define _SURF_MidY_ ((_SURF_StartY_+_SURF_EndY_)/2.0f)
#define _SURF_MidZ_ ((_SURF_StartZ_+_SURF_EndZ_)/2.0f)
#define DEBUG (1)

#include "mkl.h"
#include "ipp.h"
//#include "mkl_vml_functions.h"

void NTFF(void) {
	printf("\nNTFF calculation\n");
	float progress = 0;

	// Images
	char filename[] = "FF_00.png";
	unsigned error;
	unsigned char* image = malloc(NTFF_IMG_SIZE * NTFF_IMG_SIZE * 4);
	
	int surf_x, surf_y, surf_z;
	float k_vector;

	float *NF_eyePos;
	MKL_Complex8 *Hankel_dx, *NF_ecE, *NF_H, *normal;
	MKL_Complex8 *NF_ecM, *NF_J;
	MKL_Complex8 *NF_ecM_temp, *NF_J_temp;
	MKL_Complex8 *NF_ecL_dx, *NF_N_dx;
	MKL_Complex16 *NF_ecL_sum, *NF_N_sum;
	MKL_Complex8 *FF_ecE_x, *FF_ecE_y, *FF_ecE_z, *FF_H_x, *FF_H_y, *FF_H_z;
	MKL_Complex8 *FF_ecSr;
	
	NF_eyePos = (float*)mkl_malloc(NTFF_IMG_SIZE*NTFF_IMG_SIZE * 3 * sizeof(float),64); 
	Hankel_dx = (MKL_Complex8*)mkl_malloc(_SURF_SIZE_ * sizeof(MKL_Complex8), 64);
	NF_ecE = (MKL_Complex8*)mkl_malloc(3 * _SURF_SIZE_ * sizeof(MKL_Complex8), 64);
	NF_H = (MKL_Complex8*)mkl_malloc(3 * _SURF_SIZE_ * sizeof(MKL_Complex8), 64);
	normal = (MKL_Complex8*)mkl_malloc(3 * _SURF_SIZE_ * sizeof(MKL_Complex8), 64);

	NF_ecM = (MKL_Complex8*)mkl_malloc(3 * _SURF_SIZE_ * sizeof(MKL_Complex8), 64);
	NF_ecM_temp = (MKL_Complex8*)mkl_malloc(_SURF_SIZE_ * sizeof(MKL_Complex8), 64);
	NF_J = (MKL_Complex8*)mkl_malloc(3 * _SURF_SIZE_ * sizeof(MKL_Complex8), 64);
	NF_J_temp = (MKL_Complex8*)mkl_malloc(_SURF_SIZE_ * sizeof(MKL_Complex8), 64);

	NF_ecL_dx = (MKL_Complex8*)mkl_malloc(3 * _SURF_SIZE_ * sizeof(MKL_Complex8), 64);
	NF_N_dx = (MKL_Complex8*)mkl_malloc(3 * _SURF_SIZE_ * sizeof(MKL_Complex8), 64);
	NF_ecL_sum = (MKL_Complex16*)mkl_malloc(3 * NTFF_IMG_SIZE*NTFF_IMG_SIZE * sizeof(MKL_Complex16), 64);
	NF_N_sum = (MKL_Complex16*)mkl_malloc(3 * NTFF_IMG_SIZE*NTFF_IMG_SIZE * sizeof(MKL_Complex16), 64);

	FF_ecE_x = (MKL_Complex8*)mkl_malloc(NTFF_IMG_SIZE*NTFF_IMG_SIZE * sizeof(MKL_Complex8), 64);
	FF_ecE_y = (MKL_Complex8*)mkl_malloc(NTFF_IMG_SIZE*NTFF_IMG_SIZE * sizeof(MKL_Complex8), 64);
	FF_ecE_z = (MKL_Complex8*)mkl_malloc(NTFF_IMG_SIZE*NTFF_IMG_SIZE * sizeof(MKL_Complex8), 64);
	FF_H_x = (MKL_Complex8*)mkl_malloc(NTFF_IMG_SIZE*NTFF_IMG_SIZE * sizeof(MKL_Complex8), 64);
	FF_H_y = (MKL_Complex8*)mkl_malloc(NTFF_IMG_SIZE*NTFF_IMG_SIZE * sizeof(MKL_Complex8), 64);
	FF_H_z = (MKL_Complex8*)mkl_malloc(NTFF_IMG_SIZE*NTFF_IMG_SIZE * sizeof(MKL_Complex8), 64);
	FF_ecSr = (MKL_Complex8*)mkl_malloc(NTFF_IMG_SIZE*NTFF_IMG_SIZE * sizeof(MKL_Complex8), 64);

	
	//precalculation per image
	//theta (0~2pi), phi_upz (0~pi/2), phi_downz(=-phi_upz, -pi/2~0)
	for (int i = 0; i < NTFF_IMG_SIZE; i++) {
		for (int j = 0; j < NTFF_IMG_SIZE; j++) {
			float radius = ((float)NTFF_IMG_SIZE - 1.0f) / 2.0f ;
			float xx = ((float)i - radius );
			float yy = ((float)j - radius );
			float rr = sqrtf(xx*xx + yy*yy);
			if (radius < rr) {
				NF_eyePos[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i] = -1.0f;
				NF_eyePos[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i] = -1.0f;
				NF_eyePos[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i] = -1.0f;
				continue; }
			if ( rr < 0.1f) { // ==0.0f
				printf("center : %d, %d \n\n",i,j);
				NF_eyePos[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i] = 0.0001f; //FIXME
				NF_eyePos[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i] = 0.0001f;
				NF_eyePos[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i] = 1.0f;
				continue;
			}
			float NF_phi = 0.5f * M_PI * (radius - rr) / radius;
			NF_eyePos[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i] = cosf(NF_phi) * xx / rr;
			NF_eyePos[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i] = cosf(NF_phi) * yy / rr;    
			NF_eyePos[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i] = sinf(NF_phi);
		}
	}


	for (int freqN = 0; freqN < FREQ_N; freqN++) { //each frequency
		k_vector = RFT_K_LIST_CALCULATED[freqN] * 2.0f * M_PI / RFT_WINDOW / _c0 / _dt_;
		//J,M calculation
		for (int k = 0; k < _SURF_SIZE_; k++) {
			NF_ecE[0 * _SURF_SIZE_ + k].real = (float)FT_eps0cE[k][freqN][0][0];
			NF_ecE[1 * _SURF_SIZE_ + k].real = (float)FT_eps0cE[k][freqN][1][0];
			NF_ecE[2 * _SURF_SIZE_ + k].real = (float)FT_eps0cE[k][freqN][2][0];
			NF_ecE[0 * _SURF_SIZE_ + k].imag = (float)FT_eps0cE[k][freqN][0][1];
			NF_ecE[1 * _SURF_SIZE_ + k].imag = (float)FT_eps0cE[k][freqN][1][1];
			NF_ecE[2 * _SURF_SIZE_ + k].imag = (float)FT_eps0cE[k][freqN][2][1];
			NF_H[0 * _SURF_SIZE_ + k].real = (float)FT_H[k][freqN][0][0];
			NF_H[1 * _SURF_SIZE_ + k].real = (float)FT_H[k][freqN][1][0];
			NF_H[2 * _SURF_SIZE_ + k].real = (float)FT_H[k][freqN][2][0];
			NF_H[0 * _SURF_SIZE_ + k].imag = (float)FT_H[k][freqN][0][1];
			NF_H[1 * _SURF_SIZE_ + k].imag = (float)FT_H[k][freqN][1][1];
			NF_H[2 * _SURF_SIZE_ + k].imag = (float)FT_H[k][freqN][2][1];
		}

		//normal vector  FIXME : inefficient
		float normal_temp;
		for (int k = 0; k < _SURF_SIZE_; k++) {
			_SET_SURF_XYZ_INDEX(k);
			normal[0 * _SURF_SIZE_ + k].real = (surf_x == _SURF_StartX_) ? -1 : ((surf_x == _SURF_EndX_) ? 1 : 0);
			normal[1 * _SURF_SIZE_ + k].real = (surf_y == _SURF_StartY_) ? -1 : ((surf_y == _SURF_EndY_) ? 1 : 0);
			normal[2 * _SURF_SIZE_ + k].real = (surf_z == _SURF_StartZ_) ? -1 : ((surf_z == _SURF_EndZ_) ? 1 : 0);
			normal[0 * _SURF_SIZE_ + k].imag = 0;
			normal[1 * _SURF_SIZE_ + k].imag = 0;
			normal[2 * _SURF_SIZE_ + k].imag = 0;
			normal_temp = sqrtf(
				normal[0 * _SURF_SIZE_ + k].real*normal[0 * _SURF_SIZE_ + k].real
				+ normal[1 * _SURF_SIZE_ + k].real*normal[1 * _SURF_SIZE_ + k].real
				+ normal[2 * _SURF_SIZE_ + k].real*normal[2 * _SURF_SIZE_ + k].real);
			normal[0 * _SURF_SIZE_ + k].real /= normal_temp;
			normal[1 * _SURF_SIZE_ + k].real /= normal_temp;
			normal[2 * _SURF_SIZE_ + k].real /= normal_temp;
		}

		////M, J calc
		// minus curl ==> {{0, nz, -ny}, {-nz, 0, nx}, {ny, -nx, 0}}
		vcMul((const int)(_SURF_SIZE_), normal + 2 * _SURF_SIZE_, NF_ecE + 1 * _SURF_SIZE_, NF_ecM_temp);
		ippsAdd_32fc_I(NF_ecM_temp, NF_ecM + 0 * _SURF_SIZE_, _SURF_SIZE_);
		vcMul((const int)(_SURF_SIZE_), normal + 1 * _SURF_SIZE_, NF_ecE + 2 * _SURF_SIZE_, NF_ecM_temp);
		ippsSub_32fc_I(NF_ecM_temp, NF_ecM + 0 * _SURF_SIZE_, _SURF_SIZE_);
		vcMul((const int)(_SURF_SIZE_), normal + 0 * _SURF_SIZE_, NF_ecE + 2 * _SURF_SIZE_, NF_ecM_temp);
		ippsAdd_32fc_I(NF_ecM_temp, NF_ecM + 1 * _SURF_SIZE_, _SURF_SIZE_);
		vcMul((const int)(_SURF_SIZE_), normal + 2 * _SURF_SIZE_, NF_ecE + 0 * _SURF_SIZE_, NF_ecM_temp);
		ippsSub_32fc_I(NF_ecM_temp, NF_ecM + 1 * _SURF_SIZE_, _SURF_SIZE_);
		vcMul((const int)(_SURF_SIZE_), normal + 1 * _SURF_SIZE_, NF_ecE + 0 * _SURF_SIZE_, NF_ecM_temp);
		ippsAdd_32fc_I(NF_ecM_temp, NF_ecM + 2 * _SURF_SIZE_, _SURF_SIZE_);
		vcMul((const int)(_SURF_SIZE_), normal + 0 * _SURF_SIZE_, NF_ecE + 1 * _SURF_SIZE_, NF_ecM_temp);
		ippsSub_32fc_I(NF_ecM_temp, NF_ecM + 2 * _SURF_SIZE_, _SURF_SIZE_);
		// plus curl ==> {{0, -nz, ny,} {nz, 0, -nx}, {-ny, nx, 0}}
		vcMul((const int)(_SURF_SIZE_), normal + 2 * _SURF_SIZE_, NF_H + 1 * _SURF_SIZE_, NF_J_temp);
		ippsSub_32fc_I(NF_J_temp, NF_J + 0 * _SURF_SIZE_, _SURF_SIZE_);
		vcMul((const int)(_SURF_SIZE_), normal + 1 * _SURF_SIZE_, NF_H + 2 * _SURF_SIZE_, NF_J_temp);
		ippsAdd_32fc_I(NF_J_temp, NF_J + 0 * _SURF_SIZE_, _SURF_SIZE_);
		vcMul((const int)(_SURF_SIZE_), normal + 0 * _SURF_SIZE_, NF_H + 2 * _SURF_SIZE_, NF_J_temp);
		ippsSub_32fc_I(NF_J_temp, NF_J + 1 * _SURF_SIZE_, _SURF_SIZE_);
		vcMul((const int)(_SURF_SIZE_), normal + 2 * _SURF_SIZE_, NF_H + 0 * _SURF_SIZE_, NF_J_temp);
		ippsAdd_32fc_I(NF_J_temp, NF_J + 1 * _SURF_SIZE_, _SURF_SIZE_);
		vcMul((const int)(_SURF_SIZE_), normal + 1 * _SURF_SIZE_, NF_H + 0 * _SURF_SIZE_, NF_J_temp);
		ippsSub_32fc_I(NF_J_temp, NF_J + 2 * _SURF_SIZE_, _SURF_SIZE_);
		vcMul((const int)(_SURF_SIZE_), normal + 0 * _SURF_SIZE_, NF_H + 1 * _SURF_SIZE_, NF_J_temp);
		ippsAdd_32fc_I(NF_J_temp, NF_J + 2 * _SURF_SIZE_, _SURF_SIZE_);

		////L N clac : we don't need Lr, Nr but anyway
		printf("Phase calculation\n");
		ippsZero_64fc(NF_ecL_sum, 3*NTFF_IMG_SIZE *NTFF_IMG_SIZE);
		ippsZero_64fc(NF_N_sum, 3 * NTFF_IMG_SIZE *NTFF_IMG_SIZE);
		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			printf("%f%%\r", 100.0f*(float)i / (NTFF_IMG_SIZE - 1.0f));
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float radius = ((float)NTFF_IMG_SIZE - 1.0f) / 2.0f;
				float xx = ((float)i - radius);
				float yy = ((float)j - radius);
				if (radius < sqrtf(xx*xx + yy*yy)) { continue; }
				for (int k = 0; k < _SURF_SIZE_; k++) { //Recalculation over freq. Saves memory, though
					_SET_SURF_XYZ_INDEX(k);
					Hankel_dx[k].real = 0;
					Hankel_dx[k].imag = k_vector * _dx *(
						  ((float)surf_x - (_SURF_MidX_))*(NF_eyePos[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i])
						+ ((float)surf_y - (_SURF_MidY_))*(NF_eyePos[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i])
						+ ((float)surf_z - (_SURF_MidZ_))*(NF_eyePos[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i])
						); 
				}
				vcExp(_SURF_SIZE_, Hankel_dx, Hankel_dx); //complex exp 

				for (int ii = 0; ii < 3; ii++) { // R theta phi
					vcMul(_SURF_SIZE_, Hankel_dx, NF_ecM + ii * _SURF_SIZE_, NF_ecL_dx + ii * _SURF_SIZE_); 
					vcMul(_SURF_SIZE_, Hankel_dx, NF_J   + ii * _SURF_SIZE_,   NF_N_dx + ii * _SURF_SIZE_); 
					for (int kk = 0; kk < _SURF_SIZE_; kk++) {
						if (isfinite(NF_ecL_dx[kk + ii*_SURF_SIZE_].real) == 0) { printf("error!\n"); }
						NF_ecL_sum[ii * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i].real += NF_ecL_dx[kk + ii * _SURF_SIZE_].real ;
						NF_ecL_sum[ii * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i].imag += NF_ecL_dx[kk + ii * _SURF_SIZE_].imag ;
						NF_N_sum[ii * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i].real += NF_N_dx[kk + ii * _SURF_SIZE_].real ;
						NF_N_sum[ii * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i].imag += NF_N_dx[kk + ii * _SURF_SIZE_].imag ;
					}//use ippsSum_64fc
				}
			}
		}

		printf("\nNTFF calculation\n");

		//ippsZero_32fc(FF_ecE_x, NTFF_IMG_SIZE*NTFF_IMG_SIZE);
		//ippsZero_32fc(FF_ecE_y, NTFF_IMG_SIZE*NTFF_IMG_SIZE);
		//ippsZero_32fc(FF_ecE_z, NTFF_IMG_SIZE*NTFF_IMG_SIZE);
		//ippsZero_32fc(FF_H_x, NTFF_IMG_SIZE*NTFF_IMG_SIZE);
		//ippsZero_32fc(FF_H_y, NTFF_IMG_SIZE*NTFF_IMG_SIZE);
		//ippsZero_32fc(FF_H_z, NTFF_IMG_SIZE*NTFF_IMG_SIZE);
		//ippsZero_32fc(FF_ecSr, NTFF_IMG_SIZE*NTFF_IMG_SIZE);

		////ch14, (14.54) from uFDTD
		////ch18, (8.63) from http://my.ece.ucsb.edu/York/Bobsclass/201B/W01/chap8.pdf //check M sign 
		for (int i = 0; i < NTFF_IMG_SIZE; i++) { //FIXME : vectorization : use vzMul, ippsAdd_64fc_I
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float radius = ((float)NTFF_IMG_SIZE - 1.0f) / 2.0f;
				float xx = ((float)i - radius);
				float yy = ((float)j - radius);
				if (radius < sqrtf(xx*xx + yy*yy)) { continue; }

				// E = N + L x [xyz]_eye
				// Ex = Nx + Ly*eye_z - Lz*eye_y
				FF_ecE_x[i + j*NTFF_IMG_SIZE].real =
					NF_N_sum[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].real
					+ NF_ecL_sum[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].real * NF_eyePos[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE]
					- NF_ecL_sum[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].real * NF_eyePos[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE];
				FF_ecE_y[i + j*NTFF_IMG_SIZE].real =
					NF_N_sum[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].real
					+ NF_ecL_sum[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].real * NF_eyePos[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE]
					- NF_ecL_sum[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].real * NF_eyePos[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE];
				FF_ecE_z[i + j*NTFF_IMG_SIZE].real =
					NF_N_sum[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].real
					+ NF_ecL_sum[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].real * NF_eyePos[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE]
					- NF_ecL_sum[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].real * NF_eyePos[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE];
				FF_H_x[i + j*NTFF_IMG_SIZE].real =
					NF_ecL_sum[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].real
					+ NF_N_sum[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].real * NF_eyePos[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE]
					- NF_N_sum[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].real * NF_eyePos[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE];
				FF_H_y[i + j*NTFF_IMG_SIZE].real =
					NF_ecL_sum[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].real
					+ NF_N_sum[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].real * NF_eyePos[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE]
					- NF_N_sum[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].real * NF_eyePos[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE];
				FF_H_z[i + j*NTFF_IMG_SIZE].real =
					NF_ecL_sum[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].real
					+ NF_N_sum[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].real * NF_eyePos[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE]
					- NF_N_sum[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].real * NF_eyePos[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE];
				FF_ecE_x[i + j*NTFF_IMG_SIZE].imag =
					NF_N_sum[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].imag;
					+ NF_ecL_sum[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].imag * NF_eyePos[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE]
					- NF_ecL_sum[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].imag * NF_eyePos[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE];
				FF_ecE_y[i + j*NTFF_IMG_SIZE].imag =
					NF_N_sum[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].imag
					+ NF_ecL_sum[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].imag * NF_eyePos[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE]
					- NF_ecL_sum[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].imag * NF_eyePos[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE];
				FF_ecE_z[i + j*NTFF_IMG_SIZE].imag =
					NF_N_sum[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].imag
					+ NF_ecL_sum[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].imag * NF_eyePos[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE]
					- NF_ecL_sum[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].imag * NF_eyePos[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE];
				FF_H_x[i + j*NTFF_IMG_SIZE].imag =
					NF_ecL_sum[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].imag
					+ NF_N_sum[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].imag * NF_eyePos[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE]
					- NF_N_sum[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].imag * NF_eyePos[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE];
				FF_H_y[i + j*NTFF_IMG_SIZE].imag =
					NF_ecL_sum[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].imag
					+ NF_N_sum[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].imag * NF_eyePos[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE]
					- NF_N_sum[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].imag * NF_eyePos[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE];
				FF_H_z[i + j*NTFF_IMG_SIZE].imag =
					NF_ecL_sum[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].imag
					+ NF_N_sum[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].imag * NF_eyePos[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE]
					- NF_N_sum[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].imag * NF_eyePos[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE];
			}
		} 

		for (int i = 0; i < NTFF_IMG_SIZE; i++) {  //FIXME : vectorization : use mkl_malloc and mkl functions
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float radius = ((float)NTFF_IMG_SIZE - 1.0f) / 2.0f;
				float xx = ((float)i - radius);
				float yy = ((float)j - radius);
				if (radius < sqrtf(xx*xx + yy*yy)) { continue; }
				////Vr = [Vx,Vy,Vz] . [rx,ry,rz]
				////Vtheta = [Vx,Vy,Vz] . [-ry,rx,0]
				////Vphi = [Vx,Vy,Vz] . [-rx*rz,-ry*rz,+rr*rr] / rr
				//Sr = Sx * eye_x + Sy * eye_y + Sz * eye_z ;  Sx = EyHz-EzHy ; 
				//Polarization calculation possible   // e.g. Ey polarization S1x = EyHz // S = 0.5 E H*
				FF_ecSr[i + j*NTFF_IMG_SIZE].real = 0.5 * (
					(FF_ecE_y[i + j*NTFF_IMG_SIZE].real * FF_H_z[i + j*NTFF_IMG_SIZE].real + FF_ecE_y[i + j*NTFF_IMG_SIZE].imag * FF_H_z[i + j*NTFF_IMG_SIZE].imag
						- FF_ecE_z[i + j*NTFF_IMG_SIZE].real * FF_H_y[i + j*NTFF_IMG_SIZE].real - FF_ecE_z[i + j*NTFF_IMG_SIZE].imag * FF_H_y[i + j*NTFF_IMG_SIZE].imag)
					* NF_eyePos[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE] +
					(FF_ecE_z[i + j*NTFF_IMG_SIZE].real * FF_H_x[i + j*NTFF_IMG_SIZE].real + FF_ecE_z[i + j*NTFF_IMG_SIZE].imag * FF_H_x[i + j*NTFF_IMG_SIZE].imag
						- FF_ecE_x[i + j*NTFF_IMG_SIZE].real * FF_H_z[i + j*NTFF_IMG_SIZE].real - FF_ecE_x[i + j*NTFF_IMG_SIZE].imag * FF_H_z[i + j*NTFF_IMG_SIZE].imag)
					* NF_eyePos[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE] +
					(FF_ecE_x[i + j*NTFF_IMG_SIZE].real * FF_H_y[i + j*NTFF_IMG_SIZE].real + FF_ecE_x[i + j*NTFF_IMG_SIZE].imag * FF_H_y[i + j*NTFF_IMG_SIZE].imag
						- FF_ecE_y[i + j*NTFF_IMG_SIZE].real * FF_H_x[i + j*NTFF_IMG_SIZE].real - FF_ecE_y[i + j*NTFF_IMG_SIZE].imag * FF_H_x[i + j*NTFF_IMG_SIZE].imag)
					* NF_eyePos[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE]
					);
				FF_ecSr[i + j*NTFF_IMG_SIZE].imag = 0.5 * (
					(-FF_ecE_y[i + j*NTFF_IMG_SIZE].real * FF_H_z[i + j*NTFF_IMG_SIZE].imag - FF_ecE_y[i + j*NTFF_IMG_SIZE].imag * FF_H_z[i + j*NTFF_IMG_SIZE].real
						+ FF_ecE_z[i + j*NTFF_IMG_SIZE].real * FF_H_y[i + j*NTFF_IMG_SIZE].imag + FF_ecE_z[i + j*NTFF_IMG_SIZE].imag * FF_H_y[i + j*NTFF_IMG_SIZE].real)
					* NF_eyePos[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE] +
					(-FF_ecE_z[i + j*NTFF_IMG_SIZE].real * FF_H_x[i + j*NTFF_IMG_SIZE].imag - FF_ecE_z[i + j*NTFF_IMG_SIZE].imag * FF_H_x[i + j*NTFF_IMG_SIZE].real
						+ FF_ecE_x[i + j*NTFF_IMG_SIZE].real * FF_H_z[i + j*NTFF_IMG_SIZE].imag + FF_ecE_x[i + j*NTFF_IMG_SIZE].imag * FF_H_z[i + j*NTFF_IMG_SIZE].real)
					* NF_eyePos[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE] +
					(-FF_ecE_x[i + j*NTFF_IMG_SIZE].real * FF_H_y[i + j*NTFF_IMG_SIZE].imag - FF_ecE_x[i + j*NTFF_IMG_SIZE].imag * FF_H_y[i + j*NTFF_IMG_SIZE].real
						+ FF_ecE_y[i + j*NTFF_IMG_SIZE].real * FF_H_x[i + j*NTFF_IMG_SIZE].imag + FF_ecE_y[i + j*NTFF_IMG_SIZE].imag * FF_H_x[i + j*NTFF_IMG_SIZE].real)
					* NF_eyePos[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE]
					);
			}
		}

//		ippsMulC_32fc_I((Ipp32fc) { sqrtf(0.125f / M_PI * k_vector,0.0f) }, FF_ecSr, NTFF_IMG_SIZE*NTFF_IMG_SIZE);
		ippsMulC_32fc_I((Ipp32fc) { sqrtf(0.125f / M_PI * k_vector) , 0.0f}, FF_ecSr, NTFF_IMG_SIZE*NTFF_IMG_SIZE);
		ippsMulC_32fc_I((Ipp32fc) { sqrtf(0.125f / M_PI * k_vector) , 0.0f}, FF_ecE_x, NTFF_IMG_SIZE*NTFF_IMG_SIZE);		//for plotting
		ippsMulC_32fc_I((Ipp32fc) { sqrtf(0.125f / M_PI * k_vector) , 0.0f}, FF_ecE_y, NTFF_IMG_SIZE*NTFF_IMG_SIZE);
		ippsMulC_32fc_I((Ipp32fc) { sqrtf(0.125f / M_PI * k_vector) , 0.0f}, FF_ecE_z, NTFF_IMG_SIZE*NTFF_IMG_SIZE);
		ippsMulC_32fc_I((Ipp32fc) { sqrtf(0.125f / M_PI * k_vector) , 0.0f}, FF_H_x, NTFF_IMG_SIZE*NTFF_IMG_SIZE);
		ippsMulC_32fc_I((Ipp32fc) { sqrtf(0.125f / M_PI * k_vector) , 0.0f}, FF_H_y, NTFF_IMG_SIZE*NTFF_IMG_SIZE);
		ippsMulC_32fc_I((Ipp32fc) { sqrtf(0.125f / M_PI * k_vector), 0.0f }, FF_H_z, NTFF_IMG_SIZE*NTFF_IMG_SIZE);

		filename[4] = '0' + (char)(freqN);

		//debug
		filename[3] = 'a';
		for (int i = 0; i < NTFF_IMG_SIZE; i++)	{		for (int j = 0; j < NTFF_IMG_SIZE; j++) {
			float val = FF_ecSr[j*NTFF_IMG_SIZE + i].real * 255.0f ;
			
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val>0?(val<255?val:255):0 ;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val<0?(val>-255?-val:255):0 ;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
				//printf("%1.2e\t", val);
		}
		//printf("\n");
		}
		error = lodepng_encode32_file(filename, image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);
		if (error) printf("error %u: %s\n", error, lodepng_error_text(error));


		unsigned char* image2 = malloc(_SURF_DX_ * _SURF_DY_ * 4);
		filename[3] = 'b';
		for (int j = 0; j < _SURF_DY_ ; j++)	{
			for (int i = 0; i < _SURF_DX_ ; i++) {
				float val = NF_J[(0 * _SURF_SIZE_) + _SURF_INDEX_XYZ(_SURF_StartX_ + i, _SURF_StartY_ + j, _SURF_StartZ_)].real *255.0f * 10000.0f;
				image2[4 * _SURF_DX_ * j + 4 * i + 0] = val>0 ? val : 0;
				image2[4 * _SURF_DX_ * j + 4 * i + 1] = val<0 ? -val : 0;
				image2[4 * _SURF_DX_ * j + 4 * i + 2] = 0;
				image2[4 * _SURF_DX_ * j + 4 * i + 3] = 255;
			}
		}
		error = lodepng_encode32_file(filename, image2, _SURF_DX_ , _SURF_DY_);
		if (error) printf("error %u: %s\n", error, lodepng_error_text(error));;
		free(image2);


		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = sqrtf(0.125f / M_PI * k_vector)* NF_N_sum[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i].real * 255.0f / 300.0f;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}error = lodepng_encode32_file("Nx.png", image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);

		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = sqrtf(0.125f / M_PI * k_vector)* NF_N_sum[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i].real * 255.0f / 300.0f;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}error = lodepng_encode32_file("Ny.png", image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);

		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = sqrtf(0.125f / M_PI * k_vector)* NF_N_sum[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i].real * 255.0f / 300.0f;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}error = lodepng_encode32_file("Nz.png", image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);

		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = sqrtf(0.125f / M_PI * k_vector)*(+NF_ecL_sum[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].real * NF_eyePos[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE])
					* 255.0f / 300.0f;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}error = lodepng_encode32_file("Lx.png", image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);

		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = sqrtf(0.125f / M_PI * k_vector)*(+NF_ecL_sum[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].real * NF_eyePos[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE])
				* 255.0f / 300.0f;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}error = lodepng_encode32_file("Ly.png", image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);

		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = sqrtf(0.125f / M_PI * k_vector)*(+NF_ecL_sum[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].real * NF_eyePos[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE])
					* 255.0f / 300.0f;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}error = lodepng_encode32_file("Lz.png", image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);

		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = FF_ecE_x[j*NTFF_IMG_SIZE + i].real * 255.0f / 300.0f;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}error = lodepng_encode32_file("E_x.png", image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);

		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = FF_ecE_y[j*NTFF_IMG_SIZE + i].real * 255.0f / 300.0f;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}error = lodepng_encode32_file("E_y.png", image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);

		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = FF_ecE_z[j*NTFF_IMG_SIZE + i].real * 255.0f / 300.0f;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}error = lodepng_encode32_file("E_z.png", image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);

		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = 0.1*FF_H_x[j*NTFF_IMG_SIZE + i].real * 255.0f / 300.0f;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}error = lodepng_encode32_file("H_x.png", image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);

		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = FF_H_y[j*NTFF_IMG_SIZE + i].real * 255.0f / 300.0f;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}error = lodepng_encode32_file("H_y.png", image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);

		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = FF_H_z[j*NTFF_IMG_SIZE + i].real * 255.0f / 300.0f;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}error = lodepng_encode32_file("H_z.png", image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);


		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = sqrtf(FF_ecSr[j*NTFF_IMG_SIZE + i].real *  FF_ecSr[j*NTFF_IMG_SIZE + i].real + FF_ecSr[j*NTFF_IMG_SIZE + i].imag *  FF_ecSr[j*NTFF_IMG_SIZE + i].imag) * 255.0f / 300.0f;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}error = lodepng_encode32_file("S_r_phasor.png", image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);

		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = FF_ecSr[j*NTFF_IMG_SIZE + i].real * 255.0f / 300.0f;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}error = lodepng_encode32_file("S_r_avg.png", image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);

	}

	free(image);
	mkl_free(NF_eyePos);
	mkl_free(Hankel_dx);
	mkl_free(NF_ecE); mkl_free(NF_H); mkl_free(normal);
	mkl_free(NF_ecM); mkl_free(NF_J);
	mkl_free(NF_ecM_temp); mkl_free(NF_J_temp);
	mkl_free(NF_ecL_dx); mkl_free(NF_N_dx);
	mkl_free(NF_ecL_sum); mkl_free(NF_N_sum);
	mkl_free(FF_ecE_x); mkl_free(FF_ecE_y); mkl_free(FF_ecE_z); mkl_free(FF_H_x); mkl_free(FF_H_y); mkl_free(FF_H_z);
	mkl_free(FF_ecSr); 

}

int init(void)
{
	//tile info
	printf("total area : block(%d,%d,%d) * grid(%d,%d,%d) = %d(%d,%d,%d)\n", _blockDimX, _blockDimY, _blockDimZ, _gridDimX, _gridDimY, _gridDimZ, _threadPerGrid, _blockDimX*_gridDimX, _blockDimY*_gridDimY, _blockDimZ*_gridDimZ);
	printf("thread space (padding overhead) : %d(%d,%d,%d) \n", (_blockDimX - 2)*_gridDimX* (_blockDimY - 2)*_gridDimY* (_blockDimZ - 2)*_gridDimZ, (_blockDimX - 2)*_gridDimX, (_blockDimY - 2)*_gridDimY, (_blockDimZ - 2)*_gridDimZ);
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
	printf("\n");


	//DCP info
	printf("DCP : \n");
	printf("eq30term2 x : %e\n", (_eps0_ * _eps_inf - 0.5f * _sigma_ * _dt_ + _d2 - _C4p) / _eps0_);
	printf("eq30term3 x : %e\n", (_C5p) / _eps0_);
	printf("eq30term0 x : %e\n", (_eps0_ * _eps_inf + 0.5f * _sigma_ * _dt_ - _d2 + _C3p) / _eps0_);
	printf("\n");

	//RFT/NTFF setup
	printf("RFT/NTFF freqs : ");
	for (int i = 0; i < FREQ_N; i++)
	{
		RFT_K_LIST_CALCULATED[i] = roundf(FREQ_LIST_DESIRED[i] * _dt_ * RFT_WINDOW);
		printf("%e(%e), ", RFT_K_LIST_CALCULATED[i] / _dt_ / RFT_WINDOW, _c0 / RFT_K_LIST_CALCULATED[i] * _dt_ * RFT_WINDOW);
	}
	printf("\n\n");

	//PML constants
	float pml_eps0 = _eps0_;
	float pml_sigma_x_dt_div_eps0_max = -((float)pml_n + 1.0f)*logf(pml_R) * 0.5f / (_PML_PX_X_) / _S_factor; //eq37
	float pml_sigma_y_dt_div_eps0_max = -((float)pml_n + 1.0f)*logf(pml_R) * 0.5f / (_PML_PX_Y_) / _S_factor;
	float pml_sigma_z_dt_div_eps0_max = -((float)pml_n + 1.0f)*logf(pml_R) * 0.5f / (_PML_PX_Z_) / _S_factor;
	printf("PML : \n");
	printf("_eps0_ = %e\n", _eps0_);
	printf("pml_eps0 = %e\n", pml_eps0);
	printf("(_PML_PX_X_) = %d\n", (_PML_PX_X_));
	printf("pml_n = %d\n", pml_n);
	printf("pml_R = %e\n", pml_R);
	printf("pml_kappa_max = %e\n", pml_kappa_max);
	printf("pml_sigma_x_dt_div_eps0_max = %e\n", pml_sigma_x_dt_div_eps0_max);
	printf("\n");

	for (unsigned __int64 i = 0; i < _threadPerGrid; i++)
	{
		eps_r_inv[i] = 1.0f;
		kappaX[i] = 1.0f;
		kappaY[i] = 1.0f;
		kappaZ[i] = 1.0f;
		mask[i] = 0;

		//Indexing
		unsigned __int64 tmp = i;
		int X = 0, Y = 0, Z = 0;
		X += tmp % _blockDimX - 1; tmp /= _blockDimX;
		Y += tmp % _blockDimY - 1; tmp /= _blockDimY;
		Z += tmp % _blockDimZ - 1; tmp /= _blockDimZ;
		if (X == -1 || X == _blockDimX - 2 || Y == -1 || Y == _blockDimY - 2 || Z == -1 || Z == _blockDimZ - 2) {
			mask[i] |= (1 << 0); // 0th bit : Padding 
			continue;
		}
		X += (tmp % _gridDimX) * (_blockDimX - 2); tmp /= _gridDimX;
		Y += (tmp % _gridDimY) * (_blockDimY - 2); tmp /= _gridDimY;
		Z += (tmp % _gridDimZ) * (_blockDimZ - 2); tmp /= _gridDimZ;

		alpha_dt_div_eps0[i] = (_PML_ALPHA_TUNING_); // FIXME 

		//FIXME : check PML area
		if ((X + 1 <= ((_PML_PX_X_)) || (_DimX)-((_PML_PX_X_)) <= X)) {
			//if ((X + 1 <= ((_PML_PX_X_)) || (_DimX)-((_PML_PX_X_)) <= X) && ((_PML_PX_Y_)) < Y+1 && Y < (_DimY)-((_PML_PX_Y_)) && ((_PML_PX_Z_)) < Z + 1 && Z < (_DimZ)-((_PML_PX_Z_)) ) {
			mask[i] |= (1 << 1); // 1st bit : PML
			sigmaX_dt_div_eps0[i] = pml_sigma_x_dt_div_eps0_max * pow(fmin(fabs(((_PML_PX_X_))-X), fabs(((_PML_PX_X_))+X - (_DimX)+1)) / ((_PML_PX_X_)), (pml_n));
			kappaX[i] = 1.0f + (pml_kappa_max - 1.0f) * pow((fmin(fabs(((_PML_PX_X_))-X), fabs(((_PML_PX_X_))+X - (_DimX)+1)) - 1.0f)/ ((_PML_PX_X_)), (pml_n)); //FIXME : 0.5f? -1.0f? check 
			b_X[i] = expf( -alpha_dt_div_eps0[i] - sigmaX_dt_div_eps0[i]/kappaX[i] ); //close to 0
			C_X[i] = sigmaX_dt_div_eps0[i] / (sigmaX_dt_div_eps0[i] * kappaX[i] + alpha_dt_div_eps0[i] * kappaX[i] * kappaX[i] ) * (b_X[i] - 1.0f); //close to 1
		}
		if ((Y + 1 <= ((_PML_PX_Y_)) || (_DimY)-((_PML_PX_Y_)) <= Y)) {
			//if ((Y + 1 <= ((_PML_PX_Y_)) || (_DimY)-((_PML_PX_Y_)) <= Y) && ((_PML_PX_Z_)) < Z + 1 && Z < (_DimY)-((_PML_PX_Z_)) && ((_PML_PX_X_)) < X + 1 && X < (_DimX)-((_PML_PX_X_)) ) {
			mask[i] |= (1 << 1); // 1st bit : PML
			sigmaY_dt_div_eps0[i] = pml_sigma_y_dt_div_eps0_max * pow(fmin(fabs(((_PML_PX_Y_))-Y), fabs(((_PML_PX_Y_))+Y - (_DimY)+1)) / ((_PML_PX_Y_)), (pml_n));
			kappaY[i] = 1.0f + (pml_kappa_max - 1.0f) * pow((fmin(fabs(((_PML_PX_Y_))-Y), fabs(((_PML_PX_Y_))+Y - (_DimY)+1)) - 1.0f)/ ((_PML_PX_Y_)), (pml_n));
			b_Y[i] = expf(-alpha_dt_div_eps0[i] - sigmaY_dt_div_eps0[i] / kappaY[i]);
			C_Y[i] = sigmaY_dt_div_eps0[i] / (sigmaY_dt_div_eps0[i] * kappaY[i] + alpha_dt_div_eps0[i] * kappaY[i] * kappaY[i]) * (b_Y[i] - 1.0f);
		}
		if ((Z + 1 <= ((_PML_PX_Z_)) || (_DimZ)-((_PML_PX_Z_)) <= Z)) {
			//if ((Z + 1 <= ((_PML_PX_Z_)) || (_DimZ)-((_PML_PX_Z_)) <= Z) && ((_PML_PX_X_)) < X + 1 && X < (_DimX)-((_PML_PX_X_)) && ((_PML_PX_Y_)) < Y + 1 && Y < (_DimY)-((_PML_PX_Y_)) ) {
			mask[i] |= (1 << 1); // 1st bit : PML
			sigmaZ_dt_div_eps0[i] = pml_sigma_z_dt_div_eps0_max * pow(fmin(fabs(((_PML_PX_Z_))-Z), fabs(((_PML_PX_Z_))+Z - (_DimZ)+1)) / ((_PML_PX_Z_)), (pml_n));
			kappaZ[i] = 1.0f + (pml_kappa_max - 1.0f) * pow((fmin(fabs(((_PML_PX_Z_))-Z), fabs(((_PML_PX_Z_))+Z - (_DimZ)+1)) - 1.0f)/ ((_PML_PX_Z_)), (pml_n));
			b_Z[i] = expf(-alpha_dt_div_eps0[i] - sigmaZ_dt_div_eps0[i] / kappaZ[i]);
			C_Z[i] = sigmaZ_dt_div_eps0[i] / (sigmaZ_dt_div_eps0[i] * kappaZ[i] + alpha_dt_div_eps0[i] * kappaZ[i] * kappaZ[i]) * (b_Z[i] - 1.0f);
		}
		if (((mask[i] & (1 << 0)) >> 0) == 1) { // PML : exclude padding area
			mask[i] &= ~(1 << 1);
		}
		//NOT USED
		//if (_SURF_INDEX_XYZ(X, Y, Z) != -1) {
		//	mask[i] |= (1 << 3); // 3rd bit : RFT / NTFF
		//}

		//4th~7th bit : metal
		if ((X - _DimX / 2)*(X - _DimX / 2) + (Y - _DimY / 2)* (Y - _DimY / 2) + (Z - _DimZ / 2)*(Z - _DimZ / 2) < 10 * 10)
		{
			//mask[i] |= (1 << 4);
		}

	}
	printf("init done!\n");
	return 0;
}

void Dielectric_HE(void)
{
	//FIXME : syncPadding
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

#define _syncX_(FIELD) \
FIELD[_INDEX_THREAD(X - 1, Y, Z, _blockDimX - 1, yy, zz)] = FIELD[_INDEX_THREAD(X, Y, Z, 1, yy, zz)]; \
FIELD[_INDEX_THREAD(X, Y, Z, 0, yy, zz)] = FIELD[_INDEX_THREAD(X - 1, Y, Z, _blockDimX - 2, yy, zz)];
#define _syncY_(FIELD) \
FIELD[_INDEX_THREAD(X, Y - 1, Z, xx, _blockDimY - 1, zz)] = FIELD[_INDEX_THREAD(X, Y, Z, xx, 1, zz)]; \
FIELD[_INDEX_THREAD(X, Y, Z, xx, 0, zz)] = FIELD[_INDEX_THREAD(X, Y - 1, Z, xx, _blockDimY - 2, zz)]; 
#define _syncZ_(FIELD) \
FIELD[_INDEX_THREAD(X, Y, Z - 1, xx, yy, _blockDimZ - 1)] = FIELD[_INDEX_THREAD(X, Y, Z, xx, yy, 1)]; \
FIELD[_INDEX_THREAD(X, Y, Z, xx, yy, 0)] = FIELD[_INDEX_THREAD(X, Y, Z - 1, xx, yy, _blockDimZ - 2)];
#define _syncXY_(FIELD) \
FIELD[_INDEX_THREAD(X - 1, Y - 1, Z, _blockDimX - 1, _blockDimY - 1, zz)] = FIELD[_INDEX_THREAD(X, Y, Z, 1, 1, zz)]; \
FIELD[_INDEX_THREAD(X, Y, Z, 0, 0, zz)] = FIELD[_INDEX_THREAD(X - 1, Y - 1, Z, _blockDimX - 2, _blockDimY - 2, zz)]; \
FIELD[_INDEX_THREAD(X, Y - 1, Z, 0, _blockDimY - 1, zz)] = FIELD[_INDEX_THREAD(X - 1, Y, Z, _blockDimX - 2, 1, zz)]; \
FIELD[_INDEX_THREAD(X - 1, Y, Z, _blockDimX - 1, 0, zz)] = FIELD[_INDEX_THREAD(X, Y - 1, Z, 1, _blockDimY - 2, zz)]; 
#define _syncYZ_(FIELD) \
FIELD[_INDEX_THREAD(X, Y - 1, Z - 1, xx, _blockDimY - 1, _blockDimZ - 1)] = FIELD[_INDEX_THREAD(X, Y, Z, xx, 1, 1)]; \
FIELD[_INDEX_THREAD(X, Y, Z, xx, 0, 0)] = FIELD[_INDEX_THREAD(X, Y - 1, Z - 1, xx, _blockDimY - 2, _blockDimZ - 2)]; \
FIELD[_INDEX_THREAD(X, Y, Z - 1, xx, 0, _blockDimZ - 1)] = FIELD[_INDEX_THREAD(X, Y - 1, Z, xx, _blockDimY - 2, 1)]; \
FIELD[_INDEX_THREAD(X, Y - 1, Z, xx, _blockDimY - 1, 0)] = FIELD[_INDEX_THREAD(X, Y, Z - 1, xx, 1, _blockDimZ - 2)]; 
#define _syncZX_(FIELD) \
FIELD[_INDEX_THREAD(X - 1, Y, Z - 1, _blockDimX - 1, yy, _blockDimZ - 1)] = FIELD[_INDEX_THREAD(X, Y, Z, 1, yy, 1)]; \
FIELD[_INDEX_THREAD(X, Y, Z, 0, yy, 0)] = FIELD[_INDEX_THREAD(X - 1, Y, Z - 1, _blockDimX - 2, yy, _blockDimZ - 2)]; \
FIELD[_INDEX_THREAD(X - 1, Y, Z, _blockDimX - 1, yy, 0)] = FIELD[_INDEX_THREAD(X, Y, Z - 1, 1, yy, _blockDimZ - 2)]; \
FIELD[_INDEX_THREAD(X, Y, Z - 1, 0, yy, _blockDimZ - 1)] = FIELD[_INDEX_THREAD(X - 1, Y, Z, _blockDimX - 2, yy, 1)];
#define _syncXall _syncX_(eps0_c_Ex) _syncX_(eps0_c_Ey) _syncX_(eps0_c_Ez) _syncX_(Hx)  _syncX_(Hy) _syncX_(Hz) \
_syncX_(eps0_c_Pdx) _syncX_(eps0_c_Pdy) _syncX_(eps0_c_Pdz)  \
_syncX_(eps0_c_Pcp1x) _syncX_(eps0_c_Pcp1y) _syncX_(eps0_c_Pcp1z)  \
_syncX_(eps0_c_Pcp2x) _syncX_(eps0_c_Pcp2y) _syncX_(eps0_c_Pcp2z) 
//\
//_syncX_(eps0_c_Rx) _syncX_(eps0_c_Ry) _syncX_(eps0_c_Rz)  
#define _syncYall _syncY_(eps0_c_Ex) _syncY_(eps0_c_Ey) _syncY_(eps0_c_Ez) _syncY_(Hx)  _syncY_(Hy) _syncY_(Hz) \
_syncY_(eps0_c_Pdx) _syncY_(eps0_c_Pdy) _syncY_(eps0_c_Pdz)  \
_syncY_(eps0_c_Pcp1x) _syncY_(eps0_c_Pcp1y) _syncY_(eps0_c_Pcp1z)  \
_syncY_(eps0_c_Pcp2x) _syncY_(eps0_c_Pcp2y) _syncY_(eps0_c_Pcp2z)  
//\
//_syncY_(eps0_c_Rx) _syncY_(eps0_c_Ry) _syncY_(eps0_c_Rz)  
#define _syncZall _syncZ_(eps0_c_Ex) _syncZ_(eps0_c_Ey) _syncZ_(eps0_c_Ez) _syncZ_(Hx)  _syncZ_(Hy) _syncZ_(Hz) \
_syncZ_(eps0_c_Pdx) _syncZ_(eps0_c_Pdy) _syncZ_(eps0_c_Pdz)  \
_syncZ_(eps0_c_Pcp1x) _syncZ_(eps0_c_Pcp1y) _syncZ_(eps0_c_Pcp1z)  \
_syncZ_(eps0_c_Pcp2x) _syncZ_(eps0_c_Pcp2y) _syncZ_(eps0_c_Pcp2z)  
//\
//_syncZ_(eps0_c_Rx) _syncZ_(eps0_c_Ry) _syncZ_(eps0_c_Rz)  
#define _syncXYall _syncXY_(eps0_c_Ex) _syncXY_(eps0_c_Ey) _syncXY_(eps0_c_Ez) _syncXY_(Hx)  _syncXY_(Hy) _syncXY_(Hz) \
_syncXY_(eps0_c_Pdx) _syncXY_(eps0_c_Pdy) _syncXY_(eps0_c_Pdz)  \
_syncXY_(eps0_c_Pcp1x) _syncXY_(eps0_c_Pcp1y) _syncXY_(eps0_c_Pcp1z)  \
_syncXY_(eps0_c_Pcp2x) _syncXY_(eps0_c_Pcp2y) _syncXY_(eps0_c_Pcp2z) 
// \
//_syncXY_(eps0_c_Rx) _syncXY_(eps0_c_Ry) _syncXY_(eps0_c_Rz)  
#define _syncYZall _syncYZ_(eps0_c_Ex) _syncYZ_(eps0_c_Ey) _syncYZ_(eps0_c_Ez) _syncYZ_(Hx)  _syncYZ_(Hy) _syncYZ_(Hz) \
_syncYZ_(eps0_c_Pdx) _syncYZ_(eps0_c_Pdy) _syncYZ_(eps0_c_Pdz)  \
_syncYZ_(eps0_c_Pcp1x) _syncYZ_(eps0_c_Pcp1y) _syncYZ_(eps0_c_Pcp1z)  \
_syncYZ_(eps0_c_Pcp2x) _syncYZ_(eps0_c_Pcp2y) _syncYZ_(eps0_c_Pcp2z) 
// \
//_syncYZ_(eps0_c_Rx) _syncYZ_(eps0_c_Ry) _syncYZ_(eps0_c_Rz)  
#define _syncZXall _syncZX_(eps0_c_Ex) _syncZX_(eps0_c_Ey) _syncZX_(eps0_c_Ez) _syncZX_(Hx)  _syncZX_(Hy) _syncZX_(Hz) \
_syncZX_(eps0_c_Pdx) _syncZX_(eps0_c_Pdy) _syncZX_(eps0_c_Pdz)  \
_syncZX_(eps0_c_Pcp1x) _syncZX_(eps0_c_Pcp1y) _syncZX_(eps0_c_Pcp1z)  \
_syncZX_(eps0_c_Pcp2x) _syncZX_(eps0_c_Pcp2y) _syncZX_(eps0_c_Pcp2z) 
// \
//_syncZX_(eps0_c_Rx) _syncZX_(eps0_c_Ry) _syncZX_(eps0_c_Rz)  

//이거 안넣으면 periodic boundary 안됨
//#define _syncBoundaryX(FIELD) \
//FIELD[_INDEX_THREAD(_gridDimX-1, Y, Z, _blockDimX - 1, yy, zz)] = FIELD[_INDEX_THREAD(0, Y, Z, 1, yy, zz)]; \
//FIELD[_INDEX_THREAD(0, Y, Z, 0, yy, zz)] = FIELD[_INDEX_THREAD(_gridDimX-1, Y, Z, _blockDimX - 2, yy, zz)]; \
//#define _syncBoundaryall _syncBoundary(eps0_c_Ex) _syncBoundary(eps0_c_Ey) _syncBoundary(eps0_c_Ez)  _syncBoundary(Hx)  _syncBoundary(Hy) _syncBoundary(Hz) 
#define STR(x)   #x
#define SHOW_DEFINE(x) printf("%s=%s\n", #x, STR(x))

void syncPadding(void) {
	for (int X = 0; X < _gridDimX; X++)
		for (int Y = 0; Y < _gridDimY; Y++)
			for (int Z = 0; Z < _gridDimZ; Z++) {
				if (X>0)
					for (int yy = 0; yy<_blockDimY; yy++)
						for (int zz = 0; zz < _blockDimZ; zz++) { _syncXall }
				if (Y>0)
					for (int zz = 0; zz<_blockDimZ; zz++)
						for (int xx = 0; xx < _blockDimX; xx++) { _syncYall }
				if (Z>0)
					for (int xx = 0; xx<_blockDimX; xx++)
						for (int yy = 0; yy < _blockDimY; yy++) { _syncZall }
				if (X>0 && Y>0)
					for (int zz = 0; zz < _blockDimZ; zz++) {_syncXYall }
				if (Y>0 && Z>0)
					for (int xx = 0; xx < _blockDimX; xx++) {_syncYZall }
				if (Z>0 && X>0)
					for (int yy = 0; yy < _blockDimY; yy++) {_syncZXall }
			}
}


#define TEMP eps0_c_Ex
int snapshot(void)
{
	printf("snapshot : X=50\n");
	int X = 50;
	FILE *f = fopen("test.txt", "w");
	for (int Z = 0; Z < _DimZ; Z++) {
		if (Z%_blockDimZ == 0) { for (int Y = 0; Y < _DimY + _gridDimY - 1 ; Y++) { fprintf(f,"%+04.3e \t ", TEMP[_INDEX_XYZ(X, Y, -1)]);  } 	fprintf(f,"\n");}
		for (int Y = 0; Y < _DimY; Y++) {
			if (Y%_blockDimY == 0) { fprintf(f,"%04.3f \t", TEMP[_INDEX_XYZ(-1, Y, Z)]);}
			fprintf(f,"%+04.3e\t", TEMP[_INDEX_XYZ(-1, Y, Z)]);
			if (Y%_blockDimY == _blockDimY-1) { fprintf(f,"%+04.3f\t ", TEMP[_INDEX_XYZ(_blockDimX, Y, Z)]); }
		}
		fprintf(f,"\n");
		if (Z%_blockDimZ == _blockDimZ -1 ){ for (int Y = 0; Y < _DimY; Y++) 	{	fprintf(f,"%+04.3f\t ", eps0_c_Ex[_INDEX_XYZ(X, Y, _blockDimZ)]);	}fprintf(f,"\n");}
	}
	fclose(f);


	printf("snapshot : Z=70\n");
	int Z = 70;
	const char* filename = "test.png";
	unsigned width = _DimX, height = _DimY;
	unsigned char* image = malloc(width * height * 4);
	unsigned y, x;
	for (x = 0; x < height; x++)
		for (y = 0; y < width; y++)
		{
			int surf_x, surf_y, surf_z;
			int surf_index = _SURF_INDEX_XYZ(x, y, Z);
			_SET_SURF_XYZ_INDEX(surf_index);
			int offset = _INDEX_XYZ(x, y, Z);
			int value =0;

			//if (surf_index !=-1) value = (FT_eps0cE[_SURF_INDEX_XYZ(X, y, z)][1][0][0]* FT_eps0cE[_SURF_INDEX_XYZ(X, y, z)][1][0][0]+ FT_eps0cE[_SURF_INDEX_XYZ(X, y, z)][1][0][1]* FT_eps0cE[_SURF_INDEX_XYZ(X, y, z)][1][0][1]) * 255.0f * 5000.0f * 5000.0f;
			if (surf_index != -1) value = 255.0f * 200.0f * 200.0f*(0
				+ (FT_eps0cE[surf_index][0][0][0]) * (FT_eps0cE[surf_index][0][0][0])
				+ (FT_eps0cE[surf_index][0][0][1]) * (FT_eps0cE[surf_index][0][0][1])
				//+ (FT_eps0cE[surf_index][0][1][0]) * (FT_eps0cE[surf_index][0][1][0])
				//+ (FT_eps0cE[surf_index][0][1][1]) * (FT_eps0cE[surf_index][0][1][1])
				//+ (FT_eps0cE[surf_index][0][2][0]) * (FT_eps0cE[surf_index][0][2][0])
				//+ (FT_eps0cE[surf_index][0][2][1]) * (FT_eps0cE[surf_index][0][2][1])
				);
			//int value = (((mask[offset] & (0b1111 << 4)) >> 4)) *50.0f ;
			value = value > 255 ? 255 : value;
			value = value < -255 ? -255 : value;
			int val2 = (eps0_c_Ex[offset]) * 255.0f * 50.0f;
			val2 = val2 > 255 ? 255 : val2;
			val2 = val2 < -255 ? -255 : val2;
			image[4 * width * y + 4 * x + 0] = (unsigned char)(value>0? value : 0);
			image[4 * width * y + 4 * x + 1] = (unsigned char)(val2<0 ? -val2 : 0);
			image[4 * width * y + 4 * x + 2] = (unsigned char)(val2>0 ? val2 : 0);
			image[4 * width * y + 4 * x + 3] = 255;
		}
	unsigned error = lodepng_encode32_file(filename, image, width, height);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

	free(image);

	return 0;
}