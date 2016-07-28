//ADE DCP FDTD AVX ver
//Jongkook Choi
//VS2015, ipsxe2016 Cluster Edition
//reference : K.P.Prokopidis, 2013


#define _DimX (100)
#define _DimY (100)
#define _DimZ (100)

//eq35
int pml_thick_x_px = 8;
int pml_thick_y_px = 8;
int pml_thick_z_px = 8;
int pml_n = 3;
float pml_R = 10e-4;
float pml_kappa_max = 8.0f;

#define _S_factor (2.0f)
#define _dx (2e-9)

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

//copy by value
#include <string.h>
#define SWAP(x,y) do \
{ unsigned __int64 swap_temp[sizeof(x) == sizeof(y) ? (signed)sizeof(x) : -1]; \
memcpy(swap_temp, &y, sizeof(x)); \
memcpy(&y, &x, sizeof(x)); \
memcpy(&x, swap_temp, sizeof(x)); \
} while (0)



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
	//eps0_c_Ex[_INDEX_XYZ(50, 50, 50)] = 10.0f;//cos((float)i / 5.0f);
	for (int i = 0; i < 250; i++) {
		//eps0_c_Ex[_INDEX_XYZ(50, 50, 50)] = sin((float)i / 5.0f)/(1+(float)i/100);
		eps0_c_Ex[_INDEX_XYZ(50, 50, 50)] += sin((float)i / 2.5f);
		printf("%f\n", cos((float)i / 5.0f));
		Dielectric_HE_C();
	}
	printf("time : %f\n", (double)(clock() - start) / CLK_TCK);

	snapshot();

	return 0;
}

void Dielectric_HE_C(void)
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
			eps0_c_Ex[offset] += (Hy[offset - _offsetZ] - Hy[offset] + Hz[offset - _offsetZ] - Hz[offset - _offsetY - _offsetZ]) * eps_r_inv[offset] * _cdt_div_dx;
			eps0_c_Ey[offset] += (Hz[offset - _offsetZ] - Hz[offset + _offsetX - _offsetZ] + Hx[offset] - Hx[offset - _offsetZ]) * eps_r_inv[offset] * _cdt_div_dx;
			eps0_c_Ez[offset] += (Hx[offset - _offsetX - _offsetY] - Hx[offset - _offsetX] + Hy[offset] - Hy[offset - _offsetX]) * eps_r_inv[offset] * _cdt_div_dx;
			continue;
		}
		//PML
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

int init(void)
{
	//PML constants
	float pml_eps0 = _eps0_;
	float pml_sigma_x_dt_div_eps0_max = -((float)pml_n + 1.0f)*logf(pml_R) * 0.5f / pml_thick_x_px / _S_factor; //eq37
	float pml_sigma_y_dt_div_eps0_max = -((float)pml_n + 1.0f)*logf(pml_R) * 0.5f / pml_thick_y_px / _S_factor;
	float pml_sigma_z_dt_div_eps0_max = -((float)pml_n + 1.0f)*logf(pml_R) * 0.5f / pml_thick_z_px / _S_factor;
	printf("_eps0_ = %e\n", _eps0_);
	printf("pml_eps0 = %e\n", pml_eps0);
	printf("pml_thick_x_px = %d\n", pml_thick_x_px);
	printf("pml_n = %d\n", pml_n);
	printf("pml_R = %e\n", pml_R);
	printf("pml_kappa_max = %e\n", pml_kappa_max);
	printf("pml_sigma_x_dt_div_eps0_max = %e\n", pml_sigma_x_dt_div_eps0_max);


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

		alpha_dt_div_eps0[i] = 0.1f; // FIXME 

		//FIXME : check PML area
		if ((X + 1 <= (pml_thick_x_px) || (_DimX)-(pml_thick_x_px) <= X)) {
			//if ((X + 1 <= (pml_thick_x_px) || (_DimX)-(pml_thick_x_px) <= X) && (pml_thick_y_px) < Y+1 && Y < (_DimY)-(pml_thick_y_px) && (pml_thick_z_px) < Z + 1 && Z < (_DimZ)-(pml_thick_z_px) ) {
			mask[i] |= (1 << 1); // 1st bit : PML
			sigmaX_dt_div_eps0[i] = pml_sigma_x_dt_div_eps0_max * pow(fmin(fabs((pml_thick_x_px)-X), fabs((pml_thick_x_px)+X - (_DimX)+1)) / (pml_thick_x_px), (pml_n));
			kappaX[i] = 1.0f + (pml_kappa_max - 1.0f) * pow((fmin(fabs((pml_thick_x_px)-X), fabs((pml_thick_x_px)+X - (_DimX)+1)) - 1.0f)/ (pml_thick_x_px), (pml_n)); //FIXME : 0.5f? -1.0f? check 
			b_X[i] = expf( -alpha_dt_div_eps0[i] - sigmaX_dt_div_eps0[i]/kappaX[i] ); //close to 0
			C_X[i] = sigmaX_dt_div_eps0[i] / (sigmaX_dt_div_eps0[i] * kappaX[i] + alpha_dt_div_eps0[i] * kappaX[i] * kappaX[i] ) * (b_X[i] - 1.0f); //close to 1
		}
		if ((Y + 1 <= (pml_thick_y_px) || (_DimY)-(pml_thick_y_px) <= Y)) {
			//if ((Y + 1 <= (pml_thick_y_px) || (_DimY)-(pml_thick_y_px) <= Y) && (pml_thick_z_px) < Z + 1 && Z < (_DimY)-(pml_thick_z_px) && (pml_thick_x_px) < X + 1 && X < (_DimX)-(pml_thick_x_px) ) {
			mask[i] |= (1 << 1); // 1st bit : PML
			sigmaY_dt_div_eps0[i] = pml_sigma_y_dt_div_eps0_max * pow(fmin(fabs((pml_thick_y_px)-Y), fabs((pml_thick_y_px)+Y - (_DimY)+1)) / (pml_thick_y_px), (pml_n));
			kappaY[i] = 1.0f + (pml_kappa_max - 1.0f) * pow((fmin(fabs((pml_thick_y_px)-Y), fabs((pml_thick_y_px)+Y - (_DimY)+1)) - 1.0f)/ (pml_thick_y_px), (pml_n));
			b_Y[i] = expf(-alpha_dt_div_eps0[i] - sigmaY_dt_div_eps0[i] / kappaY[i]);
			C_Y[i] = sigmaY_dt_div_eps0[i] / (sigmaY_dt_div_eps0[i] * kappaY[i] + alpha_dt_div_eps0[i] * kappaY[i] * kappaY[i]) * (b_Y[i] - 1.0f);
		}
		if ((Z + 1 <= (pml_thick_z_px) || (_DimZ)-(pml_thick_z_px) <= Z)) {
			//if ((Z + 1 <= (pml_thick_z_px) || (_DimZ)-(pml_thick_z_px) <= Z) && (pml_thick_x_px) < X + 1 && X < (_DimX)-(pml_thick_x_px) && (pml_thick_y_px) < Y + 1 && Y < (_DimY)-(pml_thick_y_px) ) {
			mask[i] |= (1 << 1); // 1st bit : PML
			sigmaZ_dt_div_eps0[i] = pml_sigma_z_dt_div_eps0_max * pow(fmin(fabs((pml_thick_z_px)-Z), fabs((pml_thick_z_px)+Z - (_DimZ)+1)) / (pml_thick_z_px), (pml_n));
			kappaZ[i] = 1.0f + (pml_kappa_max - 1.0f) * pow((fmin(fabs((pml_thick_z_px)-Z), fabs((pml_thick_z_px)+Z - (_DimZ)+1)) - 1.0f)/ (pml_thick_z_px), (pml_n));
			b_Z[i] = expf(-alpha_dt_div_eps0[i] - sigmaZ_dt_div_eps0[i] / kappaZ[i]);
			C_Z[i] = sigmaZ_dt_div_eps0[i] / (sigmaZ_dt_div_eps0[i] * kappaZ[i] + alpha_dt_div_eps0[i] * kappaZ[i] * kappaZ[i]) * (b_Z[i] - 1.0f);
		}
		if (((mask[i] & (1 << 0)) >> 0) == 1) { // PML : exclude padding area
			mask[i] &= ~(1 << 1);
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

void DCP_HE_C(void)
{
	// before --> after
	// temp --> E
	// E --> E_old
	// pcp_old <--> pcp
//	printf("%e\n", (_eps0_ * _eps_inf + 0.5f * _sigma_ * _dt_ - _d2 + _C3p));

	syncPadding();
	for (unsigned __int64 offset = 0; offset < _threadPerGrid; offset += 1) {
		//// H
		Hx[offset] -= (eps0_c_Ey[offset] - eps0_c_Ey[offset + _offsetZ] + eps0_c_Ez[offset + _offsetX] - eps0_c_Ez[offset + _offsetX]) * _cdt_div_dx;
		Hy[offset] -= (eps0_c_Ez[offset] - eps0_c_Ez[offset + _offsetX] + eps0_c_Ex[offset + _offsetZ] - eps0_c_Ex[offset]) *_cdt_div_dx;
		Hz[offset] -= (eps0_c_Ex[offset + _offsetZ] - eps0_c_Ex[offset + _offsetY + _offsetZ] + eps0_c_Ey[offset + _offsetZ] - eps0_c_Ey[offset - _offsetX + _offsetZ]) * _cdt_div_dx;

		eps0_c_Ex[offset] = (Hy[offset - _offsetZ] - Hy[offset] + Hz[offset - _offsetZ] - Hz[offset - _offsetY - _offsetZ]) * eps_r_inv[offset] * _cdt_div_dx; //eq30term1
		eps0_c_Ey[offset] = (Hz[offset - _offsetZ] - Hz[offset + _offsetX - _offsetZ] + Hx[offset] - Hx[offset - _offsetZ]) * eps_r_inv[offset] * _cdt_div_dx;
		eps0_c_Ez[offset] = (Hx[offset - _offsetX - _offsetY] - Hx[offset - _offsetX] + Hy[offset] - Hy[offset - _offsetX]) * eps_r_inv[offset] * _cdt_div_dx;
	}

	syncPadding();
	for (unsigned __int64 offset = 0; offset < _threadPerGrid; offset += 1) {
		//// E
		tempx[offset] = (Hy[offset - _offsetZ] - Hy[offset] + Hz[offset - _offsetZ] - Hz[offset - _offsetY - _offsetZ]) * eps_r_inv[offset] * _cdt_div_dx; //eq30term1
		tempy[offset] = (Hz[offset - _offsetZ] - Hz[offset + _offsetX - _offsetZ] + Hx[offset] - Hx[offset - _offsetZ]) * eps_r_inv[offset] * _cdt_div_dx;
		tempz[offset] = (Hx[offset - _offsetX - _offsetY] - Hx[offset - _offsetX] + Hy[offset] - Hy[offset - _offsetX]) * eps_r_inv[offset] * _cdt_div_dx;
		tempx[offset] += eps0_c_Ex[offset] * (_eps0_ * _eps_inf - 0.5f * _sigma_ * _dt_ + _d2 - _C4p); //eq30term2
		tempy[offset] += eps0_c_Ey[offset] * (_eps0_ * _eps_inf - 0.5f * _sigma_ * _dt_ + _d2 - _C4p);
		tempz[offset] += eps0_c_Ez[offset] * (_eps0_ * _eps_inf - 0.5f * _sigma_ * _dt_ + _d2 - _C4p);
		tempx[offset] -= eps0_c_Ex_old[offset] * _C5p; //eq30term3
		tempy[offset] -= eps0_c_Ey_old[offset] * _C5p;
		tempz[offset] -= eps0_c_Ez_old[offset] * _C5p;
		tempx[offset] -= eps0_c_Pdx[offset] * (_d1 - 1.0f); //eq30term4
		tempy[offset] -= eps0_c_Pdy[offset] * (_d1 - 1.0f);
		tempz[offset] -= eps0_c_Pdz[offset] * (_d1 - 1.0f);
		tempx[offset] -= eps0_c_Pcp1x[offset] * (_C11 - 1.0f); //eq30term5
		tempy[offset] -= eps0_c_Pcp1y[offset] * (_C11 - 1.0f);
		tempz[offset] -= eps0_c_Pcp1z[offset] * (_C11 - 1.0f);
		tempx[offset] -= eps0_c_Pcp2x[offset] * (_C12 - 1.0f);
		tempy[offset] -= eps0_c_Pcp2y[offset] * (_C12 - 1.0f);
		tempz[offset] -= eps0_c_Pcp2z[offset] * (_C12 - 1.0f);
		tempx[offset] -= eps0_c_Pcp1x_old[offset] * _C21; //eq30term6
		tempy[offset] -= eps0_c_Pcp1y_old[offset] * _C21;
		tempz[offset] -= eps0_c_Pcp1z_old[offset] * _C21;
		tempx[offset] -= eps0_c_Pcp2x_old[offset] * _C22;
		tempy[offset] -= eps0_c_Pcp2y_old[offset] * _C22;
		tempz[offset] -= eps0_c_Pcp2z_old[offset] * _C22;
		tempx[offset] /= (_eps0_ * _eps_inf + 0.5f * _sigma_ * _dt_ - _d2 + _C3p); //eq30term0
		tempy[offset] /= (_eps0_ * _eps_inf + 0.5f * _sigma_ * _dt_ - _d2 + _C3p);
		tempz[offset] /= (_eps0_ * _eps_inf + 0.5f * _sigma_ * _dt_ - _d2 + _C3p);

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
	printf("X=50\n");
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


	const char* filename = "test.png";
	unsigned width = _DimX, height = _DimY;
	unsigned char* image = malloc(width * height * 4);
	unsigned y, z;
	for (z = 0; z < height; z++)
		for (y = 0; y < width; y++)
		{
			int offset = _INDEX_XYZ(X, y, z);
			//int value = (psiXY_dx[offset] - psiXZ_dx[offset]) * 255.0f *50.0f *1000.0f;// *_c0;
			int value = (0.0f) *50.0f ;// *_c0;
			value = value > 255 ? 255 : value;
			value = value < -255 ? -255 : value;
			int val2 = (eps0_c_Ex[offset]) * 255.0f * 50.0f;
			//int val2 = (exp(-alpha_dt_div_eps0[offset] - sigmaY_dt_div_eps0[offset] / kappaY[offset] ) )* 255.0f * 1.0f;
			val2 = val2 > 255 ? 255 : val2;
			val2 = val2 < -255 ? -255 : val2;
			image[4 * width * z + 4 * y + 0] = (unsigned char)(value>0? value : 0);
			image[4 * width * z + 4 * y + 0] += (unsigned char)(val2>0 ? val2 : 0);
			image[4 * width * z + 4 * y + 1] = (unsigned char)(val2<0 ? -val2 : 0);
			image[4 * width * z + 4 * y + 2] = (unsigned char)(value<0 ? -value : 0);
			image[4 * width * z + 4 * y + 3] = 255;
		}
	unsigned error = lodepng_encode32_file(filename, image, width, height);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

	free(image);

	return 0;
}