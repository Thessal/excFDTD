//VS2015
//ipsxe2016 Cluster Edition
//Konstantinos, 2013


#define _DimX (10)
#define _DimY (20)
#define _DimZ (30)

#define _S_factor (2.0f)
#define _c0 (3e8)
#define _dx (1e-6)
#define _dt (_dx / _c0 / _S_factor)
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
#define _b01 (_Omega1_L * _Omega1_L + _Gamma1_L * _Gamma1_L)
#define _b02 (_Omega2_L * _Omega2_L + _Gamma2_L * _Gamma2_L)
#define _b11 (2 * _Gamma1_L)
#define _b12 (2 * _Gamma2_L)
#define _b21 (1.0f)
#define _b22 (1.0f)

#define _C1_ ( (_b21 / _dt / _dt) + (_b11 / 2 / _dt) + (_b01 / 4) )
#define _C2_ ( (_b22 / _dt / _dt) + (_b12 / 2 / _dt) + (_b02 / 4) )
#define _C11 ( ( ( 2*_b21/_dt/_dt ) - (_b01*0.5f) ) / _C1_ )
#define _C12 ( ( ( 2*_b22/_dt/_dt ) - (_b02*0.5f) ) / _C2_ )
#define _C21 ( ( ( _b11 * 0.5f / _dt ) - ( _b21 / _dt / _dt ) - ( _b01 / 4 ) ) / _C1_ )
#define _C22 ( ( ( _b12 * 0.5f / _dt ) - ( _b22 / _dt / _dt ) - ( _b02 / 4 ) ) / _C2_ )
#define _C31 ( ( ( _a01 / 4.0f ) + ( _a11 / 2 / _dt ) ) / _C1_ )
#define _C32 ( ( ( _a02 / 4.0f ) + ( _a12 / 2 / _dt ) ) / _C2_ )
#define _C41 ( _a01 * 0.5f / _C1_ )
#define _C42 ( _a02 * 0.5f / _C2_ )
#define _C51 ( ( _a01 / 4.0f ) - ( _a11 * 0.5f / _dt ) ) / _C1_
#define _C52 ( ( _a02 / 4.0f ) - ( _a12 * 0.5f / _dt ) ) / _C2_

//#define _d1_
//#define _d11
//#define _d21
//#define _d31
//#define _d41

#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "immintrin.h"


#define _blockDimX (8)
#define _blockDimY (8)
#define _blockDimZ (8)
#define _blockDimXYZ (_blockDimX * _blockDimY * _blockDimZ)

#define _sizeofFloat (32)

//3D loop tiling with 1px padding, for L1 cache or cuda SHMEM utilization
//do we need 1px padding?
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

__declspec(align(32)) float eps0_c_Ex[_threadPerGrid] = { 0 };
__declspec(align(32)) float eps0_c_Ey[_threadPerGrid] = { 0 };
__declspec(align(32)) float eps0_c_Ez[_threadPerGrid] = { 0 };
__declspec(align(32)) float Hx[_threadPerGrid] = { 0 };
__declspec(align(32)) float Hy[_threadPerGrid] = { 0 };
__declspec(align(32)) float Hz[_threadPerGrid] = { 0 };
__declspec(align(32)) float eps_r_inv[_threadPerGrid] = { 0 };

__m256* mEx = (__m256*)eps0_c_Ex;
__m256* mEy = (__m256*)eps0_c_Ey;
__m256* mEz = (__m256*)eps0_c_Ez;
__m256* mHx = (__m256*)Hx;
__m256* mHy = (__m256*)Hy;
__m256* mHz = (__m256*)Hz;
__m256* meps = (__m256*)eps_r_inv;

int init(void);
void DielectricE(void);
void DielectricH(void);
void syncPadding(void);
void DrudeE(void);
void DrudeH(void);
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
	printf("dt      \t%01.2e\n", _dt);
	printf("dx      \t%01.2e\n", _dx);
	printf("\n");

	printf("check:\n");
	printf("a01/eps0\t%01.2e\n", _a01_div_eps0);
	printf("a11/eps0\t%01.2e\n", _a11_div_eps0);
	printf("b01     \t%01.2e\n", _b01);
	printf("b11     \t%01.2e\n", _b11);
	printf("b21     \t%01.2e\n", _b21);

//	init();
/*
	DielectricH();
	DielectricE();
	syncPadding();
*/

//	snapshot();
	return 0;
}

void DielectricE(void)
{
	for (unsigned __int64 offset = 0; offset < _threadPerGrid / 8; offset += 1) {
		_mm256_store_ps(eps0_c_Ex + offset * 8, // 1 * ymm(32byte) = 8 * float(4byte)
			_mm256_add_ps(
				_mm256_mul_ps(
					_mm256_mul_ps(
						_mm256_sub_ps(
							_mm256_add_ps(
								_mm256_sub_ps(
									mHy[offset - _offsetZ/8],
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

void DielectricH(void) 
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
									mEy[offset + _offsetZ/8]),
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
								_mm256_loadu_ps(eps0_c_Ez + offset + _offsetX) ),
							mEx[offset + _offsetZ ]),
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
							mEy[offset + _offsetZ / 8] ) ,
						_mm256_loadu_ps(eps0_c_Ey + offset - _offsetX + _offsetZ)),
					_mm256_broadcast_ss(&stability_factor_inv))
			)
		);
	}
}

void DCP_E(void)
{

}

#define _syncX(FIELD) \
FIELD[_INDEX_THREAD(X - 1, Y, Z, _blockDimX - 1, yy, zz)] = FIELD[_INDEX_THREAD(X, Y, Z, 1, yy, zz)]; \
FIELD[_INDEX_THREAD(X, Y, Z, 0, yy, zz)] = FIELD[_INDEX_THREAD(X - 1, Y, Z, _blockDimX - 2, yy, zz)];
#define _syncY(FIELD) \
FIELD[_INDEX_THREAD(X, Y - 1, Z, xx, _blockDimY - 1, zz)] = FIELD[_INDEX_THREAD(X, Y, Z, xx, 1, zz)]; \
FIELD[_INDEX_THREAD(X, Y, Z, xx, 0, zz)] = FIELD[_INDEX_THREAD(X, Y - 1, Z, xx, _blockDimY - 2, zz)];
#define _syncZ(FIELD) \
FIELD[_INDEX_THREAD(X, Y, Z - 1, xx, yy, _blockDimZ - 1)] = FIELD[_INDEX_THREAD(X, Y, Z, xx, yy, 1)]; \
FIELD[_INDEX_THREAD(X, Y, Z, xx, yy, 0)] = FIELD[_INDEX_THREAD(X, Y, Z - 1, xx, yy, _blockDimZ - 2)];
#define _syncXall _syncX(eps0_c_Ex) _syncX(eps0_c_Ey) _syncX(eps0_c_Ez)  _syncX(Hx)  _syncX(Hy) _syncX(Hz) 
#define _syncYall _syncY(eps0_c_Ex) _syncY(eps0_c_Ey) _syncY(eps0_c_Ez)  _syncY(Hx)  _syncY(Hy) _syncY(Hz) 
#define _syncZall _syncZ(eps0_c_Ex) _syncZ(eps0_c_Ey) _syncZ(eps0_c_Ez)  _syncZ(Hx)  _syncZ(Hy) _syncZ(Hz) 

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

//		Hy[i] = (float)(X * 10000 + Y * 100 + Z);
//		Hz[i] = (float)(X * 10000 + Y * 100 + Z);
		eps_r_inv[i] = 1.0f;


		if (X < 0 || _DimX-1 < X || Y < 0 || _DimY-1 < Y || Z < 0 || _DimZ-1 < Z) { continue; }
		else if (X==5 && Y ==2 && Z ==5) { eps0_c_Ex[i] = 1.0f; }
		//else if (0 <= X) { eps0_c_Ex[i] = 8.2f; }
		//else if (0 <= X) { 	eps0_c_Ex[i] = (float)(X*10000 + Y*100 + Z);}
		else { eps0_c_Ex[i] = 0.0f; }


	}
	return 0;
}
int snapshot(void)
{
	printf("Z=5\n");
	for (int Y = 0; Y < _DimY; Y++) {
		for (int X = 0; X < _DimX; X++) {
			int Z = 5;
			printf("%02.1f \t", eps0_c_Ex[_INDEX_XYZ(X, Y, Z) ]);
			//printf("%06.0f \t", eps0_c_Ex[_INDEX_XYZ(X, Y, Z)]);
			//printf("%d \t", _INDEX_BLOCK(X/(_blockDimX-2), Y/(_blockDimY-2), Z/(_blockDimZ-2)));
			//printf("%d \t",Y/_blockDimY);
			//printf("%d \t",_INDEX_XYZ(X,Y,Z));
		}
		printf("\n");
	}
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