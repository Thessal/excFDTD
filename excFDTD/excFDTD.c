//VS2015
//ipsxe2016 Cluster Edition

#include "stdio.h"
#include "stdlib.h"
#include "immintrin.h"

#define _DimX (10)
#define _DimY (20)
#define _DimZ (30)

#define _blockDimX (8)
#define _blockDimY (8)
#define _blockDimZ (8)
#define _blockDimXYZ (_blockDimX * _blockDimY * _blockDimZ)

#define _sizeofFloat (32)

//memory overhead : 1px padding per block
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
float stability_factor_inv = 0.5f; 
__m256* mEx = (__m256*)eps0_c_Ex;
__m256* mEy = (__m256*)eps0_c_Ey;
__m256* mEz = (__m256*)eps0_c_Ez;
__m256* mHx = (__m256*)Hx;
__m256* mHy = (__m256*)Hy;
__m256* mHz = (__m256*)Hz;
__m256* meps = (__m256*)eps_r_inv;

int init(void);
void DielectricE(voie);
void DrudeE(void);
void DrudeH(void);
int snapshot(void);
int main(int argc, char* argv[])
{
	printf("total area : block(%d,%d,%d) * grid(%d,%d,%d) = %d(%d,%d,%d)\n",_blockDimX,_blockDimY,_blockDimZ,_gridDimX,_gridDimY,_gridDimZ, _threadPerGrid, _blockDimX*_gridDimX, _blockDimY*_gridDimY, _blockDimZ*_gridDimZ);
	printf("thread space (padding overhead) : %d(%d,%d,%d) \n", (_blockDimX - 2)*_gridDimX* (_blockDimY - 2)*_gridDimY* (_blockDimZ - 2)*_gridDimZ,(_blockDimX-2)*_gridDimX , (_blockDimY - 2)*_gridDimY, (_blockDimZ - 2)*_gridDimZ);
	printf("xyz space (block overhead) : %d(%d,%d,%d) \n", _DimX*_DimY*_DimZ, _DimX, _DimY, _DimZ);
	printf("unit : sizeof(float)\n ");

	init();

	DielectricE();

	snapshot();
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
					_mm256_broadcast_ss(&stability_factor_inv)),
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

		Hy[i] = (float)(X * 10000 + Y * 100 + Z);
		Hz[i] = (float)(X * 10000 + Y * 100 + Z);
		eps_r_inv[i] = 1.0f;


		if (X < 0 || _DimX-1 < X || Y < 0 || _DimY-1 < Y || Z < 0 || _DimZ-1 < Z) { continue; }
		else if (0 <= X) { eps0_c_Ex[i] = 8.2f; }
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
			//printf("%1.1f \t", update[_INDEX_XYZ(X, Y, Z) ]);
			printf("%06.0f \t", eps0_c_Ex[_INDEX_XYZ(X, Y, Z)]);
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
			//printf("%05.0f \t", Ex[_INDEX_XYZ(X, Y, Z)]);
			printf("%06.0f \t", eps0_c_Ex[_INDEX_XYZ(X, Y, Z)]);
		}
		printf("\n");
	}
	return 0;
}