#include "stdio.h"
#include "stdlib.h"

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
#define _offsetPadding (_offsetX+_offsetY+_offsetZ)

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

__declspec(align(32)) float Ex[_threadPerGrid] = { 999.99f };
__declspec(align(32)) float update[_threadPerGrid] = { 0 };
float *Ey, *Ez, *Hx, *Hy, *Hz;


int init(void);
int snapshot(void);
int main(int argc, char* argv[])
{
	printf("total area : block(%d,%d,%d) * grid(%d,%d,%d) = %d(%d,%d,%d)\n",_blockDimX,_blockDimY,_blockDimZ,_gridDimX,_gridDimY,_gridDimZ, _threadPerGrid, _blockDimX*_gridDimX, _blockDimY*_gridDimY, _blockDimZ*_gridDimZ);
	printf("thread space (padding overhead) : %d(%d,%d,%d) \n", (_blockDimX - 2)*_gridDimX* (_blockDimY - 2)*_gridDimY* (_blockDimZ - 2)*_gridDimZ,(_blockDimX-2)*_gridDimX , (_blockDimY - 2)*_gridDimY, (_blockDimZ - 2)*_gridDimZ);
	printf("xyz space (block overhead) : %d(%d,%d,%d) \n", _DimX*_DimY*_DimZ, _DimX, _DimY, _DimZ);
	printf("unit : sizeof(float)\n ");

	init();

	unsigned int loopsize = _threadPerGrid / 64; 	// 8 * ymm(32byte) = 64 * float(4byte)
	if ((_blockDimXYZ) % 64 != 0 ) { printf("Error : block size need to be a multiple of 64 float");return -1;}
	__asm
	{
		//pushad
		mov ecx, loopsize
		mov eax, Ex
		mov esi, 0
//		prefetchnta[eax + _sizeofFloat * _blockDimXYZ] //block cache
MEMLP:
		vmovaps ymm0, [Ex + esi]
		vmovaps ymm1, [Ex + esi + 32]
		vmovaps ymm2, [Ex + esi + 64]
		vmovaps ymm3, [Ex + esi + 96]
		vmovaps ymm4, [Ex + esi + 128]
		vmovaps ymm5, [Ex + esi + 160]
		vmovaps ymm6, [Ex + esi + 192]
		vmovaps ymm7, [Ex + esi + 224]

		vmovaps [update + esi], ymm0
		vmovaps [update + esi + 32], ymm1
		vmovaps [update + esi + 64], ymm2
		vmovaps [update + esi + 96], ymm3
		vmovaps [update + esi + 128], ymm4
		vmovaps [update + esi + 160], ymm5
		vmovaps [update + esi + 192], ymm6
		vmovaps [update + esi + 224], ymm7

		add esi, 256
		dec ecx
		jnz MEMLP
		sfence

		//popad
	}

	snapshot();
	return 0;
}

int Drude_eps0_c_Ex(void)
{
	int loopsize = 1; //1px padding

	__asm
	{
		pushad

		mov ecx, loopsize
		mov eax, Ex
		mov edx, update
		mov esi, 0

		prefetchnta[eax + _sizeofFloat * _blockDimXYZ]
		MEMLP:
		vmovups ymm0, [eax + esi]
			vmovups ymm1, [eax + esi + 32]
			vmovups ymm2, [eax + esi + 64]
			vmovups ymm3, [eax + esi + 96]
			vmovups ymm4, [eax + esi + 128]
			vmovups ymm5, [eax + esi + 160]
			vmovups ymm6, [eax + esi + 192]
			vmovups ymm7, [eax + esi + 224]

			vmovups[edx + esi], ymm0
			vmovups[edx + esi + 32], ymm1
			vmovups[edx + esi + 64], ymm2
			vmovups[edx + esi + 96], ymm3
			vmovups[edx + esi + 128], ymm4
			vmovups[edx + esi + 160], ymm5
			vmovups[edx + esi + 192], ymm6
			vmovups[edx + esi + 224], ymm7

			add esi, 256
			dec ecx
			jnz MEMLP
			sfence

			popad
	}
}

int init(void)
{
	for (unsigned int i = 0; i < _threadPerGrid; i++)
	{
		update[i] = 0.0f;

		unsigned int tmp = i;
		int X = 0, Y = 0, Z = 0;
		X += tmp % _blockDimX - 1; tmp /= _blockDimX;
		Y += tmp % _blockDimY - 1; tmp /= _blockDimY;
		Z += tmp % _blockDimZ - 1; tmp /= _blockDimZ;
		X += (tmp % _gridDimX) * (_blockDimX-2); tmp /= _gridDimX;
		Y += (tmp % _gridDimY) * (_blockDimY-2); tmp /= _gridDimY;
		Z += (tmp % _gridDimZ) * (_blockDimZ-2); tmp /= _gridDimZ;

		if (X < 0 || _DimX-1 < X || Y < 0 || _DimY-1 < Y || Z < 0 || _DimZ-1 < Z) { continue; }
		//else if (0 <= X) { Ex[i] = 8.8f; }
		else if (0 <= X) { Ex[i] = (float)(X*10000 + Y*100 + Z); }
		else { Ex[i] = 0.0f; }

	}
	return 0;
}
int snapshot(void)
{
	printf("Z=2\n");
	for (int Y = 0; Y < _DimY; Y++) {
		for (int X = 0; X < _DimX; X++) {
			int Z = 2;
			//printf("%1.1f \t", update[_INDEX_XYZ(X, Y, Z) ]);
			printf("%06.0f \t", update[_INDEX_XYZ(X, Y, Z)]);
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
			printf("%06.0f \t", update[_INDEX_XYZ(X, Y, Z)]);
		}
		printf("\n");
	}
	return 0;
}
int ConvolutionC(
	unsigned char *pSrc,
	unsigned char *pDest,
	unsigned int nSizeX,
	unsigned int nSizeY,
	unsigned int nSizeZ,
	unsigned int *ROIPoint,
	short *pKernel
)
{
	unsigned int nStartX = ROIPoint[0];
	unsigned int nStartY = ROIPoint[1];
	unsigned int nStartZ = ROIPoint[2];
	unsigned int nEndX = ROIPoint[3];
	unsigned int nEndY = ROIPoint[4];
	unsigned int nEndZ = ROIPoint[5];

	if (0 == nStartX) nStartX = 1; //필터링 후 크기 1 감소
	if (0 == nStartY) nStartY = 1;
	if (0 == nStartZ) nStartZ = 1;
	if (nSizeX == nEndX) nEndX = nSizeX - 1;
	if (nSizeY == nEndY) nEndY = nSizeY - 1;
	if (nSizeZ == nEndZ) nEndZ = nSizeZ - 1;

	short total = 0;
	short value[9] = { 0 };

	for (unsigned int k = nStartZ; k < nEndZ; k++)
	{
		for (unsigned int j = nStartY; j < nEndY; j++)
		{
			for (unsigned int i = nStartX; i < nEndX; i++)
			{
				total = 0;
				value[0] = pSrc[i + j*nSizeX + k*nSizeX*nSizeY - 1];
				total += pKernel[0] * value[0];

				value[1] = pSrc[i + j*nSizeX + k*nSizeX*nSizeY + 1];
				total += pKernel[1] * value[1];

				value[2] = pSrc[i + j*nSizeX + k*nSizeX*nSizeY - nSizeX];
				total += pKernel[2] * value[2];

				value[3] = pSrc[i + j*nSizeX + k*nSizeX*nSizeY + nSizeX];
				total += pKernel[3] * value[3];

				value[4] = pSrc[i + j*nSizeX + k*nSizeX*nSizeY - nSizeX*nSizeY];
				total += pKernel[4] * value[4];

				value[5] = pSrc[i + j*nSizeX + k*nSizeX*nSizeY + nSizeX*nSizeY];
				total += pKernel[5] * value[5];

				if (total < 0) total = 0;
				if (total > 255) total = 255;

				pDest[i + j*nSizeX + k*nSizeX*nSizeY] = (unsigned char)total;
			}
		}
	}
	return 0;
}

void ConvolutionSIMD(
	unsigned char *pSrc, //char = 1byte
	unsigned char *pDest,
	unsigned int nSizeX,
	unsigned int nSizeY,
	unsigned int nSizeZ,
	unsigned int *ROIPoint,
	short *pKernel
)
{
	unsigned int nStartX = ROIPoint[0];
	unsigned int nStartY = ROIPoint[1];
	unsigned int nStartZ = ROIPoint[2];
	unsigned int nEndX = ROIPoint[3];
	unsigned int nEndY = ROIPoint[4];
	unsigned int nEndZ = ROIPoint[5];

	if (0 == nStartX) nStartX = 1; //필터링 후 크기 1 감소
	if (0 == nStartY) nStartY = 1;
	if (0 == nStartZ) nStartZ = 1;
	if (nSizeX == nEndX) nEndX = nSizeX - 1;
	if (nSizeY == nEndY) nEndY = nSizeY - 1;
	if (nSizeZ == nEndZ) nEndZ = nSizeZ - 1;

	// 더하고 나면 255보다 커지므로 2byte short로 형변환
	//__declspec(align(16)) short Mask0[8], Mask1[8], Mask2[8], Mask3[8],
	//	Mask4[8], Mask5[8];

}