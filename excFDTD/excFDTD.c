#include "stdio.h"
#include "stdlib.h"

#define _INDEX_BLOCK(blockIdxX, blockIdxY, blockIdxZ) \
		((blockIdxX) + gridDimX * (blockIdxY) + gridDimX * gridDimY * (blockIdxZ))
#define _INDEX_THREAD(blockIdxX, blockIdxY, blockIdxZ, threadIdxX, threadIdxY, threadIdxZ) \
		( \
		(_INDEX_BLOCK(blockIdxX, blockIdxY, blockIdxZ)) \
		* ((blockDimX) * (blockDimY) * (blockDimZ)) \
		+ (threadIdxX) + blockDimX * (threadIdxY) + blockDimX * blockDimY * (threadIdxZ) \
		)
#define _INDEX_XYZ(x,y,z) \
		_INDEX_THREAD((x)/blockDimX, (y)/blockDimY, (z)/blockDimZ, (x)%blockDimX, (y)%blockDimY, (z)%blockDimZ)

//thread를 blockDim개씩 묶어서 block이 gridDim개 
unsigned int DimX = 10, DimY = 10, DimZ = 10;
unsigned int blockDimX = 4, blockDimY = 4, blockDimZ = 4;
unsigned int gridDimX, gridDimY, gridDimZ;
unsigned int threadPerGrid;
float* Ex, *Ey, *Ez, *Hx, *Hy, *Hz, *update;

int init(void);
int snapshot(void);
int main(int argc, char* argv[])
{
	gridDimX = (DimX + blockDimX - 1) / blockDimX;
	gridDimY = (DimY + blockDimY - 1) / blockDimY;
	gridDimZ = (DimZ + blockDimZ - 1) / blockDimZ;

	threadPerGrid = blockDimX*gridDimX*blockDimY*gridDimY*blockDimZ*gridDimZ;

	Ex = (float*)malloc(sizeof(float) * threadPerGrid );
	update = (float*)malloc(sizeof(float) * threadPerGrid);
	init();

	int loopsize = gridDimX*gridDimY*gridDimZ;
	__asm
	{
		pushad

		mov ecx, loopsize
		mov eax, Ex
		mov edx, update
		mov esi, 0

		prefetchnta[eax + 32 * 4 * 4 * 4]
MEMLP:
		vmovups ymm0, [eax + esi]
		vmovups ymm1, [eax + esi + 32]
		vmovups ymm2, [eax + esi + 64]
		vmovups ymm3, [eax + esi + 96]
		vmovups ymm4, [eax + esi + 128]
		vmovups ymm5, [eax + esi + 160]
		vmovups ymm6, [eax + esi + 192]
		vmovups ymm7, [eax + esi + 224]

		vmovups [edx + esi], ymm0
		vmovups [edx + esi + 32], ymm1
		vmovups [edx + esi + 64], ymm2
		vmovups [edx + esi + 96], ymm3
		vmovups [edx + esi + 128], ymm4
		vmovups [edx + esi + 160], ymm5
		vmovups [edx + esi + 192], ymm6
		vmovups [edx + esi + 224], ymm7

		add esi, 256
		dec ecx
		jnz MEMLP
		sfence

		popad
	}

	snapshot();

	free(Ex);
	free(update);
	return 0;
}
int init(void)
{
	for (unsigned int i = 0; i < threadPerGrid; i++)
	{
		unsigned int tmp = i;
		unsigned int X = 0, Y = 0, Z = 0;
		X += tmp % blockDimX; tmp /= blockDimX;
		Y += tmp % blockDimY; tmp /= blockDimY;
		Z += tmp % blockDimZ; tmp /= blockDimZ;
		X += (tmp % gridDimX) * blockDimX; tmp /= gridDimX;
		Y += (tmp % gridDimY) * blockDimY; tmp /= gridDimY;
		Z += (tmp % gridDimZ) * blockDimZ; tmp /= gridDimZ;

		if (0 <= X && X <= 5 && 2 <= Y && Y <= 7 && 5 <= Z && Z <= 10) { Ex[i] = 8.8f; }
		else { Ex[i] = 0.0f; }
		update[i] = 0.0f;
	}
	return 0;
}
int snapshot(void)
{
	printf("Z=8\n");
	for (unsigned int Y = 0; Y < blockDimY*gridDimY; Y++) {
		for (unsigned int X = 0; X < blockDimX*gridDimX; X++) {
			unsigned int Z = 8;
			unsigned int k = _INDEX_XYZ(X, Y, Z);
			printf("%1.1f \t", update[_INDEX_XYZ(X, Y, Z) ]);
			//printf("%d \t", _INDEX_BLOCK(X/blockDimX, Y/blockDimY, Z/blockDimZ));
			//printf("%d \t",Y/blockDimY);
		}
		printf("\n");
	}
	printf("\n");
	printf("Y=5\n");
	for (unsigned int Z = 0; Z < blockDimZ*gridDimZ; Z++) {
		for (unsigned int X = 0; X < blockDimX*gridDimX; X++) {
			unsigned int Y = 5;
			unsigned int k = _INDEX_XYZ(X, Y, Z);
			printf("%1.1f \t", update[_INDEX_XYZ(X, Y, Z)]);
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