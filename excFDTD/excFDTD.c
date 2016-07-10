//VS2015
//ipsxe2016 Cluster Edition

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
__declspec(align(32)) float stability_factor_inv[8] = { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f }; 
//__declspec(align(32)) float eps_r_inv[_threadPerGrid] = { 0 }; //이거 조절해서 zdivision 처리하는 것도 가능할듯

__declspec(align(32)) float test[8] = { 0.1f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};

int init(void);
int DrudeE(void);
int DrudeH(void);
int snapshot(void);
int main(int argc, char* argv[])
{
	printf("total area : block(%d,%d,%d) * grid(%d,%d,%d) = %d(%d,%d,%d)\n",_blockDimX,_blockDimY,_blockDimZ,_gridDimX,_gridDimY,_gridDimZ, _threadPerGrid, _blockDimX*_gridDimX, _blockDimY*_gridDimY, _blockDimZ*_gridDimZ);
	printf("thread space (padding overhead) : %d(%d,%d,%d) \n", (_blockDimX - 2)*_gridDimX* (_blockDimY - 2)*_gridDimY* (_blockDimZ - 2)*_gridDimZ,(_blockDimX-2)*_gridDimX , (_blockDimY - 2)*_gridDimY, (_blockDimZ - 2)*_gridDimZ);
	printf("xyz space (block overhead) : %d(%d,%d,%d) \n", _DimX*_DimY*_DimZ, _DimX, _DimY, _DimZ);
	printf("unit : sizeof(float)\n ");

	init();

	DrudeE();
	//DrudeH();

	snapshot();
	return 0;
}

int DrudeE(void)
{
	float* pEx = eps0_c_Ex;// +_offsetPadding;
	float* pEy = eps0_c_Ey;// + _offsetPadding;
	float* pEz = eps0_c_Ez;// + _offsetPadding;
	float* pHx = Hx;// + _offsetPadding;
	float* pHy = Hy;// + _offsetPadding;
	float* pHz = Hz;// + _offsetPadding;
	float* peps = eps_r_inv;
	float* pS = stability_factor_inv;
	//float* ptest = test;//vmovaps ymm0, [rbp]
	unsigned int loopsize = _threadPerGrid / 64; 	// 8 * ymm(32byte) = 64 * float(4byte)
	if ((_blockDimXYZ) % 64 != 0) { printf("Error : block size need to be a multiple of 64 float"); return -1; }

	__asm
	{
		nop

		//PUSHA
		push rax 
		push rbx 
		push rcx 
		push rdx 
		push rdi 
		push rsi 
		push r8 
		push r9 
		push r10 
		push r11 
		push r12 
		push r13 
		push r14 
		push r15 
		
		mov rax, pEx
		mov rbx, pEy
		mov rcx, pEz
		mov rdx, pHx
		mov rdi, pHy
		mov rsi, pHz
		mov r13, peps
		mov r14, pS
		mov r15, 1000 //FIXME:counter

	EX:
		vmovaps ymm0, [rdi- _offsetZ_byte]
		vmovaps ymm1, [rdi]
		vsubps ymm0, ymm0, ymm1
		vmovaps ymm1, [rsi - _offsetZ_byte]
		vaddps ymm0, ymm0, ymm1
		vmovaps ymm1, [rsi - _offsetY_byte - _offsetZ_byte]
		vsubps ymm0, ymm0, ymm1
		vmovaps ymm1, [r13]
		vmulps ymm0, ymm0, ymm1
		vmovaps ymm1, [r14]
		//movss ymm1, 0x3F0000003F000000 // unsupported?
		//vshufps ymm1,  ymm1, 0 // splat across all lanes of xmm1
		vmulps ymm0, ymm0, ymm1
		vmovaps ymm1, [rax]
		vaddps ymm0, ymm0, ymm1

		vmovaps [rax], ymm0


		add rax, 32
		add rbx, 32
		add rcx, 32
		add rdx, 32
		add rdi, 32
		add rsi, 32
		add r13, 32
		dec r15
		jnz EX
		sfence

		//POPA
		pop r15 
		pop r14 
		pop r13 
		pop r12 
		pop r11 
		pop r10 
		pop r9 
		pop r8 
		pop rsi 
		pop rdi 
		pop rdx 
		pop rcx 
		pop rbx 
		pop rax 

		nop

		//movdqa xmm0 
	}
}


int noupdate(void)
{
	float* pEx = eps0_c_Ex;
	float* pEy = eps0_c_Ey;
	unsigned int loopsize = _threadPerGrid / 64; 	// 8 * ymm(32byte) = 64 * float(4byte)
	if ((_blockDimXYZ) % 64 != 0) { printf("Error : block size need to be a multiple of 64 float"); return -1; }
	__asm
	{
		//pushad
		mov rcx, loopsize
		mov rax, pEx
		mov rdx, pEy
		mov rsi, 0
		//		prefetchnta[eax + _sizeofFloat * _blockDimXYZ] //block cache
		MEMLP:
		vmovaps ymm0, [rax + rsi]
			vmovaps ymm1, [rax + rsi + 32]
			vmovaps ymm2, [rax + rsi + 64]
			vmovaps ymm3, [rax + rsi + 96]
			vmovaps ymm4, [rax + rsi + 128]
			vmovaps ymm5, [rax + rsi + 160]
			vmovaps ymm6, [rax + rsi + 192]
			vmovaps ymm7, [rax + rsi + 224]

			vmovaps[rdx + rsi], ymm0
			vmovaps[rdx + rsi + 32], ymm1
			vmovaps[rdx + rsi + 64], ymm2
			vmovaps[rdx + rsi + 96], ymm3
			vmovaps[rdx + rsi + 128], ymm4
			vmovaps[rdx + rsi + 160], ymm5
			vmovaps[rdx + rsi + 192], ymm6
			vmovaps[rdx + rsi + 224], ymm7

			add rsi, 256
			dec rcx
			jnz MEMLP
			sfence

			//popad
	}
}




int init(void)
{
	for (unsigned int i = 0; i < _threadPerGrid; i++)
	{
		//eps0_c_Ey[i] = 0.0f;

		unsigned int tmp = i;
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
		//else if (0 <= X) { eps0_c_Ex[i] = 8.8f; }
		else if (0 <= X) { 	eps0_c_Ex[i] = (float)(X*10000 + Y*100 + Z);}
		else { eps0_c_Ex[i] = 0.0f; }


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