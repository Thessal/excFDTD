//ADE DCP FDTD AVX ver
//Jongkook Choi
//VS2015, ipsxe2016 Cluster Edition, Parallel MKL, IPP (multithread)
//reference : K.P.Prokopidis, 2013
//reference : uFDTD  (eecs.wsu.edu/~schneidj/ufdtd/)
//reference : Torok et al, 2006 (doi: 10.1364/JOSAA.23.000713) formulation used for NTFF


#define _DimX (104)
#define _DimY (60)
#define _DimZ (200)

#define _SOURCE_WAVELENGTH_ (450e-9)
#define _T_DECAY (100)
#define _STEP (6*_T_DECAY*6)

//eq35
//consider using simple PML for NTFF calculation
#define _PML_PX_X_ (0)
#define _PML_PX_Y_ (0)
#define _PML_PX_Z_ (8)
int pml_n = 3;
float pml_R = 10e-4;
float pml_kappa_max = 8.0f;
#define _NTFF_Margin_X_ (0)
#define _NTFF_Margin_Y_ (0)
#define _NTFF_Margin_Z_ ((int)(_DimZ/2 - _PML_PX_Z_ - 50))

#define _S_factor ((double)(2.0))
#define _dx (5e-9)
#define _PML_ALPHA_TUNING_ (1.0)
#define _PML_OMEGA_DT_TUNING_ (2.0*M_PI*_dx/_SOURCE_WAVELENGTH_/_S_factor)

#define __SUBSTRATE (100)
#define __PITCH (60)
#define __METAL_HOLE (14)
#define __METAL_DISK  (4)
#define __RADIUS_TOP_IN  (16)
#define __RADIUS_BOT_IN ( 5)
#define __RADIUS_TOP_OUT ( 27)
#define __RADIUS_BOT_OUT  (16)
#define __RADIUS_DISK_TOP ( 10)
#define __RADIUS_DISK_BOT ( 14)
#define __SLOT  (4.2)
#define __SIN_BOT  (6)
#define __SIN_TOP  (10)
#define __DEPTH ( 4)
#define __REF ( 1.0)
#define __SIO_INDEX (1.46)
#define __BACK  (1.2f)
#define __SLOT_RADIUS  (24)
#define __SMOOTHING  (9)

#define __METAL_ON__
#define _AREA_METAL_

#ifdef __METAL_ON__ 
#define _AREA_METAL_ \
for(int ind = 0; ind < 5; ind++){\
if ((-__SLOT<zz) && (zz <= 0) && (rr[ind] <= __SLOT_RADIUS))\
{eps_r_inv[offset] = 1.0f / (__BACK*__BACK); /*SiO slot*/} \
}\
for(int ind = 0; ind < 5; ind++){\
if (\
(-__SLOT < zz) && (zz <= (-__SLOT + __METAL_DISK))\
	&& (rr[ind] <= ((__RADIUS_DISK_TOP - __RADIUS_DISK_BOT) / __METAL_DISK*(zz + __SLOT) + __RADIUS_DISK_BOT))\
	) {\
	mask[offset] = mask[offset] | (0b0001 << 4);\
		eps_r_inv[offset] = 1.0f/(__BACK*__BACK);\
} /*Au disk*/\
if (\
(0 < zz) && (zz <= (__SIN_TOP + __METAL_HOLE))\
	&& (rr[ind] <= __RADIUS_BOT_OUT + (float)(__RADIUS_TOP_OUT - __RADIUS_BOT_OUT) / (float)(__SIN_TOP + __METAL_HOLE) * zz)\
	) {\
	mask[offset] = mask[offset] | (0b0001 << 4);\
		eps_r_inv[offset] = 1.0f/(__BACK*__BACK);\
} /*Au sidewall*/\
}\
if ((__SIN_TOP<zz) && (zz <= (__SIN_TOP + __METAL_HOLE))) {\
	mask[offset] = mask[offset] | (0b0001 << 4);\
}/*Au top*/\
for(int ind = 0; ind < 5; ind++){\
if (\
	(\
		(0 < zz) && (zz <= __SIN_TOP + __METAL_HOLE) &&\
		(rr[ind] <= __RADIUS_BOT_IN + ((float)(__RADIUS_TOP_IN - __RADIUS_BOT_IN) / (float)(__SIN_TOP + __METAL_HOLE)) * zz) \
	)\
			||\
	(\
		(0 < zz) && (zz <= __SMOOTHING) &&\
		(rr[ind] <= __RADIUS_BOT_IN + __SMOOTHING - zz)\
	)\
) {\
	mask[offset] = mask[offset] & ~(0b0001 << 4);\
} /*thruhole*/\
} 
#endif


#define STRUCTURE \
eps_r_inv[offset] = 1.0f / (__BACK*__BACK);\
if (((-__SLOT - __SIN_BOT<zz) && (zz <= -__SLOT))\
	|| ((0<zz) && (zz <= __SIN_TOP))) {\
	eps_r_inv[offset] = 1.0f / (1.8f*1.8f); /*SiN*/\
}\
if (((-__SLOT<zz) && (zz <= 0) )\
	|| (zz <= (-__SLOT - __SIN_BOT))) {\
	eps_r_inv[offset] = 1.0f / (__SIO_INDEX * __SIO_INDEX); /*SiO*/\
} \
if (((-__SLOT - __SIN_BOT - 8)<zz) && (zz <= (-__SLOT - __SIN_BOT))) {\
	eps_r_inv[offset] = 1.0f / (1.8f*1.8f); /*ITO*/\
} \
_AREA_METAL_\



#define _c0 (299792458.0)
#define _USE_MATH_DEFINES
#define _mu0_ ( 4e-7 * M_PI )
#define _eps0_ ( 1.0 / _c0 / _c0 / _mu0_ )
#define _dt_ (_dx / _c0 / _S_factor)
#define _cdt__ (_dx / _S_factor)
#define _cdt_div_dx (1 / _S_factor)
float stability_factor_inv = 1.0 / _S_factor;

//doi:10.1088/0022-3727/40/22/043, Au
//400-1000nm
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
#define _phi2_L (-0.087341)



// RFT NTFF frequency
#define RFT_WINDOW _STEP
//#if RFT_WINDOW > _STEP
//	#define RFT_WINDOW _STEP
//#endif
#define FREQ_N 1
float FREQ_LIST_DESIRED[FREQ_N] = { _c0 / 450e-9};
float RFT_K_LIST_CALCULATED[FREQ_N];


// == loop tiling ==

#define _blockDimX (20)
#define _blockDimY (20)
#define _blockDimZ (20)
#define _blockDimXYZ (_blockDimX * _blockDimY * _blockDimZ)

#define _sizeofFloat (32)

//3D loop tiling with 1px padding, for L1 cache or cuda SHMEM utilization
//1px padding required for easy L1 cache optimization
#define _gridDimX ((int)((_DimX + (_blockDimX-2) - 1) / (_blockDimX-2)))
#define _gridDimY ((int)((_DimY + (_blockDimY-2) - 1) / (_blockDimY-2)))
#define _gridDimZ ((int)((_DimZ + (_blockDimZ-2) - 1) / (_blockDimZ-2)))

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
//#define _SURF_Margin_ _NTFF_Margin_
#define _SURF_StartX_ ((_PML_PX_X_)+_NTFF_Margin_X_)
#define _SURF_StartY_ ((_PML_PX_Y_)+_NTFF_Margin_Y_)
#define _SURF_StartZ_ ((_PML_PX_Z_)+_NTFF_Margin_Z_)
#define _SURF_EndX_ ((_DimX-1)-(_PML_PX_X_)-_NTFF_Margin_X_) 
#define _SURF_EndY_ ((_DimY-1)-(_PML_PX_Y_)-_NTFF_Margin_Y_)
#define _SURF_EndZ_ ((_DimZ-1)-(_PML_PX_Z_)-_NTFF_Margin_Z_)
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

#define _C1_ ( (_b21 / _dt_ / _dt_) + (_b11 * 0.5 / _dt_) + (_b01 * 0.25) )
#define _C2_ ( (_b22 / _dt_ / _dt_) + (_b12 * 0.5 / _dt_) + (_b02 * 0.25) )
#define _C11 ( ( ( 2.0*_b21/_dt_/_dt_ ) - (_b01*0.5) ) / _C1_ )
#define _C12 ( ( ( 2.0*_b22/_dt_/_dt_ ) - (_b02*0.5) ) / _C2_ )
#define _C21 ( ( ( _b11 * 0.5 / _dt_ ) - ( _b21 / _dt_ / _dt_ ) - ( _b01 * 0.25 ) ) / _C1_ )
#define _C22 ( ( ( _b12 * 0.5 / _dt_ ) - ( _b22 / _dt_ / _dt_ ) - ( _b02 * 0.25 ) ) / _C2_ )
#define _C31 ( ( ( _a01 * 0.25 ) + ( _a11 * 0.5 / _dt_ ) ) / _C1_ )
#define _C32 ( ( ( _a02 * 0.25 ) + ( _a12 * 0.5 / _dt_ ) ) / _C2_ )
#define _C41 ( _a01 * 0.5 / _C1_ )
#define _C42 ( _a02 * 0.5 / _C2_ )
#define _C51 ( ( ( _a01 * 0.25 ) - ( _a11 * 0.5 / _dt_ ) ) / _C1_ )
#define _C52 ( ( ( _a02 * 0.25 ) - ( _a12 * 0.5 / _dt_ ) ) / _C2_ )
#define _C3p ( _C31 + _C32 )
#define _C4p ( _C41 + _C42 )
#define _C5p ( _C51 + _C52 )

#define _gamma__ (_gamma_D)
											   //FIXME : gamma = gamma_D ?  check
#define _d1 ( ( 2.0 - _gamma__ * _dt_ ) / ( 2 + _gamma__ * _dt_) )
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
__declspec(align(32)) static float sigmaX_dt_div_eps0[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float sigmaY_dt_div_eps0[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float sigmaZ_dt_div_eps0[_threadPerGrid] = { 0.0f };
__declspec(align(32)) static float kappaX[_threadPerGrid] = { 1.0f }; //1.0f not filled
__declspec(align(32)) static float kappaY[_threadPerGrid] = { 1.0f };
__declspec(align(32)) static float kappaZ[_threadPerGrid] = { 1.0f };
typedef struct { double re; double im; } complex16;
typedef struct { float re; float im; } complex8;
complex8 complex_make(float a, float b) { complex8 result = { a,b };  return result; }
complex8 complex_add(complex8 a, complex8 b) { return (complex_make(a.re + b.re, a.im + b.im)); }
complex8 complex_sub(complex8 a, complex8 b) { return (complex_make(a.re - b.re, a.im - b.im)); }
complex8 complex_mul(complex8 a, complex8 b) { return (complex_make(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)); }
complex8 complex_div(complex8 a, complex8 b) {
	return (complex_make((a.re * b.re + a.im * b.im) / (b.re * b.re + b.im * b.im)
		, (-a.re * b.im + a.im * b.re) / (b.re * b.re + b.im * b.im)));
}
//FIXME : wasting a lot of memory 
//used only inside PML area
__declspec(align(32)) static complex8 sx[_threadPerGrid];
__declspec(align(32)) static complex8 sy[_threadPerGrid];
__declspec(align(32)) static complex8 sz[_threadPerGrid];

__declspec(align(32)) static float eps0_c_Ex_swp[_threadPerGrid];
__declspec(align(32)) static float eps0_c_Ey_swp[_threadPerGrid];
__declspec(align(32)) static float eps0_c_Ez_swp[_threadPerGrid];
__declspec(align(32)) static float Rx[_threadPerGrid];
__declspec(align(32)) static float Ry[_threadPerGrid];
__declspec(align(32)) static float Rz[_threadPerGrid];
__declspec(align(32)) static float Rx_old[_threadPerGrid];
__declspec(align(32)) static float Ry_old[_threadPerGrid];
__declspec(align(32)) static float Rz_old[_threadPerGrid];
__declspec(align(32)) static float Sx[_threadPerGrid];
__declspec(align(32)) static float Sy[_threadPerGrid];
__declspec(align(32)) static float Sz[_threadPerGrid];

__declspec(align(32)) static float Hx_swp[_threadPerGrid];
__declspec(align(32)) static float Hy_swp[_threadPerGrid];
__declspec(align(32)) static float Hz_swp[_threadPerGrid];
__declspec(align(32)) static float Bx[_threadPerGrid];
__declspec(align(32)) static float By[_threadPerGrid];
__declspec(align(32)) static float Bz[_threadPerGrid];
__declspec(align(32)) static float Bx_old[_threadPerGrid];
__declspec(align(32)) static float By_old[_threadPerGrid];
__declspec(align(32)) static float Bz_old[_threadPerGrid];
__declspec(align(32)) static float Tx[_threadPerGrid];
__declspec(align(32)) static float Ty[_threadPerGrid];
__declspec(align(32)) static float Tz[_threadPerGrid];

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
void RFT(float background_index);
void NTFF_onlyUpside(void);
void NTFF(float background_index);
int snapshot(char*);
void snapshotStructure(void);

#include "mkl.h"
int main(int argc, char* argv[])
{
	init();
	time_t start;
	start = clock();
	printf("\nCalculating field \n");

	int sourcePos = _DimZ / 2 - __SLOT - __SIN_BOT - 8 - 10;
	char filename[256];
	FILE *fx = fopen("plane_output_Ex.txt", "a"); double planeoutX;
	FILE *fy = fopen("plane_output_Ey.txt", "a"); double planeoutY;
	FILE *fz = fopen("plane_output_Ez.txt", "a"); double planeoutZ;
	for (int i = 0; i <= _STEP; i++) {
		printf("%f%%\r", 100.0f*(float)i / _STEP);
		float addval = -sin(2 * M_PI* i * (_dt_ * _c0 / _SOURCE_WAVELENGTH_)) * exp(-(float)(i - 6 * _T_DECAY)*(i - 6 * _T_DECAY) / (float)(2 * _T_DECAY * _T_DECAY));
		addval /= __SIO_INDEX;
		for (int ii = 0; ii < _DimX; ii++) {
			for (int jj = 0; jj < _DimY; jj++) {
				eps0_c_Ey[_INDEX_XYZ(ii, jj, sourcePos)] += addval *0.5f;
				Hx[_INDEX_XYZ(ii, jj, sourcePos)] -= addval *0.25f;
				Hx[_INDEX_XYZ(ii, jj, sourcePos - 1)] -= addval *0.25f;
			}
		}
		DCP_HE_C();
		//RFT(__BACK); 
		planeoutX = 0.0;		planeoutY = 0.0;		planeoutZ = 0.0;
		for (int ii = 0; ii < _DimX; ii++) {
			for (int jj = 0; jj < _DimY; jj++) {
				planeoutX += eps0_c_Ex[_INDEX_XYZ(ii, jj, (_DimZ / 2 + __SIN_TOP + __METAL_HOLE + 20))];
				planeoutY += eps0_c_Ey[_INDEX_XYZ(ii, jj, (_DimZ / 2 + __SIN_TOP + __METAL_HOLE + 20))];
				planeoutZ += eps0_c_Ez[_INDEX_XYZ(ii, jj, (_DimZ / 2 + __SIN_TOP + __METAL_HOLE + 20))];
			}
		}

			
		fprintf(fx, "%e\t%30e\n", _dt_*(float)i, planeoutX*__BACK);
		fprintf(fy, "%e\t%30e\n", _dt_*(float)i, planeoutY*__BACK);
		fprintf(fz, "%e\t%30e\n", _dt_*(float)i, planeoutZ*__BACK);
		if ((i) % 50 == 0) {
			sprintf(filename, "%05d", i);
			snapshot(filename);
		}
	}
	printf("\ntime : %f\n", (double)(clock() - start) / CLK_TCK);
	//NTFF_onlyUpside();
	//NTFF(__BACK);
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
	//temporary variables
	float Ex1__Ex0, Ex1__Sx0, Ex1__Sx1, Sx1__Sx0, Sx1__Rx0, Sx1__Rx1;
	float Ey1__Ey0, Ey1__Sy0, Ey1__Sy1, Sy1__Sy0, Sy1__Ry0, Sy1__Ry1;
	float Ez1__Ez0, Ez1__Sz0, Ez1__Sz1, Sz1__Sz0, Sz1__Rz0, Sz1__Rz1;
	float Hx1__Hx0, Hx1__Tx0, Hx1__Tx1, Tx1__Tx0, Tx1__Bx0, Tx1__Bx1;
	float Hy1__Hy0, Hy1__Ty0, Hy1__Ty1, Ty1__Ty0, Ty1__By0, Ty1__By1;
	float Hz1__Hz0, Hz1__Tz0, Hz1__Tz1, Tz1__Tz0, Tz1__Bz0, Tz1__Bz1;

	syncPadding();


	/*Update H*/
	for (unsigned __int64 offset = 0; offset < _threadPerGrid; offset += 1) {
		if (((mask[offset] & (1 << 0)) >> 0) == 1) { continue; } // skip padding
		/* PML swap*/
		if (((mask[offset] & (1 << 1)) >> 1) == 1) {
			//swap
			Hx_swp[offset] = Hx[offset];
			Hy_swp[offset] = Hy[offset];
			Hz_swp[offset] = Hz[offset];
			Hx[offset] = complex_mul(complex_mul(sy[offset], sz[offset]), complex_div(complex_make(Hx[offset], 0), sx[offset])).re;
			Hy[offset] = complex_mul(complex_mul(sz[offset], sx[offset]), complex_div(complex_make(Hy[offset], 0), sy[offset])).re;
			Hz[offset] = complex_mul(complex_mul(sx[offset], sy[offset]), complex_div(complex_make(Hz[offset], 0), sz[offset])).re;
			Bx_old[offset] = Hx[offset];
			By_old[offset] = Hy[offset];
			Bz_old[offset] = Hz[offset];
		}
		/* update */
		Hx[offset] -= (eps0_c_Ey[offset] - eps0_c_Ey[offset + _offsetZ] + eps0_c_Ez[offset + _offsetX + _offsetY] - eps0_c_Ez[offset + _offsetX]) * _cdt_div_dx;
		Hy[offset] -= (eps0_c_Ez[offset] - eps0_c_Ez[offset + _offsetX] + eps0_c_Ex[offset + _offsetZ] - eps0_c_Ex[offset]) *_cdt_div_dx;
		Hz[offset] -= (eps0_c_Ex[offset + _offsetZ] - eps0_c_Ex[offset + _offsetY + _offsetZ] + eps0_c_Ey[offset + _offsetZ] - eps0_c_Ey[offset - _offsetX + _offsetZ]) * _cdt_div_dx;

		/* PML update*/
		if (((mask[offset] & (1 << 1)) >> 1) == 1) {
			//recover
			Bx[offset] = Hx[offset]; By[offset] = Hy[offset]; Bz[offset] = Hz[offset];
			Hx[offset] = Hx_swp[offset]; Hy[offset] = Hy_swp[offset]; Hz[offset] = Hz_swp[offset];

			//(43), (41)
			Hx1__Hx0 = (2.0f * kappaZ[offset] - sigmaZ_dt_div_eps0[offset] - alpha_dt_div_eps0[offset] * kappaZ[offset]) / (2.0f * kappaZ[offset] + sigmaZ_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaZ[offset]);
			Hx1__Tx1 = (2.0f * kappaX[offset] + sigmaX_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaX[offset]) / (2.0f * kappaZ[offset] + sigmaZ_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaZ[offset]);
			Hx1__Tx0 = -(2.0f * kappaX[offset] - sigmaX_dt_div_eps0[offset] - alpha_dt_div_eps0[offset] * kappaX[offset]) / (2.0f * kappaZ[offset] + sigmaZ_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaZ[offset]);
			Tx1__Tx0 = (2.0f * kappaY[offset] - sigmaY_dt_div_eps0[offset] - alpha_dt_div_eps0[offset] * kappaY[offset]) / (2.0f * kappaY[offset] + sigmaY_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaY[offset]);
			Tx1__Bx1 = (2.0f + alpha_dt_div_eps0[offset]) / (2.0f * kappaY[offset] + sigmaY_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaY[offset]);
			Tx1__Bx0 = -(2.0f - alpha_dt_div_eps0[offset]) / (2.0f * kappaY[offset] + sigmaY_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaY[offset]);

			Hy1__Hy0 = (2.0f * kappaX[offset] - sigmaX_dt_div_eps0[offset] - alpha_dt_div_eps0[offset] * kappaX[offset]) / (2.0f * kappaX[offset] + sigmaX_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaX[offset]);
			Hy1__Ty1 = (2.0f * kappaY[offset] + sigmaY_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaY[offset]) / (2.0f * kappaX[offset] + sigmaX_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaX[offset]);
			Hy1__Ty0 = -(2.0f * kappaY[offset] - sigmaY_dt_div_eps0[offset] - alpha_dt_div_eps0[offset] * kappaY[offset]) / (2.0f * kappaX[offset] + sigmaX_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaX[offset]);
			Ty1__Ty0 = (2.0f * kappaZ[offset] - sigmaZ_dt_div_eps0[offset] - alpha_dt_div_eps0[offset] * kappaZ[offset]) / (2.0f * kappaZ[offset] + sigmaZ_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaZ[offset]);
			Ty1__By1 = (2.0f + alpha_dt_div_eps0[offset]) / (2.0f * kappaZ[offset] + sigmaZ_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaZ[offset]);
			Ty1__By0 = -(2.0f - alpha_dt_div_eps0[offset]) / (2.0f * kappaZ[offset] + sigmaZ_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaZ[offset]);

			Hz1__Hz0 = (2.0f * kappaY[offset] - sigmaY_dt_div_eps0[offset] - alpha_dt_div_eps0[offset] * kappaY[offset]) / (2.0f * kappaY[offset] + sigmaY_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaY[offset]);
			Hz1__Tz1 = (2.0f * kappaZ[offset] + sigmaZ_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaZ[offset]) / (2.0f * kappaY[offset] + sigmaY_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaY[offset]);
			Hz1__Tz0 = -(2.0f * kappaZ[offset] - sigmaZ_dt_div_eps0[offset] - alpha_dt_div_eps0[offset] * kappaZ[offset]) / (2.0f * kappaY[offset] + sigmaY_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaY[offset]);
			Tz1__Tz0 = (2.0f * kappaX[offset] - sigmaX_dt_div_eps0[offset] - alpha_dt_div_eps0[offset] * kappaX[offset]) / (2.0f * kappaX[offset] + sigmaX_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaX[offset]);
			Tz1__Bz1 = (2.0f + alpha_dt_div_eps0[offset]) / (2.0f * kappaX[offset] + sigmaX_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaX[offset]);
			Tz1__Bz0 = -(2.0f - alpha_dt_div_eps0[offset]) / (2.0f * kappaX[offset] + sigmaX_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaX[offset]);

			//H write (0)
			Hx[offset] = Hx1__Hx0*Hx[offset] + Hx1__Tx0 * Tx[offset];
			Hy[offset] = Hy1__Hy0*Hy[offset] + Hy1__Ty0 * Ty[offset];
			Hz[offset] = Hz1__Hz0*Hz[offset] + Hz1__Tz0 * Tz[offset];
			//T update
			Tx[offset] = Tx1__Tx0 * Tx[offset] + Tx1__Bx0 * Bx_old[offset] + Tx1__Bx1 * Bx[offset];
			Ty[offset] = Ty1__Ty0 * Ty[offset] + Ty1__By0 * By_old[offset] + Ty1__By1 * By[offset];
			Tz[offset] = Tz1__Tz0 * Tz[offset] + Tz1__Bz0 * Bz_old[offset] + Tz1__Bz1 * Bz[offset];
			//H write (1)
			Hx[offset] += Hx1__Tx1 * Tx[offset];
			Hy[offset] += Hy1__Ty1 * Ty[offset];
			Hz[offset] += Hz1__Tz1 * Tz[offset];
		}
	}


	syncPadding();

	/*Update E*/
	for (unsigned __int64 offset = 0; offset < _threadPerGrid; offset += 1) {
		if (((mask[offset] & (1 << 0)) >> 0) == 1) { continue; } // skip padding
		/* PML swap*/
		if (((mask[offset] & (1 << 1)) >> 1) == 1) {
			//swap
			eps0_c_Ex_swp[offset] = eps0_c_Ex[offset];
			eps0_c_Ey_swp[offset] = eps0_c_Ey[offset];
			eps0_c_Ez_swp[offset] = eps0_c_Ez[offset];
			eps0_c_Ex[offset] = complex_mul(complex_mul(sy[offset], sz[offset]), complex_div(complex_make(eps0_c_Ex[offset], 0), sx[offset])).re;
			eps0_c_Ey[offset] = complex_mul(complex_mul(sz[offset], sx[offset]), complex_div(complex_make(eps0_c_Ey[offset], 0), sy[offset])).re;
			eps0_c_Ez[offset] = complex_mul(complex_mul(sx[offset], sy[offset]), complex_div(complex_make(eps0_c_Ez[offset], 0), sz[offset])).re;
			Rx_old[offset] = eps0_c_Ex[offset];
			Ry_old[offset] = eps0_c_Ey[offset];
			Rz_old[offset] = eps0_c_Ez[offset];
		}

		/* update */
		if (dielectric_flag == 0 && ((mask[offset] & (0b1111 << 4)) >> 4) > 0) {// metal
			tempx[offset] = (Hy[offset - _offsetZ] - Hy[offset] + Hz[offset - _offsetZ] - Hz[offset - _offsetY - _offsetZ]) * _cdt_div_dx ; //eq30term1
			tempy[offset] = (Hz[offset - _offsetZ] - Hz[offset + _offsetX - _offsetZ] + Hx[offset] - Hx[offset - _offsetZ]) * _cdt_div_dx ;
			tempz[offset] = (Hx[offset - _offsetX - _offsetY] - Hx[offset - _offsetX] + Hy[offset] - Hy[offset - _offsetX]) * _cdt_div_dx ;
			tempx[offset] += eps0_c_Ex[offset] * (((double)_eps0_ * (double)_eps_inf - 0.5 * _sigma_ * (double)_dt_ + _d2 - _C4p) / (double)_eps0_); //eq30term2
			tempy[offset] += eps0_c_Ey[offset] * (((double)_eps0_ * (double)_eps_inf - 0.5 * _sigma_ * (double)_dt_ + _d2 - _C4p) / (double)_eps0_); //FIXME : div_eps0 coeffs cleanup
			tempz[offset] += eps0_c_Ez[offset] * (((double)_eps0_ * (double)_eps_inf - 0.5 * _sigma_ * (double)_dt_ + _d2 - _C4p) / (double)_eps0_);
			tempx[offset] -= eps0_c_Ex_old[offset] * (_C5p / (double)_eps0_); //eq30term3
			tempy[offset] -= eps0_c_Ey_old[offset] * (_C5p / (double)_eps0_);
			tempz[offset] -= eps0_c_Ez_old[offset] * (_C5p / (double)_eps0_);
			tempx[offset] -= eps0_c_Pdx[offset] * ((_d1 - 1.0) / (double)_eps0_); //eq30term4
			tempy[offset] -= eps0_c_Pdy[offset] * ((_d1 - 1.0) / (double)_eps0_);
			tempz[offset] -= eps0_c_Pdz[offset] * ((_d1 - 1.0) / (double)_eps0_);
			tempx[offset] -= eps0_c_Pcp1x[offset] * ((_C11 - 1.0) / (double)_eps0_); //eq30term5
			tempy[offset] -= eps0_c_Pcp1y[offset] * ((_C11 - 1.0) / (double)_eps0_);
			tempz[offset] -= eps0_c_Pcp1z[offset] * ((_C11 - 1.0) / (double)_eps0_);
			tempx[offset] -= eps0_c_Pcp2x[offset] * ((_C12 - 1.0) / (double)_eps0_);
			tempy[offset] -= eps0_c_Pcp2y[offset] * ((_C12 - 1.0) / (double)_eps0_);
			tempz[offset] -= eps0_c_Pcp2z[offset] * ((_C12 - 1.0) / (double)_eps0_);
			tempx[offset] -= eps0_c_Pcp1x_old[offset] * (_C21 / (double)_eps0_); //eq30term6
			tempy[offset] -= eps0_c_Pcp1y_old[offset] * (_C21 / (double)_eps0_);
			tempz[offset] -= eps0_c_Pcp1z_old[offset] * (_C21 / (double)_eps0_);
			tempx[offset] -= eps0_c_Pcp2x_old[offset] * (_C22 / (double)_eps0_);
			tempy[offset] -= eps0_c_Pcp2y_old[offset] * (_C22 / (double)_eps0_);
			tempz[offset] -= eps0_c_Pcp2z_old[offset] * (_C22 / (double)_eps0_);
			tempx[offset] /= (((double)_eps0_ * _eps_inf + 0.5 * _sigma_ * (double)_dt_ - _d2 + _C3p) / (double)_eps0_); //eq30term0
			tempy[offset] /= (((double)_eps0_ * _eps_inf + 0.5 * _sigma_ * (double)_dt_ - _d2 + _C3p) / (double)_eps0_);
			tempz[offset] /= (((double)_eps0_ * _eps_inf + 0.5 * _sigma_ * (double)_dt_ - _d2 + _C3p) / (double)_eps0_);

			//PD (single drude)
			eps0_c_Pdx[offset] *= _d1; //eq27term1
			eps0_c_Pdy[offset] *= _d1;
			eps0_c_Pdz[offset] *= _d1;
			eps0_c_Pdx[offset] -= tempx[offset] * _d2; //eq27term2
			eps0_c_Pdy[offset] -= tempy[offset] * _d2;
			eps0_c_Pdz[offset] -= tempz[offset] * _d2;
			eps0_c_Pdx[offset] -= eps0_c_Ex[offset] * _d2; //eq27term3
			eps0_c_Pdy[offset] -= eps0_c_Ey[offset] * _d2;
			eps0_c_Pdz[offset] -= eps0_c_Ez[offset] * _d2;

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
			eps0_c_Pcp1x_old[offset] += tempx[offset] * _C31 ; //eq14term3
			eps0_c_Pcp1y_old[offset] += tempy[offset] * _C31 ;
			eps0_c_Pcp1z_old[offset] += tempz[offset] * _C31 ;
			eps0_c_Pcp2x_old[offset] += tempx[offset] * _C32 ;
			eps0_c_Pcp2y_old[offset] += tempy[offset] * _C32 ;
			eps0_c_Pcp2z_old[offset] += tempz[offset] * _C32 ;
			eps0_c_Pcp1x_old[offset] += eps0_c_Ex[offset] * _C41 ; //eq14term4
			eps0_c_Pcp1y_old[offset] += eps0_c_Ey[offset] * _C41 ;
			eps0_c_Pcp1z_old[offset] += eps0_c_Ez[offset] * _C41 ;
			eps0_c_Pcp2x_old[offset] += eps0_c_Ex[offset] * _C42 ;
			eps0_c_Pcp2y_old[offset] += eps0_c_Ey[offset] * _C42 ;
			eps0_c_Pcp2z_old[offset] += eps0_c_Ez[offset] * _C42 ;
			eps0_c_Pcp1x_old[offset] += eps0_c_Ex_old[offset] * _C51 ; //eq14term5
			eps0_c_Pcp1y_old[offset] += eps0_c_Ey_old[offset] * _C51 ;
			eps0_c_Pcp1z_old[offset] += eps0_c_Ez_old[offset] * _C51 ;
			eps0_c_Pcp2x_old[offset] += eps0_c_Ex_old[offset] * _C52 ;
			eps0_c_Pcp2y_old[offset] += eps0_c_Ey_old[offset] * _C52 ;
			eps0_c_Pcp2z_old[offset] += eps0_c_Ez_old[offset] * _C52 ;
			
			//SWAP
			eps0_c_Ex_old[offset] = eps0_c_Ex[offset]; eps0_c_Ex[offset] = tempx[offset];
			eps0_c_Ey_old[offset] = eps0_c_Ey[offset]; eps0_c_Ey[offset] = tempy[offset];
			eps0_c_Ez_old[offset] = eps0_c_Ez[offset]; eps0_c_Ez[offset] = tempz[offset];

			tempx[offset] = eps0_c_Pcp1x[offset]; eps0_c_Pcp1x[offset] = eps0_c_Pcp1x_old[offset]; eps0_c_Pcp1x_old[offset] = tempx[offset];
			tempy[offset] = eps0_c_Pcp1y[offset]; eps0_c_Pcp1y[offset] = eps0_c_Pcp1y_old[offset]; eps0_c_Pcp1y_old[offset] = tempy[offset];
			tempz[offset] = eps0_c_Pcp1z[offset]; eps0_c_Pcp1z[offset] = eps0_c_Pcp1z_old[offset]; eps0_c_Pcp1z_old[offset] = tempz[offset];

			tempx[offset] = eps0_c_Pcp2x[offset]; eps0_c_Pcp2x[offset] = eps0_c_Pcp2x_old[offset]; eps0_c_Pcp2x_old[offset] = tempx[offset];
			tempy[offset] = eps0_c_Pcp2y[offset]; eps0_c_Pcp2y[offset] = eps0_c_Pcp2y_old[offset]; eps0_c_Pcp2y_old[offset] = tempy[offset];
			tempz[offset] = eps0_c_Pcp2z[offset]; eps0_c_Pcp2z[offset] = eps0_c_Pcp2z_old[offset]; eps0_c_Pcp2z_old[offset] = tempz[offset];
		}
		else {	// non metal
			eps0_c_Ex[offset] += (Hy[offset - _offsetZ] - Hy[offset] + Hz[offset - _offsetZ] - Hz[offset - _offsetY - _offsetZ]) * eps_r_inv[offset] * _cdt_div_dx;
			eps0_c_Ey[offset] += (Hz[offset - _offsetZ] - Hz[offset + _offsetX - _offsetZ] + Hx[offset] - Hx[offset - _offsetZ]) * eps_r_inv[offset] * _cdt_div_dx;
			eps0_c_Ez[offset] += (Hx[offset - _offsetX - _offsetY] - Hx[offset - _offsetX] + Hy[offset] - Hy[offset - _offsetX]) * eps_r_inv[offset] * _cdt_div_dx;
		}

		/* PML update */
		if (((mask[offset] & (1 << 1)) >> 1) == 1) {//PML 
			//recover
			Rx[offset] = eps0_c_Ex[offset]; Ry[offset] = eps0_c_Ey[offset]; Rz[offset] = eps0_c_Ez[offset];
			eps0_c_Ex[offset] = eps0_c_Ex_swp[offset]; eps0_c_Ey[offset] = eps0_c_Ey_swp[offset]; eps0_c_Ez[offset] = eps0_c_Ez_swp[offset];

			//(43), (41)
			Ex1__Ex0 = (2.0f * kappaZ[offset] - sigmaZ_dt_div_eps0[offset] - alpha_dt_div_eps0[offset] * kappaZ[offset]) / (2.0f * kappaZ[offset] + sigmaZ_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaZ[offset]);
			Ex1__Sx1 = (2.0f * kappaX[offset] + sigmaX_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaX[offset]) / (2.0f * kappaZ[offset] + sigmaZ_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaZ[offset]);
			Ex1__Sx0 = -(2.0f * kappaX[offset] - sigmaX_dt_div_eps0[offset] - alpha_dt_div_eps0[offset] * kappaX[offset]) / (2.0f * kappaZ[offset] + sigmaZ_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaZ[offset]);
			Sx1__Sx0 = (2.0f * kappaY[offset] - sigmaY_dt_div_eps0[offset] - alpha_dt_div_eps0[offset] * kappaY[offset]) / (2.0f * kappaY[offset] + sigmaY_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaY[offset]);
			Sx1__Rx1 = (2.0f + alpha_dt_div_eps0[offset]) / (2.0f * kappaY[offset] + sigmaY_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaY[offset]);
			Sx1__Rx0 = -(2.0f - alpha_dt_div_eps0[offset]) / (2.0f * kappaY[offset] + sigmaY_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaY[offset]);

			Ey1__Ey0 = (2.0f * kappaX[offset] - sigmaX_dt_div_eps0[offset] - alpha_dt_div_eps0[offset] * kappaX[offset]) / (2.0f * kappaX[offset] + sigmaX_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaX[offset]);
			Ey1__Sy1 = (2.0f * kappaY[offset] + sigmaY_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaY[offset]) / (2.0f * kappaX[offset] + sigmaX_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaX[offset]);
			Ey1__Sy0 = -(2.0f * kappaY[offset] - sigmaY_dt_div_eps0[offset] - alpha_dt_div_eps0[offset] * kappaY[offset]) / (2.0f * kappaX[offset] + sigmaX_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaX[offset]);
			Sy1__Sy0 = (2.0f * kappaZ[offset] - sigmaZ_dt_div_eps0[offset] - alpha_dt_div_eps0[offset] * kappaZ[offset]) / (2.0f * kappaZ[offset] + sigmaZ_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaZ[offset]);
			Sy1__Ry1 = (2.0f + alpha_dt_div_eps0[offset]) / (2.0f * kappaZ[offset] + sigmaZ_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaZ[offset]);
			Sy1__Ry0 = -(2.0f - alpha_dt_div_eps0[offset]) / (2.0f * kappaZ[offset] + sigmaZ_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaZ[offset]);

			Ez1__Ez0 = (2.0f * kappaY[offset] - sigmaY_dt_div_eps0[offset] - alpha_dt_div_eps0[offset] * kappaY[offset]) / (2.0f * kappaY[offset] + sigmaY_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaY[offset]);
			Ez1__Sz1 = (2.0f * kappaZ[offset] + sigmaZ_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaZ[offset]) / (2.0f * kappaY[offset] + sigmaY_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaY[offset]);
			Ez1__Sz0 = -(2.0f * kappaZ[offset] - sigmaZ_dt_div_eps0[offset] - alpha_dt_div_eps0[offset] * kappaZ[offset]) / (2.0f * kappaY[offset] + sigmaY_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaY[offset]);
			Sz1__Sz0 = (2.0f * kappaX[offset] - sigmaX_dt_div_eps0[offset] - alpha_dt_div_eps0[offset] * kappaX[offset]) / (2.0f * kappaX[offset] + sigmaX_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaX[offset]);
			Sz1__Rz1 = (2.0f + alpha_dt_div_eps0[offset]) / (2.0f * kappaX[offset] + sigmaX_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaX[offset]);
			Sz1__Rz0 = -(2.0f - alpha_dt_div_eps0[offset]) / (2.0f * kappaX[offset] + sigmaX_dt_div_eps0[offset] + alpha_dt_div_eps0[offset] * kappaX[offset]);

			//E write (0)
			eps0_c_Ex[offset] = Ex1__Ex0*eps0_c_Ex[offset] + Ex1__Sx0 * Sx[offset];
			eps0_c_Ey[offset] = Ey1__Ey0*eps0_c_Ey[offset] + Ey1__Sy0 * Sy[offset];
			eps0_c_Ez[offset] = Ez1__Ez0*eps0_c_Ez[offset] + Ez1__Sz0 * Sz[offset];
			//S update
			Sx[offset] = Sx1__Sx0 * Sx[offset] + Sx1__Rx0 * Rx_old[offset] + Sx1__Rx1 * Rx[offset];
			Sy[offset] = Sy1__Sy0 * Sy[offset] + Sy1__Ry0 * Ry_old[offset] + Sy1__Ry1 * Ry[offset];
			Sz[offset] = Sz1__Sz0 * Sz[offset] + Sz1__Rz0 * Rz_old[offset] + Sz1__Rz1 * Rz[offset];
			//E write (1)
			eps0_c_Ex[offset] += Ex1__Sx1 * Sx[offset];
			eps0_c_Ey[offset] += Ey1__Sy1 * Sy[offset];
			eps0_c_Ez[offset] += Ez1__Sz1 * Sz[offset];

		}

	}
}

int RFT_counter = 0;
void RFT(float refractive_index) {
	if (RFT_counter < _STEP - RFT_WINDOW) { RFT_counter++; return; }
	unsigned __int64 surf_x, surf_y, surf_z, offset;
	for (int i = 0; i < _SURF_SIZE_; i++) {
		for (int j = 0; j < FREQ_N; j++) {
			_SET_SURF_XYZ_INDEX(i);
			offset = _INDEX_XYZ(surf_x, surf_y, surf_z);
			//Yee cell mismatch consideration //FIXME : see 14.79
			FT_eps0cE[i][j][0][1] -= eps0_c_Ex[offset] / (RFT_WINDOW - 1) * sinf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			FT_eps0cE[i][j][1][1] -= eps0_c_Ey[offset] / (RFT_WINDOW - 1) * sinf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			FT_eps0cE[i][j][2][1] -= eps0_c_Ez[offset] / (RFT_WINDOW - 1) * sinf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			FT_H[i][j][0][1] -= Hx[offset] / (RFT_WINDOW - 1) * sinf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			FT_H[i][j][1][1] -= Hy[offset] / (RFT_WINDOW - 1) * sinf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			FT_H[i][j][2][1] -= Hz[offset] / (RFT_WINDOW - 1) * sinf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			if (RFT_counter == _STEP) { continue; }
			FT_eps0cE[i][j][0][0] += eps0_c_Ex[offset] / RFT_WINDOW * cosf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			FT_eps0cE[i][j][1][0] += eps0_c_Ey[offset] / RFT_WINDOW * cosf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			FT_eps0cE[i][j][2][0] += eps0_c_Ez[offset] / RFT_WINDOW * cosf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			FT_H[i][j][0][0] += Hx[offset] / RFT_WINDOW * cosf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter); //RFT_counter is not q but it shoud not matter
			FT_H[i][j][1][0] += Hy[offset] / RFT_WINDOW * cosf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			FT_H[i][j][2][0] += Hz[offset] / RFT_WINDOW * cosf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);


			//imag
			//FIXME : loop tiling boundary
			//offset = _INDEX_XYZ(surf_x - 1, surf_y, surf_z);
			//FT_eps0cE[i][j][0][1] -= 0.5 * (eps0_c_Ex[offset])
			//offset = _INDEX_XYZ(surf_x, surf_y, surf_z);
			//FT_eps0cE[i][j][0][1] -= 0.5 * ( eps0_c_Ex[offset]) 
			//	/ (RFT_WINDOW - 1) * sinf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			//FT_eps0cE[i][j][1][1] -= 0.5 * (eps0_c_Ey[offset - _offsetX - _offsetY] + eps0_c_Ey[offset - _offsetX]) 
			//	/ (RFT_WINDOW - 1) * sinf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			//FT_eps0cE[i][j][2][1] -= 0.5 * (eps0_c_Ez[offset - _offsetZ] + eps0_c_Ez[offset])
			//	/ (RFT_WINDOW - 1) * sinf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			//FT_H[i][j][0][1] -= 0.25 * (Hx[offset - _offsetX - _offsetY - _offsetZ] + Hx[offset - _offsetX - _offsetY] + Hx[offset - _offsetX - _offsetZ] + Hx[offset - _offsetX]) / (RFT_WINDOW - 1) * sinf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			//FT_H[i][j][1][1] -= 0.25 * (Hy[offset - _offsetX - _offsetZ] + Hy[offset - _offsetX] + Hy[offset - _offsetZ] + Hy[offset]) / (RFT_WINDOW - 1) * sinf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			//FT_H[i][j][2][1] -= 0.25 * (Hz[offset - _offsetZ - _offsetX - _offsetY] + Hz[offset - _offsetZ - _offsetX] + Hz[offset - _offsetZ - _offsetY] + Hz[offset - _offsetZ]) / (RFT_WINDOW - 1) * sinf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			//if (RFT_counter == _STEP) { continue; }
			////real
			//FT_eps0cE[i][j][0][0] += 0.5 * (eps0_c_Ex[offset - _offsetX] + eps0_c_Ex[offset]) / RFT_WINDOW * cosf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter); //RFT_counter is not q but it shoud not matter
			//FT_eps0cE[i][j][1][0] += 0.5 * (eps0_c_Ey[offset - _offsetX - _offsetY] + eps0_c_Ey[offset - _offsetX]) / RFT_WINDOW * cosf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			//FT_eps0cE[i][j][2][0] += 0.5 * (eps0_c_Ez[offset - _offsetZ] + eps0_c_Ez[offset]) / RFT_WINDOW * cosf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			//FT_H[i][j][0][0] += 0.25 * (Hx[offset - _offsetX - _offsetY - _offsetZ] + Hx[offset - _offsetX - _offsetY] + Hx[offset - _offsetX - _offsetZ] + Hx[offset - _offsetX]) / RFT_WINDOW * cosf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter); //RFT_counter is not q but it shoud not matter
			//FT_H[i][j][1][0] += 0.25 * (Hy[offset - _offsetX - _offsetZ] + Hy[offset - _offsetX] + Hy[offset - _offsetZ] + Hy[offset]) / RFT_WINDOW * cosf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
			//FT_H[i][j][2][0] += 0.25 * (Hz[offset - _offsetZ - _offsetX - _offsetY] + Hz[offset - _offsetZ - _offsetX] + Hz[offset - _offsetZ - _offsetY] + Hz[offset - _offsetZ]) / RFT_WINDOW * cosf(2 * M_PI*RFT_K_LIST_CALCULATED[j] / RFT_WINDOW*RFT_counter);
		} //FIXME : use ipp functions!
	}
	if (RFT_counter == _STEP) {
		printf("RFT finish!\n");
		for (int i = 0; i < _SURF_SIZE_; i++) {
			for (int j = 0; j < FREQ_N; j++) {
				FT_eps0cE[i][j][0][1] *= refractive_index;
				FT_eps0cE[i][j][1][1] *= refractive_index;
				FT_eps0cE[i][j][2][1] *= refractive_index;
				FT_H[i][j][0][1] *= refractive_index;
				FT_H[i][j][1][1] *= refractive_index;
				FT_H[i][j][2][1] *= refractive_index;
				FT_eps0cE[i][j][0][0] *= refractive_index;
				FT_eps0cE[i][j][1][0] *= refractive_index;
				FT_eps0cE[i][j][2][0] *= refractive_index;
				FT_H[i][j][0][0] *= refractive_index;
				FT_H[i][j][1][0] *= refractive_index;
				FT_H[i][j][2][0] *= refractive_index;
			}
		}
	}
	RFT_counter++;
}

void NTFF_onlyUpside(void) {
	for (int i = 0; i < _SURF_SIZE_; i++) {
		for (int j = 0; j < FREQ_N; j++) {
			int surf_x, surf_y, surf_z;
			_SET_SURF_XYZ_INDEX(i);
			if (surf_z != _SURF_EndZ_) {
				FT_eps0cE[i][j][0][0] = 0;
				FT_eps0cE[i][j][1][0] = 0;
				FT_eps0cE[i][j][2][0] = 0;
				FT_eps0cE[i][j][0][1] = 0;
				FT_eps0cE[i][j][1][1] = 0;
				FT_eps0cE[i][j][2][1] = 0;
				FT_H[i][j][0][0] = 0;
				FT_H[i][j][1][0] = 0;
				FT_H[i][j][2][0] = 0;
				FT_H[i][j][0][1] = 0;
				FT_H[i][j][1][1] = 0;
				FT_H[i][j][2][1] = 0;
			}
		}
	}
}

#define NTFF_IMG_SIZE 200
#define _SURF_MidX_ ((_SURF_StartX_+_SURF_EndX_)/2.0f)
#define _SURF_MidY_ ((_SURF_StartY_+_SURF_EndY_)/2.0f)
#define _SURF_MidZ_ ((_SURF_StartZ_+_SURF_EndZ_)/2.0f)
#define DEBUG (1)

#include "mkl.h"
#include "ipp.h"
//#include "mkl_vml_functions.h"

void NTFF(float refractive_index) {
	printf("\nNTFF calculation\n");
	float progress = 0;

	// Images
	char filename[] = "FF_00.png";
	unsigned error;
	unsigned char* image = malloc(NTFF_IMG_SIZE * NTFF_IMG_SIZE * 4);
	unsigned char* image2 = malloc(_SURF_DX_ * _SURF_DY_ * 4);
	FILE *f;
	char filenameFull[256];

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

	NF_eyePos = (float*)mkl_malloc(NTFF_IMG_SIZE*NTFF_IMG_SIZE * 3 * sizeof(float), 64);
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
			float radius = ((float)NTFF_IMG_SIZE - 1.0f) / 2.0f;
			float xx = ((float)i - radius);
			float yy = ((float)j - radius);
			float rr = sqrtf(xx*xx + yy*yy);
			if (radius < rr) {
				NF_eyePos[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i] = -1.0f;
				NF_eyePos[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i] = -1.0f;
				NF_eyePos[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i] = -1.0f;
				continue;
			}
			if (rr < 0.1f) { // ==0.0f
				printf("center : %d, %d \n\n", i, j);
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
		k_vector = RFT_K_LIST_CALCULATED[freqN] * refractive_index * 2.0f * M_PI / RFT_WINDOW / _c0 / _dt_;
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
		//FIXME : do not repeat
		float normal_temp;
		for (int k = 0; k < _SURF_SIZE_; k++) {
			_SET_SURF_XYZ_INDEX(k);
			normal[0 * _SURF_SIZE_ + k].real = (surf_x == _SURF_StartX_) ? -1 : ((surf_x == _SURF_EndX_) ? 1 : 0);
			normal[1 * _SURF_SIZE_ + k].real = (surf_y == _SURF_StartY_) ? -1 : ((surf_y == _SURF_EndY_) ? 1 : 0);
			normal[2 * _SURF_SIZE_ + k].real = (surf_z == _SURF_StartZ_) ? -1 : ((surf_z == _SURF_EndZ_) ? 1 : 0);
			if (_NTFF_Margin_X_ == 0) { normal[0 * _SURF_SIZE_ + k].real = 0; }
			if (_NTFF_Margin_Y_ == 0) { normal[1 * _SURF_SIZE_ + k].real = 0; }
			if (_NTFF_Margin_Z_ == 0) { normal[2 * _SURF_SIZE_ + k].real = 0; }
			normal[0 * _SURF_SIZE_ + k].imag = 0;
			normal[1 * _SURF_SIZE_ + k].imag = 0;
			normal[2 * _SURF_SIZE_ + k].imag = 0;
			normal_temp = sqrtf(
				normal[0 * _SURF_SIZE_ + k].real*normal[0 * _SURF_SIZE_ + k].real
				+ normal[1 * _SURF_SIZE_ + k].real*normal[1 * _SURF_SIZE_ + k].real
				+ normal[2 * _SURF_SIZE_ + k].real*normal[2 * _SURF_SIZE_ + k].real);
			if (normal_temp > 0) {
				normal[0 * _SURF_SIZE_ + k].real /= normal_temp;
				normal[1 * _SURF_SIZE_ + k].real /= normal_temp;
				normal[2 * _SURF_SIZE_ + k].real /= normal_temp;
			}
		}

		////M, J calc
		// minus curl ==> {{0, nz, -ny}, {-nz, 0, nx}, {ny, -nx, 0}}
		ippsZero_32fc(NF_ecM, 3 * _SURF_SIZE_);
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
		ippsZero_32fc(NF_J, 3 * _SURF_SIZE_);
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
		ippsZero_64fc(NF_ecL_sum, 3 * NTFF_IMG_SIZE *NTFF_IMG_SIZE);
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

				for (int ii = 0; ii < 3; ii++) { // x y z
					vcMul(_SURF_SIZE_, Hankel_dx, NF_ecM + ii * _SURF_SIZE_, NF_ecL_dx + ii * _SURF_SIZE_);
					vcMul(_SURF_SIZE_, Hankel_dx, NF_J + ii * _SURF_SIZE_, NF_N_dx + ii * _SURF_SIZE_);
					for (int kk = 0; kk < _SURF_SIZE_; kk++) {
						if (isfinite(NF_ecL_dx[kk + ii*_SURF_SIZE_].real) == 0) { printf("error!\n"); }
						NF_ecL_sum[ii * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i].real += NF_ecL_dx[kk + ii * _SURF_SIZE_].real;
						NF_ecL_sum[ii * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i].imag += NF_ecL_dx[kk + ii * _SURF_SIZE_].imag;
						NF_N_sum[ii * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i].real += NF_N_dx[kk + ii * _SURF_SIZE_].real;
						NF_N_sum[ii * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i].imag += NF_N_dx[kk + ii * _SURF_SIZE_].imag;
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
				+NF_ecL_sum[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].imag * NF_eyePos[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE]
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
					(-FF_ecE_y[i + j*NTFF_IMG_SIZE].real * FF_H_z[i + j*NTFF_IMG_SIZE].imag + FF_ecE_y[i + j*NTFF_IMG_SIZE].imag * FF_H_z[i + j*NTFF_IMG_SIZE].real
						+ FF_ecE_z[i + j*NTFF_IMG_SIZE].real * FF_H_y[i + j*NTFF_IMG_SIZE].imag - FF_ecE_z[i + j*NTFF_IMG_SIZE].imag * FF_H_y[i + j*NTFF_IMG_SIZE].real)
					* NF_eyePos[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE] +
					(-FF_ecE_z[i + j*NTFF_IMG_SIZE].real * FF_H_x[i + j*NTFF_IMG_SIZE].imag + FF_ecE_z[i + j*NTFF_IMG_SIZE].imag * FF_H_x[i + j*NTFF_IMG_SIZE].real
						+ FF_ecE_x[i + j*NTFF_IMG_SIZE].real * FF_H_z[i + j*NTFF_IMG_SIZE].imag - FF_ecE_x[i + j*NTFF_IMG_SIZE].imag * FF_H_z[i + j*NTFF_IMG_SIZE].real)
					* NF_eyePos[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE] +
					(-FF_ecE_x[i + j*NTFF_IMG_SIZE].real * FF_H_y[i + j*NTFF_IMG_SIZE].imag + FF_ecE_x[i + j*NTFF_IMG_SIZE].imag * FF_H_y[i + j*NTFF_IMG_SIZE].real
						+ FF_ecE_y[i + j*NTFF_IMG_SIZE].real * FF_H_x[i + j*NTFF_IMG_SIZE].imag - FF_ecE_y[i + j*NTFF_IMG_SIZE].imag * FF_H_x[i + j*NTFF_IMG_SIZE].real)
					* NF_eyePos[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE]
					);
			}
		}

		//		ippsMulC_32fc_I((Ipp32fc) { sqrtf(0.125f / M_PI * k_vector,0.0f) }, FF_ecSr, NTFF_IMG_SIZE*NTFF_IMG_SIZE);
		ippsMulC_32fc_I((Ipp32fc) { sqrtf(0.125f / M_PI * k_vector), 0.0f }, FF_ecSr, NTFF_IMG_SIZE*NTFF_IMG_SIZE);
		ippsMulC_32fc_I((Ipp32fc) { sqrtf(0.125f / M_PI * k_vector), 0.0f }, FF_ecE_x, NTFF_IMG_SIZE*NTFF_IMG_SIZE);		//for plotting
		ippsMulC_32fc_I((Ipp32fc) { sqrtf(0.125f / M_PI * k_vector), 0.0f }, FF_ecE_y, NTFF_IMG_SIZE*NTFF_IMG_SIZE);
		ippsMulC_32fc_I((Ipp32fc) { sqrtf(0.125f / M_PI * k_vector), 0.0f }, FF_ecE_z, NTFF_IMG_SIZE*NTFF_IMG_SIZE);
		ippsMulC_32fc_I((Ipp32fc) { sqrtf(0.125f / M_PI * k_vector), 0.0f }, FF_H_x, NTFF_IMG_SIZE*NTFF_IMG_SIZE);
		ippsMulC_32fc_I((Ipp32fc) { sqrtf(0.125f / M_PI * k_vector), 0.0f }, FF_H_y, NTFF_IMG_SIZE*NTFF_IMG_SIZE);
		ippsMulC_32fc_I((Ipp32fc) { sqrtf(0.125f / M_PI * k_vector), 0.0f }, FF_H_z, NTFF_IMG_SIZE*NTFF_IMG_SIZE);


		for (int j = 0; j < _SURF_DY_; j++) {
			for (int i = 0; i < _SURF_DX_; i++) {
				float val = sqrtf(
					pow(NF_J[(0 * _SURF_SIZE_) + _SURF_INDEX_XYZ(_SURF_StartX_ + i, _SURF_StartY_ + j, _SURF_StartZ_)].real, 2) +
					pow(NF_J[(1 * _SURF_SIZE_) + _SURF_INDEX_XYZ(_SURF_StartX_ + i, _SURF_StartY_ + j, _SURF_StartZ_)].real, 2) +
					pow(NF_J[(2 * _SURF_SIZE_) + _SURF_INDEX_XYZ(_SURF_StartX_ + i, _SURF_StartY_ + j, _SURF_StartZ_)].real, 2)
				)
					*255.0f * 1000000.0f;
				image2[4 * _SURF_DX_ * j + 4 * i + 0] = val>0 ? (val>255 ? 255 : val) : 0;
				image2[4 * _SURF_DX_ * j + 4 * i + 1] = val<0 ? (val<-255 ? 255 : -val) : 0;
				image2[4 * _SURF_DX_ * j + 4 * i + 2] = 0;
				image2[4 * _SURF_DX_ * j + 4 * i + 3] = 255;
			}
		}
		sprintf(filenameFull, "NF_J_%d.png", freqN);
		error = lodepng_encode32_file(filenameFull, image2, _SURF_DX_, _SURF_DY_);
		if (error) printf("error %u: %s\n", error, lodepng_error_text(error));;
		for (int j = 0; j < _SURF_DY_; j++) {
			for (int i = 0; i < _SURF_DX_; i++) {
				float val = sqrtf(
					pow(NF_ecM[(0 * _SURF_SIZE_) + _SURF_INDEX_XYZ(_SURF_StartX_ + i, _SURF_StartY_ + j, _SURF_StartZ_)].real, 2) +
					pow(NF_ecM[(1 * _SURF_SIZE_) + _SURF_INDEX_XYZ(_SURF_StartX_ + i, _SURF_StartY_ + j, _SURF_StartZ_)].real, 2) +
					pow(NF_ecM[(2 * _SURF_SIZE_) + _SURF_INDEX_XYZ(_SURF_StartX_ + i, _SURF_StartY_ + j, _SURF_StartZ_)].real, 2)
				)
					*255.0f * 1000000.0f;
				image2[4 * _SURF_DX_ * j + 4 * i + 0] = val>0 ? (val>255 ? 255 : val) : 0;
				image2[4 * _SURF_DX_ * j + 4 * i + 1] = val<0 ? (val<-255 ? 255 : -val) : 0;
				image2[4 * _SURF_DX_ * j + 4 * i + 2] = 0;
				image2[4 * _SURF_DX_ * j + 4 * i + 3] = 255;
			}
		}
		sprintf(filenameFull, "NF_M_%d.png", freqN);
		error = lodepng_encode32_file(filenameFull, image2, _SURF_DX_, _SURF_DY_);
		if (error) printf("error %u: %s\n", error, lodepng_error_text(error));;
		for (int j = 0; j < _SURF_DY_; j++) {
			for (int i = 0; i < _SURF_DX_; i++) {
				float val = sqrtf(
					pow(NF_ecE[(0 * _SURF_SIZE_) + _SURF_INDEX_XYZ(_SURF_StartX_ + i, _SURF_StartY_ + j, _SURF_StartZ_)].real, 2) +
					pow(NF_ecE[(1 * _SURF_SIZE_) + _SURF_INDEX_XYZ(_SURF_StartX_ + i, _SURF_StartY_ + j, _SURF_StartZ_)].real, 2) +
					pow(NF_ecE[(2 * _SURF_SIZE_) + _SURF_INDEX_XYZ(_SURF_StartX_ + i, _SURF_StartY_ + j, _SURF_StartZ_)].real, 2)
				)
					*255.0f * 1000000.0f;
				image2[4 * _SURF_DX_ * j + 4 * i + 0] = val>0 ? (val>255 ? 255 : val) : 0;
				image2[4 * _SURF_DX_ * j + 4 * i + 1] = val<0 ? (val<-255 ? 255 : -val) : 0;
				image2[4 * _SURF_DX_ * j + 4 * i + 2] = 0;
				image2[4 * _SURF_DX_ * j + 4 * i + 3] = 255;
			}
		}
		sprintf(filenameFull, "NF_E_%d.png", freqN);
		error = lodepng_encode32_file(filenameFull, image2, _SURF_DX_, _SURF_DY_);
		if (error) printf("error %u: %s\n", error, lodepng_error_text(error));;
		for (int j = 0; j < _SURF_DY_; j++) {
			for (int i = 0; i < _SURF_DX_; i++) {
				float val = sqrtf(
					pow(NF_H[(0 * _SURF_SIZE_) + _SURF_INDEX_XYZ(_SURF_StartX_ + i, _SURF_StartY_ + j, _SURF_StartZ_)].real, 2) +
					pow(NF_H[(1 * _SURF_SIZE_) + _SURF_INDEX_XYZ(_SURF_StartX_ + i, _SURF_StartY_ + j, _SURF_StartZ_)].real, 2) +
					pow(NF_H[(2 * _SURF_SIZE_) + _SURF_INDEX_XYZ(_SURF_StartX_ + i, _SURF_StartY_ + j, _SURF_StartZ_)].real, 2)
				)
					*255.0f * 1000000.0f;
				image2[4 * _SURF_DX_ * j + 4 * i + 0] = val>0 ? (val>255 ? 255 : val) : 0;
				image2[4 * _SURF_DX_ * j + 4 * i + 1] = val<0 ? (val<-255 ? 255 : -val) : 0;
				image2[4 * _SURF_DX_ * j + 4 * i + 2] = 0;
				image2[4 * _SURF_DX_ * j + 4 * i + 3] = 255;
			}
		}
		sprintf(filenameFull, "NF_H_%d.png", freqN);
		error = lodepng_encode32_file(filenameFull, image2, _SURF_DX_, _SURF_DY_);
		if (error) printf("error %u: %s\n", error, lodepng_error_text(error));;
		for (int j = 0; j < _SURF_DY_; j++) {
			for (int i = 0; i < _SURF_DX_; i++) {
				float val = Hankel_dx[_SURF_INDEX_XYZ(_SURF_StartX_ + i, _SURF_StartY_ + j, _SURF_StartZ_)].real *255.0f;
				image2[4 * _SURF_DX_ * j + 4 * i + 0] = val>0 ? (val>255 ? 255 : val) : 0;
				image2[4 * _SURF_DX_ * j + 4 * i + 1] = val<0 ? (val<-255 ? 255 : -val) : 0;
				image2[4 * _SURF_DX_ * j + 4 * i + 2] = 0;
				image2[4 * _SURF_DX_ * j + 4 * i + 3] = 255;
			}
		}
		sprintf(filenameFull, "Hankel_real_%d.png", freqN);
		error = lodepng_encode32_file(filenameFull, image2, _SURF_DX_, _SURF_DY_);
		if (error) printf("error %u: %s\n", error, lodepng_error_text(error));;
		for (int j = 0; j < _SURF_DY_; j++) {
			for (int i = 0; i < _SURF_DX_; i++) {
				float val = Hankel_dx[_SURF_INDEX_XYZ(_SURF_StartX_ + i, _SURF_StartY_ + j, _SURF_StartZ_)].imag *255.0f;
				image2[4 * _SURF_DX_ * j + 4 * i + 0] = val>0 ? (val>255 ? 255 : val) : 0;
				image2[4 * _SURF_DX_ * j + 4 * i + 1] = val<0 ? (val<-255 ? 255 : -val) : 0;
				image2[4 * _SURF_DX_ * j + 4 * i + 2] = 0;
				image2[4 * _SURF_DX_ * j + 4 * i + 3] = 255;
			}
		}
		sprintf(filenameFull, "Hankel_imag_%d.png", freqN);
		error = lodepng_encode32_file(filenameFull, image2, _SURF_DX_, _SURF_DY_);
		if (error) printf("error %u: %s\n", error, lodepng_error_text(error));


		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = sqrtf(0.125f / M_PI * k_vector)* NF_N_sum[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i].real * 255.0f / 1.0f;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}
		sprintf(filenameFull, "Nx_%d.png", freqN); error = lodepng_encode32_file(filenameFull, image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);

		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = sqrtf(0.125f / M_PI * k_vector)* NF_N_sum[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i].real * 255.0f / 1.0f;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}
		sprintf(filenameFull, "Ny_%d.png", freqN); error = lodepng_encode32_file(filenameFull, image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);

		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = sqrtf(0.125f / M_PI * k_vector)* NF_N_sum[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + j*NTFF_IMG_SIZE + i].real * 255.0f / 1.0f;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}
		sprintf(filenameFull, "Nz_%d.png", freqN); error = lodepng_encode32_file(filenameFull, image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);

		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = sqrtf(0.125f / M_PI * k_vector)*(+NF_ecL_sum[0 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].real * NF_eyePos[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE])
					* 255.0f / 1.0f;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}
		sprintf(filenameFull, "Lx_%d.png", freqN); error = lodepng_encode32_file(filenameFull, image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);

		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = sqrtf(0.125f / M_PI * k_vector)*(+NF_ecL_sum[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].real * NF_eyePos[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE])
					* 255.0f / 1.0f;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}
		sprintf(filenameFull, "Ly_%d.png", freqN); error = lodepng_encode32_file(filenameFull, image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);

		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = sqrtf(0.125f / M_PI * k_vector)*(+NF_ecL_sum[2 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE].real * NF_eyePos[1 * NTFF_IMG_SIZE *NTFF_IMG_SIZE + i + j*NTFF_IMG_SIZE])
					* 255.0f / 1.0f;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}
		sprintf(filenameFull, "Lz_%d.png", freqN); error = lodepng_encode32_file(filenameFull, image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);

		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = FF_ecE_x[j*NTFF_IMG_SIZE + i].real * 255.0f / 1.0f;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}error = lodepng_encode32_file("E_x.png", image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);

		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = FF_ecE_y[j*NTFF_IMG_SIZE + i].real * 255.0f / 1.0f;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}
		sprintf(filenameFull, "E_y_%d.png", freqN); error = lodepng_encode32_file(filenameFull, image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);

		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = FF_ecE_z[j*NTFF_IMG_SIZE + i].real * 255.0f / 1.0f;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}
		sprintf(filenameFull, "E_z_%d.png", freqN); error = lodepng_encode32_file(filenameFull, image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);

		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = 255.0f * sqrtf(
					FF_ecE_x[j*NTFF_IMG_SIZE + i].real * FF_ecE_x[j*NTFF_IMG_SIZE + i].real +
					FF_ecE_y[j*NTFF_IMG_SIZE + i].real * FF_ecE_y[j*NTFF_IMG_SIZE + i].real +
					FF_ecE_z[j*NTFF_IMG_SIZE + i].real * FF_ecE_z[j*NTFF_IMG_SIZE + i].real +
					FF_ecE_x[j*NTFF_IMG_SIZE + i].imag * FF_ecE_x[j*NTFF_IMG_SIZE + i].imag +
					FF_ecE_y[j*NTFF_IMG_SIZE + i].imag * FF_ecE_y[j*NTFF_IMG_SIZE + i].imag +
					FF_ecE_z[j*NTFF_IMG_SIZE + i].imag * FF_ecE_z[j*NTFF_IMG_SIZE + i].imag
				);
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}
		sprintf(filenameFull, "E2_%d.png", freqN); error = lodepng_encode32_file(filenameFull, image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);

		sprintf(filenameFull, "E2_%d.txt", freqN);
		f = fopen(filenameFull, "w");
		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				fprintf(f, "%+04.3e\t", sqrtf(
					FF_ecE_x[j*NTFF_IMG_SIZE + i].real * FF_ecE_x[j*NTFF_IMG_SIZE + i].real +
					FF_ecE_y[j*NTFF_IMG_SIZE + i].real * FF_ecE_y[j*NTFF_IMG_SIZE + i].real +
					FF_ecE_z[j*NTFF_IMG_SIZE + i].real * FF_ecE_z[j*NTFF_IMG_SIZE + i].real +
					FF_ecE_x[j*NTFF_IMG_SIZE + i].imag * FF_ecE_x[j*NTFF_IMG_SIZE + i].imag +
					FF_ecE_y[j*NTFF_IMG_SIZE + i].imag * FF_ecE_y[j*NTFF_IMG_SIZE + i].imag +
					FF_ecE_z[j*NTFF_IMG_SIZE + i].imag * FF_ecE_z[j*NTFF_IMG_SIZE + i].imag
				));
			}
			fprintf(f, "\n");
		}
		fclose(f);

		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = 0.1*FF_H_x[j*NTFF_IMG_SIZE + i].real * 255.0f / 1.0f;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}
		sprintf(filenameFull, "H_x_%d.png", freqN); error = lodepng_encode32_file(filenameFull, image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);

		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = FF_H_y[j*NTFF_IMG_SIZE + i].real * 255.0f / 1.0f;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}
		sprintf(filenameFull, "H_y_%d.png", freqN); error = lodepng_encode32_file(filenameFull, image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);

		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = FF_H_z[j*NTFF_IMG_SIZE + i].real * 255.0f / 1.0f;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}
		sprintf(filenameFull, "H_z_%d.png", freqN); error = lodepng_encode32_file(filenameFull, image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);


		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = sqrtf(FF_ecSr[j*NTFF_IMG_SIZE + i].real *  FF_ecSr[j*NTFF_IMG_SIZE + i].real + FF_ecSr[j*NTFF_IMG_SIZE + i].imag *  FF_ecSr[j*NTFF_IMG_SIZE + i].imag)
					* 255.0f * 1e5;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}
		sprintf(filenameFull, "S_r_phasor_%d.png", freqN); error = lodepng_encode32_file(filenameFull, image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);

		sprintf(filenameFull, "S_r_phasor_%d.txt", freqN);
		f = fopen(filenameFull, "w");
		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				fprintf(f, "%+04.3e\t", sqrtf(FF_ecSr[j*NTFF_IMG_SIZE + i].real *  FF_ecSr[j*NTFF_IMG_SIZE + i].real + FF_ecSr[j*NTFF_IMG_SIZE + i].imag *  FF_ecSr[j*NTFF_IMG_SIZE + i].imag));
			}
			fprintf(f, "\n");
		}
		fclose(f);

		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				float val = FF_ecSr[j*NTFF_IMG_SIZE + i].real * 255.0f * 1e5;
				image[4 * NTFF_IMG_SIZE * j + 4 * i + 0] = val > 0 ? (val < 255 ? val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 1] = val < 0 ? (val > -255 ? -val : 255) : 0;		image[4 * NTFF_IMG_SIZE * j + 4 * i + 2] = 0;  image[4 * NTFF_IMG_SIZE * j + 4 * i + 3] = 255;
			}
		}
		sprintf(filenameFull, "S_r_avg_%d.png", freqN); error = lodepng_encode32_file(filenameFull, image, NTFF_IMG_SIZE, NTFF_IMG_SIZE);

		sprintf(filenameFull, "S_r_avg_%d.txt", freqN);
		f = fopen(filenameFull, "w");
		for (int i = 0; i < NTFF_IMG_SIZE; i++) {
			for (int j = 0; j < NTFF_IMG_SIZE; j++) {
				fprintf(f, "%+04.3e\t", FF_ecSr[j*NTFF_IMG_SIZE + i].real);
			}
			fprintf(f, "\n");
		}
		fclose(f);

	}

	free(image);
	free(image2);
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
	float pml_sigma_x_dt_div_eps0_max = -((float)pml_n + 1.0f)*logf(pml_R) * 0.5f / (_PML_PX_X_) / _S_factor; //eq37
	float pml_sigma_y_dt_div_eps0_max = -((float)pml_n + 1.0f)*logf(pml_R) * 0.5f / (_PML_PX_Y_) / _S_factor;
	float pml_sigma_z_dt_div_eps0_max = -((float)pml_n + 1.0f)*logf(pml_R) * 0.5f / (_PML_PX_Z_) / _S_factor;
	printf("PML : \n");
	printf("_eps0_ = %e\n", _eps0_);
	printf("(_PML_PX_X_) = %d\n", (_PML_PX_X_));
	printf("pml_n = %d\n", pml_n);
	printf("pml_R = %e\n", pml_R);
	printf("pml_kappa_max = %e\n", pml_kappa_max);
	printf("pml_sigma_x_dt_div_eps0_max = %e\n", pml_sigma_x_dt_div_eps0_max);
	printf("alpha_dt_div_eps0[i] = %e\n", (_PML_ALPHA_TUNING_)* (_dt_) /( _eps0_));
	printf("\n");

	for (unsigned __int64 i = 0; i < _threadPerGrid; i++)
	{
		eps_r_inv[i] = 1.0f;
		kappaX[i] = 1.0f;
		kappaY[i] = 1.0f;
		kappaZ[i] = 1.0f;
		sx[i].re = 1.0f;		sx[i].im = 0.0f;
		sy[i].re = 1.0f;		sy[i].im = 0.0f;
		sz[i].re = 1.0f;		sz[i].im = 0.0f;
		Sx[i] = 0.0f;		Sy[i] = 0.0f;		Sz[i] = 0.0f;
		Tx[i] = 0.0f;		Ty[i] = 0.0f;		Tz[i] = 0.0f;
		mask[i] = 0;

		//Indexing
		unsigned __int64 tmp = i;
		int X = 0, Y = 0, Z = 0;
		X += (tmp % _blockDimX) - 1; tmp /= _blockDimX;
		Y += (tmp % _blockDimY) - 1; tmp /= _blockDimY;
		Z += (tmp % _blockDimZ) - 1; tmp /= _blockDimZ;
		if (X == -1 || X == _blockDimX - 2 || Y == -1 || Y == _blockDimY - 2 || Z == -1 || Z == _blockDimZ - 2) {
			mask[i] |= (1 << 0); // 0th bit : Padding 
			continue;
		}
		X += (tmp % _gridDimX) * (_blockDimX - 2); tmp /= _gridDimX;
		Y += (tmp % _gridDimY) * (_blockDimY - 2); tmp /= _gridDimY;
		Z += (tmp % _gridDimZ) * (_blockDimZ - 2); tmp /= _gridDimZ;
		if (X >= _DimX || Y >= _DimY || Z >= _DimZ) {
			mask[i] |= (1 << 0); // 0th bit : Padding 
			continue;
		}

		alpha_dt_div_eps0[i] = (_PML_ALPHA_TUNING_) * (_dt_ )/ (_eps0_); // FIXME 


													 //FIXME : check PML area
													 //FIXME : sigma do not need to be an array
		if ((X + 1 <= ((_PML_PX_X_)) || (_DimX)-((_PML_PX_X_)) <= X) && X<_DimX) {
			mask[i] |= (1 << 1); // 1st bit : PML
								 //sigmaX_dt_div_eps0[i] = pml_sigma_x_dt_div_eps0_max * pow(fmin(fabs(((_PML_PX_X_)) - X), fabs(((_PML_PX_X_)) + X - (_DimX)+1) - 1.0f) / ((_PML_PX_X_)), (pml_n));
								 //kappaX[i] = 1.0f + (pml_kappa_max - 1.0f) * pow((fmin(fabs(((_PML_PX_X_)) - X), fabs(((_PML_PX_X_)) + X - (_DimX)+1)) - 1.0f) / ((float)(_PML_PX_X_)), (pml_n)); //FIXME : 0.5f? -1.0f? check 
			sigmaX_dt_div_eps0[i] = pml_sigma_x_dt_div_eps0_max * pow(fmin(fabs(((_PML_PX_X_)) - X), fabs(((_PML_PX_X_)) + X - (_DimX)+1)) / ((_PML_PX_X_)), (pml_n));
			kappaX[i] = 1.0f + (pml_kappa_max - 1.0f) * pow((fmin(fabs(((_PML_PX_X_)) - X), fabs(((_PML_PX_X_)) + X - (_DimX)+1))) / ((float)(_PML_PX_X_)), (pml_n));
			//sx[i].re = kappaX[i] + sigmaX_dt_div_eps0[i] * alpha_dt_div_eps0[i] / (alpha_dt_div_eps0[i] * alpha_dt_div_eps0[i] + _PML_OMEGA_DT_TUNING_ * _PML_OMEGA_DT_TUNING_);//prokopidis eq34
			//sx[i].im = - sigmaX_dt_div_eps0[i] * _PML_OMEGA_DT_TUNING_ / (alpha_dt_div_eps0[i] * alpha_dt_div_eps0[i] + _PML_OMEGA_DT_TUNING_ * _PML_OMEGA_DT_TUNING_);
			//lots of 0s, not efficient 
			sx[i] = complex_add(complex_make(kappaX[i], 0), complex_div(complex_make(sigmaX_dt_div_eps0[i], 0), complex_make(alpha_dt_div_eps0[i], _PML_OMEGA_DT_TUNING_)));

		}
		if (((Y + 1 <= ((_PML_PX_Y_))) || ((_DimY)-((_PML_PX_Y_)) <= Y)) && Y<_DimY) {
			mask[i] |= (1 << 1); // 1st bit : PML
								 //sigmaY_dt_div_eps0[i] = pml_sigma_y_dt_div_eps0_max * pow(fmin(fabs(((_PML_PX_Y_)) - Y), fabs(((_PML_PX_Y_)) + Y - (_DimY)+1) - 1.0f) / ((_PML_PX_Y_)), (pml_n));
								 //kappaY[i] = 1.0f + (pml_kappa_max - 1.0f) * pow((fmin(fabs(((_PML_PX_Y_)) - Y), fabs(((_PML_PX_Y_)) + Y - (_DimY)+1)) - 1.0f) / ((float)(_PML_PX_Y_)), (pml_n));
			sigmaY_dt_div_eps0[i] = pml_sigma_y_dt_div_eps0_max * pow(fmin(fabs(((_PML_PX_Y_)) - Y), fabs(((_PML_PX_Y_)) + Y - (_DimY)+1)) / ((_PML_PX_Y_)), (pml_n));
			kappaY[i] = 1.0f + (pml_kappa_max - 1.0f) * pow((fmin(fabs(((_PML_PX_Y_)) - Y), fabs(((_PML_PX_Y_)) + Y - (_DimY)+1))) / ((float)(_PML_PX_Y_)), (pml_n));
			sy[i] = complex_add(complex_make(kappaY[i], 0), complex_div(complex_make(sigmaY_dt_div_eps0[i], 0), complex_make(alpha_dt_div_eps0[i], _PML_OMEGA_DT_TUNING_)));
		}
		if ((Z + 1 <= ((_PML_PX_Z_)) || (_DimZ)-((_PML_PX_Z_)) <= Z) && Z<_DimZ) {
			mask[i] |= (1 << 1); // 1st bit : PML
								 //sigmaZ_dt_div_eps0[i] = pml_sigma_z_dt_div_eps0_max * pow(fmin(fabs(((_PML_PX_Z_)) - Z), fabs(((_PML_PX_Z_)) + Z - (_DimZ)+1) - 1.0f) / ((_PML_PX_Z_)), (pml_n));
								 //kappaZ[i] = 1.0f + (pml_kappa_max - 1.0f) * pow((fmin(fabs(((_PML_PX_Z_)) - Z), fabs(((_PML_PX_Z_)) + Z - (_DimZ)+1)) - 1.0f) / ((float)(_PML_PX_Z_)), (pml_n));
			sigmaZ_dt_div_eps0[i] = pml_sigma_z_dt_div_eps0_max * pow(fmin(fabs(((_PML_PX_Z_)) - Z), fabs(((_PML_PX_Z_)) + Z - (_DimZ)+1)) / ((_PML_PX_Z_)), (pml_n));
			kappaZ[i] = 1.0f + (pml_kappa_max - 1.0f) * pow((fmin(fabs(((_PML_PX_Z_)) - Z), fabs(((_PML_PX_Z_)) + Z - (_DimZ)+1))) / ((float)(_PML_PX_Z_)), (pml_n));
			sz[i] = complex_add(complex_make(kappaZ[i], 0), complex_div(complex_make(sigmaZ_dt_div_eps0[i], 0), complex_make(alpha_dt_div_eps0[i], _PML_OMEGA_DT_TUNING_)));
		}
		if (((mask[i] & (1 << 0)) >> 0) == 1) { // PML : exclude padding area
			mask[i] &= ~(1 << 1);
		}

	}

	int offset, xx, yy, zz;
	float rr[5];
	float arrX[5] = { -52,-52,0,52,52 };
	float arrY[5] = { -30,30,0,-30,30 };
	for (int x = 0; x < _DimX; x++) {
		for (int y = 0; y < _DimY; y++) {
			for (int z = 0; z < _DimZ; z++) {
				xx = x - ((_DimX) / 2);
				yy = y - ((_DimY) / 2);
				zz = z - ((_DimZ) / 2);
				for (int ind = 0; ind < 5; ind++) {
					rr[ind] = sqrtf(((float)xx - arrX[ind])*((float)xx - arrX[ind]) + ((float)yy - arrY[ind])*((float)yy - arrY[ind]));
				}
				offset = _INDEX_XYZ(x, y, z);
				STRUCTURE
			}
		}
	}
	snapshotStructure();

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
									mHy[offset - _offsetZ / 8], //strange
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
									_mm256_loadu_ps(Hz + offset * 8 + _offsetX - _offsetZ)),//SSE2 : _mm256_permutevar8x32_ps(mHy+offset*8,-1)									 
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

//FIXME : remove ternary ops --> use temp vars or macro

#define _syncX_(FIELD) \
FIELD[_INDEX_THREAD((X>0? X - 1:_gridDimX - 1), Y, Z, \
	(X>0? _blockDimX - 1 : (((_DimX-1))%(_blockDimX-2)+2) ), yy, zz )] \
= FIELD[_INDEX_THREAD(X, Y, Z, 1, yy, zz)]; \
FIELD[_INDEX_THREAD(X, Y, Z, 0, yy, zz)] \
= FIELD[_INDEX_THREAD((X>0? X - 1:_gridDimX - 1), Y, Z, \
	(X>0? _blockDimX - 2 : (((_DimX-1))%(_blockDimX-2)+1) ), yy, zz )];
#define _syncY_(FIELD) \
FIELD[_INDEX_THREAD(X, (Y>0? Y - 1:_gridDimY - 1), Z, \
	xx, (Y>0? _blockDimY - 1 : (((_DimY-1))%(_blockDimY-2)+2) ), zz )] \
= FIELD[_INDEX_THREAD(X, Y, Z, xx, 1, zz)]; \
FIELD[_INDEX_THREAD(X, Y, Z, xx, 0, zz)] \
= FIELD[_INDEX_THREAD(X, (Y>0? Y - 1:_gridDimY - 1), Z, \
	xx, (Y>0? _blockDimY - 2 : (((_DimY-1))%(_blockDimY-2)+1) ), zz )]; 
#define _syncZ_(FIELD) \
FIELD[_INDEX_THREAD(X, Y, (Z>0? Z - 1:_gridDimZ - 1), \
	xx, yy, (Z>0? _blockDimZ - 1 : (((_DimZ-1))%(_blockDimZ-2)+2) ) )] \
= FIELD[_INDEX_THREAD(X, Y, Z, xx, yy, 1)]; \
FIELD[_INDEX_THREAD(X, Y, Z, xx, yy, 0)] \
= FIELD[_INDEX_THREAD(X, Y, (Z>0? Z - 1:_gridDimZ - 1), \
	xx, yy, (Z>0? _blockDimZ - 2 : (((_DimZ-1))%(_blockDimZ-2)+1) ) )];
#define _syncXY_(FIELD) \
FIELD[_INDEX_THREAD((X>0? X - 1:_gridDimX - 1), (Y>0? Y - 1:_gridDimY - 1), Z, (X>0? _blockDimX - 1 : (((_DimX-1))%(_blockDimX-2)+2) ), (Y>0? _blockDimY - 1 : (((_DimY-1))%(_blockDimY-2)+2) ), zz)] = \
FIELD[_INDEX_THREAD(X, Y, Z, 1, 1, zz)]; \
FIELD[_INDEX_THREAD(X, Y, Z, 0, 0, zz)] = \
FIELD[_INDEX_THREAD((X>0? X - 1:_gridDimX - 1), (Y>0? Y - 1:_gridDimY - 1), Z, (X>0? _blockDimX - 2 : (((_DimX-1))%(_blockDimX-2)+1) ), (Y>0? _blockDimY - 2 : (((_DimY-1))%(_blockDimY-2)+1) ), zz)]; \
FIELD[_INDEX_THREAD(X, (Y>0? Y - 1:_gridDimY - 1), Z, 0, (Y>0? _blockDimY - 1 : (((_DimY-1))%(_blockDimY-2)+2) ), zz)] = \
FIELD[_INDEX_THREAD((X>0? X - 1:_gridDimX - 1), Y, Z, (X>0? _blockDimX - 2 : (((_DimX-1))%(_blockDimX-2)+1) ), 1, zz)]; \
FIELD[_INDEX_THREAD((X>0? X - 1:_gridDimX - 1), Y, Z, (X>0? _blockDimX - 1 : (((_DimX-1))%(_blockDimX-2)+2) ), 0, zz)] = \
FIELD[_INDEX_THREAD(X, (Y>0? Y - 1:_gridDimY - 1), Z, 1, (Y>0? _blockDimY - 2 : (((_DimY-1))%(_blockDimY-2)+1) ), zz)]; 
#define _syncYZ_(FIELD) \
FIELD[_INDEX_THREAD(X, (Y>0? Y - 1:_gridDimY - 1), (Z>0? Z - 1:_gridDimZ - 1), xx, (Y>0? _blockDimY - 1 : (((_DimY-1))%(_blockDimY-2)+2) ), (Z>0? _blockDimZ - 1 : (((_DimZ-1))%(_blockDimZ-2)+2) ))] = \
FIELD[_INDEX_THREAD(X, Y, Z, xx, 1, 1)]; \
FIELD[_INDEX_THREAD(X, Y, Z, xx, 0, 0)] = \
FIELD[_INDEX_THREAD(X, (Y>0? Y - 1:_gridDimY - 1), (Z>0? Z - 1:_gridDimZ - 1), xx, (Y>0? _blockDimY - 2 : (((_DimY-1))%(_blockDimY-2)+1) ), (Z>0? _blockDimZ - 2 : (((_DimZ-1))%(_blockDimZ-2)+1) ))]; \
FIELD[_INDEX_THREAD(X, Y, (Z>0? Z - 1:_gridDimZ - 1), xx, 0, (Z>0? _blockDimZ - 1 : (((_DimZ-1))%(_blockDimZ-2)+2) ))] = \
FIELD[_INDEX_THREAD(X, (Y>0? Y - 1:_gridDimY - 1), Z, xx, (Y>0? _blockDimY - 2 : (((_DimY-1))%(_blockDimY-2)+1) ), 1)]; \
FIELD[_INDEX_THREAD(X, (Y>0? Y - 1:_gridDimY - 1), Z, xx, (Y>0? _blockDimY - 1 : (((_DimY-1))%(_blockDimY-2)+2) ), 0)] = \
FIELD[_INDEX_THREAD(X, Y, (Z>0? Z - 1:_gridDimZ - 1), xx, 1, (Z>0? _blockDimZ - 2 : (((_DimZ-1))%(_blockDimZ-2)+1) ))]; 
#define _syncZX_(FIELD) \
FIELD[_INDEX_THREAD((X>0? X - 1:_gridDimX - 1), Y, (Z>0? Z - 1:_gridDimZ - 1), (X>0? _blockDimX - 1 : (((_DimX-1))%(_blockDimX-2)+2) ), yy, (Z>0? _blockDimZ - 1 : (((_DimZ-1))%(_blockDimZ-2)+2) ))] = \
FIELD[_INDEX_THREAD(X, Y, Z, 1, yy, 1)]; \
FIELD[_INDEX_THREAD(X, Y, Z, 0, yy, 0)] = \
FIELD[_INDEX_THREAD((X>0? X - 1:_gridDimX - 1), Y, (Z>0? Z - 1:_gridDimZ - 1), (X>0? _blockDimX - 2 : (((_DimX-1))%(_blockDimX-2)+1) ), yy, (Z>0? _blockDimZ - 2 : (((_DimZ-1))%(_blockDimZ-2)+1) ))]; \
FIELD[_INDEX_THREAD((X>0? X - 1:_gridDimX - 1), Y, Z, (X>0? _blockDimX - 1 : (((_DimX-1))%(_blockDimX-2)+2) ), yy, 0)] = \
FIELD[_INDEX_THREAD(X, Y, (Z>0? Z - 1:_gridDimZ - 1), 1, yy, (Z>0? _blockDimZ - 2 : (((_DimZ-1))%(_blockDimZ-2)+1) ))]; \
FIELD[_INDEX_THREAD(X, Y, (Z>0? Z - 1:_gridDimZ - 1), 0, yy, (Z>0? _blockDimZ - 1 : (((_DimZ-1))%(_blockDimZ-2)+2) ))] = \
FIELD[_INDEX_THREAD((X>0? X - 1:_gridDimX - 1), Y, Z, (X>0? _blockDimX - 2 : (((_DimX-1))%(_blockDimX-2)+1) ), yy, 1)];

#define _syncXall _syncX_(eps0_c_Ex) _syncX_(eps0_c_Ey) _syncX_(eps0_c_Ez) _syncX_(Hx)  _syncX_(Hy) _syncX_(Hz) \
//_syncX_(eps0_c_Pdx) _syncX_(eps0_c_Pdy) _syncX_(eps0_c_Pdz)  \
//_syncX_(eps0_c_Pcp1x) _syncX_(eps0_c_Pcp1y) _syncX_(eps0_c_Pcp1z)  \
//_syncX_(eps0_c_Pcp2x) _syncX_(eps0_c_Pcp2y) _syncX_(eps0_c_Pcp2z) 
//\
//_syncX_(eps0_c_Rx) _syncX_(eps0_c_Ry) _syncX_(eps0_c_Rz)  
#define _syncYall _syncY_(eps0_c_Ex) _syncY_(eps0_c_Ey) _syncY_(eps0_c_Ez) _syncY_(Hx)  _syncY_(Hy) _syncY_(Hz) \
//_syncY_(eps0_c_Pdx) _syncY_(eps0_c_Pdy) _syncY_(eps0_c_Pdz)  \
//_syncY_(eps0_c_Pcp1x) _syncY_(eps0_c_Pcp1y) _syncY_(eps0_c_Pcp1z)  \
//_syncY_(eps0_c_Pcp2x) _syncY_(eps0_c_Pcp2y) _syncY_(eps0_c_Pcp2z)  
//\
//_syncY_(eps0_c_Rx) _syncY_(eps0_c_Ry) _syncY_(eps0_c_Rz)  
#define _syncZall _syncZ_(eps0_c_Ex) _syncZ_(eps0_c_Ey) _syncZ_(eps0_c_Ez) _syncZ_(Hx)  _syncZ_(Hy) _syncZ_(Hz) \
//_syncZ_(eps0_c_Pdx) _syncZ_(eps0_c_Pdy) _syncZ_(eps0_c_Pdz)  \
//_syncZ_(eps0_c_Pcp1x) _syncZ_(eps0_c_Pcp1y) _syncZ_(eps0_c_Pcp1z)  \
//_syncZ_(eps0_c_Pcp2x) _syncZ_(eps0_c_Pcp2y) _syncZ_(eps0_c_Pcp2z)  
//\
//_syncZ_(eps0_c_Rx) _syncZ_(eps0_c_Ry) _syncZ_(eps0_c_Rz)  
#define _syncXYall _syncXY_(eps0_c_Ex) _syncXY_(eps0_c_Ey) _syncXY_(eps0_c_Ez) _syncXY_(Hx)  _syncXY_(Hy) _syncXY_(Hz) \
//_syncXY_(eps0_c_Pdx) _syncXY_(eps0_c_Pdy) _syncXY_(eps0_c_Pdz)  \
//_syncXY_(eps0_c_Pcp1x) _syncXY_(eps0_c_Pcp1y) _syncXY_(eps0_c_Pcp1z)  \
//_syncXY_(eps0_c_Pcp2x) _syncXY_(eps0_c_Pcp2y) _syncXY_(eps0_c_Pcp2z) 
// \
//_syncXY_(eps0_c_Rx) _syncXY_(eps0_c_Ry) _syncXY_(eps0_c_Rz)  
#define _syncYZall _syncYZ_(eps0_c_Ex) _syncYZ_(eps0_c_Ey) _syncYZ_(eps0_c_Ez) _syncYZ_(Hx)  _syncYZ_(Hy) _syncYZ_(Hz) \
//_syncYZ_(eps0_c_Pdx) _syncYZ_(eps0_c_Pdy) _syncYZ_(eps0_c_Pdz)  \
//_syncYZ_(eps0_c_Pcp1x) _syncYZ_(eps0_c_Pcp1y) _syncYZ_(eps0_c_Pcp1z)  \
//_syncYZ_(eps0_c_Pcp2x) _syncYZ_(eps0_c_Pcp2y) _syncYZ_(eps0_c_Pcp2z) 
// \
//_syncYZ_(eps0_c_Rx) _syncYZ_(eps0_c_Ry) _syncYZ_(eps0_c_Rz)  
#define _syncZXall _syncZX_(eps0_c_Ex) _syncZX_(eps0_c_Ey) _syncZX_(eps0_c_Ez) _syncZX_(Hx)  _syncZX_(Hy) _syncZX_(Hz) \
//_syncZX_(eps0_c_Pdx) _syncZX_(eps0_c_Pdy) _syncZX_(eps0_c_Pdz)  \
//_syncZX_(eps0_c_Pcp1x) _syncZX_(eps0_c_Pcp1y) _syncZX_(eps0_c_Pcp1z)  \
//_syncZX_(eps0_c_Pcp2x) _syncZX_(eps0_c_Pcp2y) _syncZX_(eps0_c_Pcp2z) 
// \
//_syncZX_(eps0_c_Rx) _syncZX_(eps0_c_Ry) _syncZX_(eps0_c_Rz)  






void syncPadding(void) {
	for (int X = 0; X < _gridDimX; X++)
		for (int Y = 0; Y < _gridDimY; Y++)
			for (int Z = 0; Z < _gridDimZ; Z++) {
				for (int yy = 0; yy<_blockDimY; yy++)
					for (int zz = 0; zz < _blockDimZ; zz++) { _syncXall }
				for (int zz = 0; zz<_blockDimZ; zz++)
					for (int xx = 0; xx < _blockDimX; xx++) { _syncYall }
				for (int xx = 0; xx<_blockDimX; xx++)
					for (int yy = 0; yy < _blockDimY; yy++) { _syncZall }
				for (int zz = 0; zz < _blockDimZ; zz++) { _syncXYall }
				for (int xx = 0; xx < _blockDimX; xx++) { _syncYZall }
				for (int yy = 0; yy < _blockDimY; yy++) { _syncZXall }
			}
	

	if (_PML_PX_X_ > 0)
		for (int Y = 0; Y < _gridDimY; Y++)
			for (int Z = 0; Z < _gridDimZ; Z++)
				for (int yy = 0; yy < _blockDimY; yy++)
					for (int zz = 0; zz < _blockDimZ; zz++) {
						eps0_c_Ex[_INDEY_THREAD(0, Y, Z, 0, yy, zz)] = 0; eps0_c_Ey[_INDEY_THREAD(0, Y, Z, 0, yy, zz)] = 0; eps0_c_Ez[_INDEY_THREAD(0, Y, Z, 0, yy, zz)] = 0;
						Hx[_INDEY_THREAD(0, Y, Z, 0, yy, zz)] = 0; Hy[_INDEY_THREAD(0, Y, Z, 0, yy, zz)] = 0; Hz[_INDEY_THREAD(0, Y, Z, 0, yy, zz)] = 0;
						eps0_c_Ex[_INDEY_THREAD(_gridDimX - 1, Y, Z, ((_DimX - 1)) % (_blockDimX - 2) + 2, yy, zz)] = 0; eps0_c_Ey[_INDEY_THREAD(_gridDimX - 1, Y, Z, ((_DimX - 1)) % (_blockDimX - 2) + 2, yy, zz)] = 0; eps0_c_Ez[_INDEY_THREAD(_gridDimX - 1, Y, Z, ((_DimX - 1)) % (_blockDimX - 2) + 2, yy, zz)] = 0;
						Hx[_INDEY_THREAD(_gridDimX - 1, Y, Z, ((_DimX - 1)) % (_blockDimX - 2) + 2, yy, zz)] = 0; Hy[_INDEY_THREAD(_gridDimX - 1, Y, Z, ((_DimX - 1)) % (_blockDimX - 2) + 2, yy, zz)] = 0; Hz[_INDEY_THREAD(_gridDimX - 1, Y, Z, ((_DimX - 1)) % (_blockDimX - 2) + 2, yy, zz)] = 0;
					}
	if (_PML_PX_Y_ > 0)
		for (int X = 0; X < _gridDimX; X++)
			for (int Z = 0; Z < _gridDimZ; Z++)
				for (int xx = 0; xx < _blockDimX; xx++)
					for (int zz = 0; zz < _blockDimZ; zz++) {
						eps0_c_Ex[_INDEX_THREAD(X, 0, Z, xx, 0, zz)] = 0; eps0_c_Ey[_INDEX_THREAD(X, 0, Z, xx, 0, zz)] = 0; eps0_c_Ez[_INDEX_THREAD(X, 0, Z, xx, 0, zz)] = 0;
						Hx[_INDEX_THREAD(X, 0, Z, xx, 0, zz)] = 0; Hy[_INDEX_THREAD(X, 0, Z, xx, 0, zz)] = 0; Hz[_INDEX_THREAD(X, 0, Z, xx, 0, zz)] = 0;
						eps0_c_Ex[_INDEX_THREAD(X, _gridDimY - 1, Z, xx, ((_DimY - 1)) % (_blockDimY - 2) + 2, zz)] = 0; eps0_c_Ey[_INDEX_THREAD(X, _gridDimY - 1, Z, xx, ((_DimY - 1)) % (_blockDimY - 2) + 2, zz)] = 0; eps0_c_Ez[_INDEX_THREAD(X, _gridDimY - 1, Z, xx, ((_DimY - 1)) % (_blockDimY - 2) + 2, zz)] = 0;
						Hx[_INDEX_THREAD(X, _gridDimY - 1, Z, xx, ((_DimY - 1)) % (_blockDimY - 2) + 2, zz)] = 0; Hy[_INDEX_THREAD(X, _gridDimY - 1, Z, xx, ((_DimY - 1)) % (_blockDimY - 2) + 2, zz)] = 0; Hz[_INDEX_THREAD(X, _gridDimY - 1, Z, xx, ((_DimY - 1)) % (_blockDimY - 2) + 2, zz)] = 0;
					}
	if (_PML_PX_Z_ > 0)
		for (int X = 0; X < _gridDimX; X++)
			for (int Y = 0; Y < _gridDimY; Y++)
				for (int xx = 0; xx < _blockDimX; xx++)
					for (int yy = 0; yy < _blockDimY; yy++) {
						eps0_c_Ex[_INDEX_THREAD(X, Y, 0, xx, yy, 0)] = 0; eps0_c_Ey[_INDEX_THREAD(X, Y, 0, xx, yy, 0)] = 0; eps0_c_Ez[_INDEX_THREAD(X, Y, 0, xx, yy, 0)] = 0;
						Hx[_INDEX_THREAD(X, Y, 0, xx, yy, 0)] = 0; Hy[_INDEX_THREAD(X, Y, 0, xx, yy, 0)] = 0; Hz[_INDEX_THREAD(X, Y, 0, xx, yy, 0)] = 0;
						eps0_c_Ex[_INDEX_THREAD(X, Y, _gridDimZ - 1, xx, yy, ((_DimZ - 1)) % (_blockDimZ - 2) + 2)] = 0; eps0_c_Ey[_INDEX_THREAD(X, Y, _gridDimZ - 1, xx, yy, ((_DimZ - 1)) % (_blockDimZ - 2) + 2)] = 0; eps0_c_Ez[_INDEX_THREAD(X, Y, _gridDimZ - 1, xx, yy, ((_DimZ - 1)) % (_blockDimZ - 2) + 2)] = 0;
						Hx[_INDEX_THREAD(X, Y, _gridDimZ - 1, xx, yy, ((_DimZ - 1)) % (_blockDimZ - 2) + 2)] = 0; Hy[_INDEX_THREAD(X, Y, _gridDimZ - 1, xx, yy, ((_DimZ - 1)) % (_blockDimZ - 2) + 2)] = 0; Hz[_INDEX_THREAD(X, Y, _gridDimZ - 1, xx, yy, ((_DimZ - 1)) % (_blockDimZ - 2) + 2)] = 0;
					}
}


int snapshot(char* filename)
{
	char filenameFull[256];


	
	sprintf(filenameFull, "Ex_Y0_%s.txt", filename);
	int Y = _DimY / 2;
	FILE *f = fopen(filenameFull, "w");
	for (int Z = 0; Z < _DimZ; Z++) {
		//if (X%_blockDimX == 0) { for (int Z = 0; Z < _DimZ + _gridDimZ - 1 ; Z++) { fprintf(f,"%+04.3e \t ", TEMP[_INDEX_XYZ(-1, Y, Z)]);  } 	fprintf(f,"\n");}
		for (int X = 0; X < _DimX; X++) {
			//if (Z%_blockDimZ == 0) { fprintf(f,"%04.3f \t", TEMP[_INDEX_XYZ(X, Y, -1)]);}
			fprintf(f, "%+04.3e\t", eps0_c_Ex[_INDEX_XYZ(X, Y, Z)]);
			//if (Z%_blockDimZ == _blockDimZ-1) { fprintf(f,"%+04.3f\t ", TEMP[_INDEX_XYZ(_blockDimX, Y, Z)]); }
		}
		fprintf(f, "\n");
		//if (X%_blockDimX == _blockDimX -1 ){ for (int Z = 0; Z < _DimZ; Z++) 	{	fprintf(f,"%+04.3f\t ", eps0_c_Ex[_INDEX_XYZ(_blockDimX, Y, Z)]);	}fprintf(f,"\n");}
	}
	fclose(f);
	sprintf(filenameFull, "Ey_Y0_%s.txt", filename);
	f = fopen(filenameFull, "w");
	for (int Z = 0; Z < _DimZ; Z++) {
		//if (X%_blockDimX == 0) { for (int Z = 0; Z < _DimZ + _gridDimZ - 1 ; Z++) { fprintf(f,"%+04.3e \t ", TEMP[_INDEX_XYZ(-1, Y, Z)]);  } 	fprintf(f,"\n");}
		for (int X = 0; X < _DimX; X++) {
			//if (Z%_blockDimZ == 0) { fprintf(f,"%04.3f \t", TEMP[_INDEX_XYZ(X, Y, -1)]);}
			fprintf(f, "%+04.3e\t", eps0_c_Ey[_INDEX_XYZ(X, Y, Z)]);
			//if (Z%_blockDimZ == _blockDimZ-1) { fprintf(f,"%+04.3f\t ", TEMP[_INDEX_XYZ(_blockDimX, Y, Z)]); }
		}
		fprintf(f, "\n");
		//if (X%_blockDimX == _blockDimX -1 ){ for (int Z = 0; Z < _DimZ; Z++) 	{	fprintf(f,"%+04.3f\t ", eps0_c_Ex[_INDEX_XYZ(_blockDimX, Y, Z)]);	}fprintf(f,"\n");}
	}
	fclose(f);
	sprintf(filenameFull, "Ez_Y0_%s.txt", filename);
	f = fopen(filenameFull, "w");
	for (int Z = 0; Z < _DimZ; Z++) {
		//if (X%_blockDimX == 0) { for (int Z = 0; Z < _DimZ + _gridDimZ - 1 ; Z++) { fprintf(f,"%+04.3e \t ", TEMP[_INDEX_XYZ(-1, Y, Z)]);  } 	fprintf(f,"\n");}
		for (int X = 0; X < _DimX; X++) {
			//if (Z%_blockDimZ == 0) { fprintf(f,"%04.3f \t", TEMP[_INDEX_XYZ(X, Y, -1)]);}
			fprintf(f, "%+04.3e\t", eps0_c_Ez[_INDEX_XYZ(X, Y, Z)]);
			//if (Z%_blockDimZ == _blockDimZ-1) { fprintf(f,"%+04.3f\t ", TEMP[_INDEX_XYZ(_blockDimX, Y, Z)]); }
		}
		fprintf(f, "\n");
		//if (X%_blockDimX == _blockDimX -1 ){ for (int Z = 0; Z < _DimZ; Z++) 	{	fprintf(f,"%+04.3f\t ", eps0_c_Ex[_INDEX_XYZ(_blockDimX, Y, Z)]);	}fprintf(f,"\n");}
	}
	fclose(f);
	
	unsigned char* image;
	unsigned x, y, z;
	unsigned width, height;
	unsigned error;


	printf("snapshot : X=%d\n", _DimX / 2);
	int X = _DimX / 2;
	width = _DimY, height = _DimZ;
	image = malloc(width * height * 4);
	for (z = 0; z < height; z++)
		for (y = 0; y < width; y++)
		{
			int surf_x, surf_y, surf_z;
			int surf_index = _SURF_INDEX_XYZ(X, y, z);
			_SET_SURF_XYZ_INDEX(surf_index);
			int offset = _INDEX_XYZ(X, y, z);
			int value = 0;

			//if (surf_index !=-1) value = (FT_eps0cE[_SURF_INDEX_XYZ(X, y, z)][1][0][0]* FT_eps0cE[_SURF_INDEX_XYZ(X, y, z)][1][0][0]+ FT_eps0cE[_SURF_INDEX_XYZ(X, y, z)][1][0][1]* FT_eps0cE[_SURF_INDEX_XYZ(X, y, z)][1][0][1]) * 255.0f * 5000.0f * 5000.0f;
			if (surf_index != -1) value = 255.0f * 20000.0f * 20000.0f*(0
				+ (FT_eps0cE[surf_index][0][0][0]) * (FT_eps0cE[surf_index][0][0][0])
				+ (FT_eps0cE[surf_index][0][0][1]) * (FT_eps0cE[surf_index][0][0][1])
				+ (FT_eps0cE[surf_index][0][1][0]) * (FT_eps0cE[surf_index][0][1][0])
				+ (FT_eps0cE[surf_index][0][1][1]) * (FT_eps0cE[surf_index][0][1][1])
				+ (FT_eps0cE[surf_index][0][2][0]) * (FT_eps0cE[surf_index][0][2][0])
				+ (FT_eps0cE[surf_index][0][2][1]) * (FT_eps0cE[surf_index][0][2][1])
				);
			//int value = (((mask[offset] & (0b1111 << 4)) >> 4)) *50.0f ;
			value = value > 255 ? 255 : value;
			value = value < -255 ? -255 : value;
			int val2 = sqrtf(
				eps0_c_Ex[offset] * eps0_c_Ex[offset] +
				eps0_c_Ey[offset] * eps0_c_Ey[offset] +
				eps0_c_Ez[offset] * eps0_c_Ez[offset]
			) * 255.0f;
			val2 = val2 > 255 ? 255 : val2;
			val2 = val2 < -255 ? -255 : val2;
			image[4 * width * (height - 1 - z) + 4 * y + 0] = (unsigned char)(value>0 ? value : 0);
			image[4 * width * (height - 1 - z) + 4 * y + 1] = (unsigned char)(val2<0 ? -val2 : 0);
			image[4 * width * (height - 1 - z) + 4 * y + 2] = (unsigned char)(val2>0 ? val2 : 0);
			image[4 * width * (height - 1 - z) + 4 * y + 3] = 255;
		}
	sprintf(filenameFull, "E2_X0_%s.png", filename);
	error = lodepng_encode32_file(filenameFull, image, width, height);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

	free(image);


	

	printf("snapshot : Z=%d\n", _DimZ / 2);
	int Z = _DimZ / 2;
	width = _DimX, height = _DimY;
	image = malloc(width * height * 4);
	for (y = 0; y < height; y++)
		for (x = 0; x < width; x++)
		{
			int surf_x, surf_y, surf_z;
			int surf_index = _SURF_INDEX_XYZ(x, y, Z);
			_SET_SURF_XYZ_INDEX(surf_index);
			int offset = _INDEX_XYZ(x, y, Z);
			int value = 0;

			//if (surf_index !=-1) value = (FT_eps0cE[_SURF_INDEX_XYZ(X, y, z)][1][0][0]* FT_eps0cE[_SURF_INDEX_XYZ(X, y, z)][1][0][0]+ FT_eps0cE[_SURF_INDEX_XYZ(X, y, z)][1][0][1]* FT_eps0cE[_SURF_INDEX_XYZ(X, y, z)][1][0][1]) * 255.0f * 5000.0f * 5000.0f;
			if (surf_index != -1) value = 255.0f * 20000.0f * 20000.0f*(0
				+ (FT_eps0cE[surf_index][0][0][0]) * (FT_eps0cE[surf_index][0][0][0])
				+ (FT_eps0cE[surf_index][0][0][1]) * (FT_eps0cE[surf_index][0][0][1])
				+ (FT_eps0cE[surf_index][0][1][0]) * (FT_eps0cE[surf_index][0][1][0])
				+ (FT_eps0cE[surf_index][0][1][1]) * (FT_eps0cE[surf_index][0][1][1])
				+ (FT_eps0cE[surf_index][0][2][0]) * (FT_eps0cE[surf_index][0][2][0])
				+ (FT_eps0cE[surf_index][0][2][1]) * (FT_eps0cE[surf_index][0][2][1])
				);
			//int value = (((mask[offset] & (0b1111 << 4)) >> 4)) *50.0f ;
			value = value > 255 ? 255 : value;
			value = value < -255 ? -255 : value;
			int val2 = sqrtf(
				eps0_c_Ex[offset] * eps0_c_Ex[offset] +
				eps0_c_Ey[offset] * eps0_c_Ey[offset] +
				eps0_c_Ez[offset] * eps0_c_Ez[offset]
			) * 255.0f;
			val2 = val2 > 255 ? 255 : val2;
			val2 = val2 < -255 ? -255 : val2;
			image[4 * width * (height - 1 - y) + 4 * x + 0] = (unsigned char)(value>0 ? value : 0);
			image[4 * width * (height - 1 - y) + 4 * x + 1] = (unsigned char)(val2<0 ? -val2 : 0);
			image[4 * width * (height - 1 - y) + 4 * x + 2] = (unsigned char)(val2>0 ? val2 : 0);
			image[4 * width * (height - 1 - y) + 4 * x + 3] = 255;
		}
	sprintf(filenameFull, "E2_Z0_%s.png", filename);
	error = lodepng_encode32_file(filenameFull, image, width, height);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

	free(image);
	
	return 0;
}



void snapshotStructure() {

	printf("saving structure image\n");

	int Y = _DimY / 2;
	unsigned width = _DimX, height = _DimZ;
	unsigned char* image = malloc(width * height * 4);
	int z, x;
	for (z = 0; z < height; z++) {
		for (x = 0; x < width; x++)
		{
			int surf_x, surf_y, surf_z;
			int surf_index = _SURF_INDEX_XYZ(x, Y, z);
			_SET_SURF_XYZ_INDEX(surf_index);
			int offset = _INDEX_XYZ(x, Y, z);
			int value = 0;
			int val2 = 0;

			if (((mask[offset] >> 4) & 0b1111) > 0) { value = 255; }
			else { val2 = 255.0f / sqrtf(eps_r_inv[offset]) * 0.5f; }

			value = value > 255 ? 255 : value;
			value = value < -255 ? -255 : value;
			val2 = val2 > 255 ? 255 : val2;
			val2 = val2 < -255 ? -255 : val2;
			image[4 * width * (height - z - 1) + 4 * x + 0] = (unsigned char)(val2 > 0 ? val2 : 0);
			image[4 * width * (height - z - 1) + 4 * x + 1] = (unsigned char)(value > 0 ? value : 0);
			image[4 * width * (height - z - 1) + 4 * x + 2] = (unsigned char)(val2 < 0 ? -val2 : 0);
			image[4 * width * (height - z - 1) + 4 * x + 3] = 255;
		}
	}
	unsigned error = lodepng_encode32_file("structure_Y.png", image, width, height);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

	free(image);

	int Z = _DimZ / 2;
	width = _DimX; height = _DimY;
	unsigned char* image2 = malloc(width * height * 4);
	int y;//  , x;
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++)
		{
			int surf_x, surf_y, surf_z;
			int surf_index = _SURF_INDEX_XYZ(x, y, Z);
			_SET_SURF_XYZ_INDEX(surf_index);
			int offset = _INDEX_XYZ(x, y, Z);
			int value = 0;
			int val2 = 0;

			if (((mask[offset] >> 4) & 0b1111) > 0) { value = 255; }
			else { val2 = 255.0f / sqrtf(eps_r_inv[offset]) * 0.5f; }

			value = value > 255 ? 255 : value;
			value = value < -255 ? -255 : value;
			val2 = val2 > 255 ? 255 : val2;
			val2 = val2 < -255 ? -255 : val2;
			image2[4 * width * (y)+4 * x + 0] = (unsigned char)(val2 > 0 ? val2 : 0);
			image2[4 * width * (y)+4 * x + 1] = (unsigned char)(value > 0 ? value : 0);
			image2[4 * width * (y)+4 * x + 2] = (unsigned char)(val2 < 0 ? -val2 : 0);
			image2[4 * width * (y)+4 * x + 3] = 255;
		}
	}
	error = lodepng_encode32_file("structure_Z.png", image2, width, height);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

	free(image2);

	printf("saved\n");

}