/*
 * sfcalc_gpu.cu
 * =============
 * GPU-accelerated structure factor calculation via real-space atom spreading.
 *
 * Algorithm (matches gemmi sfcalc):
 *   For each atom i with element Z, B-factor Bi, position (xi,yi,zi):
 *     Use IT92 5-Gaussian form factor f(s) = Σ_k a_k exp(-b_k s²) + c
 *     where s = sin(θ)/λ = 1/(2d)
 *
 *   For each Gaussian component k:
 *     Effective B  : s_k = b_k + Bi
 *     Real-space Gaussian width on grid (in fractional): sigma_k = sqrt(s_k) / (2*pi)
 *     Density at grid point r: rho += a_k * (pi/s_k)^(3/2) * exp(-pi^2 * d^2 / s_k)
 *     where d is the distance from atom to grid point.
 *
 *   The c term (constant in reciprocal space) contributes only to F(0,0,0).
 *
 * Compiled as a shared library:
 *   nvcc -O3 -arch=sm_70 -shared -Xcompiler -fPIC \
 *        -lcufft sfcalc_gpu.cu -o sfcalc_gpu.so
 *
 * Exported function:
 *   int spread_atoms(
 *       int natoms, float *x, float *y, float *z, float *B,
 *       int *elem,  // element index (0=C,1=H,2=N,3=O,4=P,5=S)
 *       int nx, int ny, int nz,
 *       float ax, float ay, float az,   // cell edge lengths in Angstroms
 *       float *map_out,                 // output: nx*ny*nz float32, Fortran order (z,y,x fast)
 *       float *Freal_out,               // output: (nx/2+1)*ny*nz complex (r2c FFT), NULL = skip
 *       float Bmax_skip                 // skip atoms with B > Bmax_skip (use 0 to disable)
 *   )
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>

// ---------------------------------------------------------------------------
// IT92 5-Gaussian form factors for C, H, N, O, P, S
// f(s) = sum_k a_k exp(-b_k s^2) + c,   s = sin(theta)/lambda
// Source: International Tables for Crystallography, Vol. C, Table 6.1.1.4
// ---------------------------------------------------------------------------
// Element order: C=0, H=1, N=2, O=3, P=4, S=5

static const int N_ELEM = 6;
static const int N_GAUSS = 4;  // a1-a4, b1-b4, c (IT92 uses 4+1 Gaussians)

__constant__ float d_a[N_ELEM][N_GAUSS] = {
    // C
    {2.31000f, 1.02000f, 1.58860f, 0.86500f},
    // H
    {0.493002f, 0.322912f, 0.140191f, 0.04081f},
    // N
    {12.2126f, 3.13220f, 2.01250f, 1.16630f},
    // O
    {3.04850f, 2.28680f, 1.54630f, 0.86700f},
    // P
    {6.43450f, 4.17910f, 1.78000f, 1.49080f},
    // S
    {6.90530f, 5.20340f, 1.43790f, 1.58630f},
};

__constant__ float d_b[N_ELEM][N_GAUSS] = {
    // C
    {20.8439f, 10.2075f, 0.56870f, 51.6512f},
    // H
    {10.5109f, 26.1257f, 3.14236f, 57.7997f},
    // N
    {0.00570f, 9.89330f, 28.9975f, 0.58260f},
    // O
    {13.2771f, 5.70110f, 0.32390f, 32.9089f},
    // P
    {1.90670f, 27.1570f, 0.52600f, 68.1645f},
    // S
    {1.46790f, 22.2151f, 0.25360f, 56.1720f},
};

__constant__ float d_c[N_ELEM] = {
    0.2156f,   // C
    0.003038f, // H
   -11.529f,   // N  (negative!)
    0.2508f,   // O
    1.11490f,  // P
    0.86630f,  // S
};

// ---------------------------------------------------------------------------
// Kernel 1: precompute per-atom Gaussian prefactors and cutoff radii.
// One thread per atom; results stored in flat device arrays.
// ---------------------------------------------------------------------------
#define NG1 (N_GAUSS + 1)   // 5 terms per atom (4 Gaussians + c term)

__global__ void precompute_atoms(
    int natoms,
    const float* __restrict__ B,
    const int*   __restrict__ elem,
    float* __restrict__ pre_all,   // [natoms, NG1]  amplitude prefactors
    float* __restrict__ wk_all,    // [natoms, NG1]  exponent coefficients
    float* __restrict__ rcut2      // [natoms]        cutoff radius^2 (Angstroms^2)
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;

    const float PI  = 3.14159265358979f;
    const float PI2 = PI * PI;

    float Bi = B[i];
    int   ei = elem[i];

    float b_max = 0.0f;
    for (int k = 0; k < N_GAUSS; k++) {
        float sk = d_b[ei][k] + Bi;
        if (sk > b_max) b_max = sk;
        pre_all[i*NG1 + k] = d_a[ei][k] * (4.0f*PI/sk) * sqrtf(4.0f*PI/sk);
        wk_all [i*NG1 + k] = 4.0f * PI2 / sk;
    }
    // c term: treat as Gaussian with effective b_c = 0, so s_c = Bi
    if (Bi > 0.0f) {
        pre_all[i*NG1 + N_GAUSS] = d_c[ei] * (4.0f*PI/Bi) * sqrtf(4.0f*PI/Bi);
        wk_all [i*NG1 + N_GAUSS] = 4.0f * PI2 / Bi;
    } else {
        pre_all[i*NG1 + N_GAUSS] = 0.0f;
        wk_all [i*NG1 + N_GAUSS] = 0.0f;
    }
    // cutoff: exp(-4pi^2 r^2 / b_max) < 1e-5  =>  r^2 < 5*ln(10)/(4pi^2) * b_max
    rcut2[i] = 0.2914f * b_max;
}

// ---------------------------------------------------------------------------
// Kernel 2: one thread per atom, scatters Gaussian density to nearby voxels.
// Uses atomicAdd so voxels can receive contributions from multiple atoms.
// Complexity: O(N_atoms * (rcut/pixel)^3) instead of O(N_atoms * N_voxels),
// which is orders of magnitude faster when atoms are sparse relative to voxels.
// Uses __expf (fast single-precision exp, ~1 ULP error) for speed.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Kernel 2: one thread per atom, general metric tensor.
//
// For orthogonal cells: g12=g13=g23=0, rfx=1/ax, rfy=1/ay, rfz=1/az,
//   schur33=az^2.  Performance identical to the old kernel.
//
// For non-orthogonal cells (e.g. hexagonal H32 with gamma=120°):
//   g12 = ax*ay*cos(gamma), etc.  rfx,rfy,rfz are derived from the inverse
//   metric tensor so the bounding box is still tight.  schur33 = G_Schur[2,2]
//   is the minimum r^2 contribution from dfz alone, enabling the z early exit.
// ---------------------------------------------------------------------------
__global__ void spread_kernel_atom(
    int natoms,
    int nx, int ny, int nz,
    float g11, float g22, float g33,   // diagonal metric tensor (= ax^2,ay^2,az^2)
    float g12, float g13, float g23,   // off-diagonal metric tensor
    float schur33,                     // Schur complement [2,2]; z-loop lower bound
    float rfx, float rfy, float rfz,   // bounding box: max fractional extent per unit rc
    const float* __restrict__ xf,
    const float* __restrict__ yf,
    const float* __restrict__ zf,
    const float* __restrict__ pre_all,
    const float* __restrict__ wk_all,
    const float* __restrict__ rcut2_arr,
    float* __restrict__ rho
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natoms) return;

    float xi = xf[i], yi = yf[i], zi = zf[i];
    float rc2 = rcut2_arr[i];
    float rc  = sqrtf(rc2);

    // Bounding box in unbounded grid-index space (periodic wrap applied below)
    int ix0 = (int)floorf((xi - rc * rfx) * (float)nx) - 1;
    int ix1 = (int)ceilf ((xi + rc * rfx) * (float)nx) + 1;
    int iy0 = (int)floorf((yi - rc * rfy) * (float)ny) - 1;
    int iy1 = (int)ceilf ((yi + rc * rfy) * (float)ny) + 1;
    int iz0 = (int)floorf((zi - rc * rfz) * (float)nz) - 1;
    int iz1 = (int)ceilf ((zi + rc * rfz) * (float)nz) + 1;

    const float* pre = pre_all + i * NG1;
    const float* wk  = wk_all  + i * NG1;

    for (int kz = iz0; kz <= iz1; kz++) {
        float dfz = (float)kz / (float)nz - zi;
        // schur33*dfz^2 = min r^2 over all dfx,dfy for this dfz
        if (schur33 * dfz * dfz >= rc2) continue;
        int gz = ((kz % nz) + nz) % nz;

        for (int ky = iy0; ky <= iy1; ky++) {
            float dfy = (float)ky / (float)ny - yi;
            int gy = ((ky % ny) + ny) % ny;

            for (int kx = ix0; kx <= ix1; kx++) {
                float dfx = (float)kx / (float)nx - xi;
                // Full metric tensor distance
                float r2 = g11*dfx*dfx + g22*dfy*dfy + g33*dfz*dfz
                         + 2.0f*(g12*dfx*dfy + g13*dfx*dfz + g23*dfy*dfz);
                if (r2 >= rc2) continue;
                int gx = ((kx % nx) + nx) % nx;

                float val = 0.f;
                for (int k = 0; k < NG1; k++)
                    val += pre[k] * __expf(-wk[k] * r2);
                atomicAdd(&rho[gx + nx * (gy + ny * gz)], val);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: deinterleave cuFFT complex output into separate real/imag arrays.
// Avoids malloc'ing a large host buffer (whose pages would fault on first
// cudaMemcpy write), by doing the split on the device then copying two
// contiguous float arrays to pre-faulted host buffers.
// ---------------------------------------------------------------------------
__global__ void deinterleave_kernel(size_t n, const cufftComplex* __restrict__ src,
                                    float* __restrict__ re, float* __restrict__ im)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { re[i] = src[i].x; im[i] = src[i].y; }
}

// ---------------------------------------------------------------------------
// Exported C function
// ---------------------------------------------------------------------------
extern "C" {

int spread_and_fft(
    int natoms,
    float *x_orth,   // orthogonal (Cartesian) coordinates in Angstroms
    float *y_orth,
    float *z_orth,
    float *B,
    int   *elem,
    int nx, int ny, int nz,
    float ax, float ay, float az,         // cell edge lengths in Angstroms
    float alpha_deg, float beta_deg, float gamma_deg,  // cell angles in degrees
    float Bmax_skip,         // skip atoms with B > Bmax_skip (0 = keep all)
    float  *map_out,         // output real-space map nx*ny*nz float32 (may be NULL)
    float  *F_real,          // output Re(F), size (nx/2+1)*ny*nz float32 (may be NULL)
    float  *F_imag           // output Im(F), same size float32 (may be NULL)
)
{
    // ----- Fractionalization matrix (general triclinic cell) -----
    // Standard orientation: a along x, b in xy-plane.
    // M_inv converts Cartesian -> fractional:
    //   xf = f00*X + f01*Y + f02*Z
    //   yf =         f11*Y + f12*Z
    //   zf =                 f22*Z
    const float PI = 3.14159265358979f;
    float ca = cosf(alpha_deg * PI / 180.0f);
    float cb = cosf(beta_deg  * PI / 180.0f);
    float cg = cosf(gamma_deg * PI / 180.0f);
    float sg = sinf(gamma_deg * PI / 180.0f);
    // Volume fraction: V = sqrt(1 - ca^2 - cb^2 - cg^2 + 2*ca*cb*cg)
    float V   = sqrtf(1.0f - ca*ca - cb*cb - cg*cg + 2.0f*ca*cb*cg);
    float f00 = 1.0f / ax;
    float f01 = -cg / (ax * sg);
    float f02 = (ca*cg - cb) / (ax * V * sg);
    float f11 = 1.0f / (ay * sg);
    float f12 = (cb*cg - ca) / (ay * V * sg);
    float f22 = sg / (az * V);

    // ----- Metric tensor G  (G_ij = a_i · a_j) -----
    float g11 = ax*ax,             g22 = ay*ay,             g33 = az*az;
    float g12 = ax*ay*cg,          g13 = ax*az*cb,          g23 = ay*az*ca;

    // ----- Inverse metric tensor diagonals (for bounding box rfx,rfy,rfz) -----
    // det of 2x2 upper-left block
    float det_ab = g11*g22 - g12*g12;
    // det of full G
    float det_G  = g33*det_ab - g22*g13*g13 + 2.0f*g12*g13*g23 - g11*g23*g23;
    float Ginv11 = (g22*g33 - g23*g23) / det_G;
    float Ginv22 = (g11*g33 - g13*g13) / det_G;
    float Ginv33 = det_ab / det_G;
    float rfx = sqrtf(Ginv11);   // max fractional extent per unit rc along a
    float rfy = sqrtf(Ginv22);   // ... along b
    float rfz = sqrtf(Ginv33);   // ... along c

    // ----- Schur complement [2,2]: lower bound on r^2 per dfz^2 -----
    // r^2_min(dfz) = schur33 * dfz^2  allows z-loop early exit
    float schur33 = g33 - (g22*g13*g13 - 2.0f*g12*g13*g23 + g11*g23*g23) / det_ab;

    // ----- Fractionalize atoms, filter high-B -----
    float *xf_all = (float*)malloc(natoms * sizeof(float));
    float *yf_all = (float*)malloc(natoms * sizeof(float));
    float *zf_all = (float*)malloc(natoms * sizeof(float));
    float *B_all  = (float*)malloc(natoms * sizeof(float));
    int   *el_all = (int*)  malloc(natoms * sizeof(int));

    int nkeep = 0;
    for (int i = 0; i < natoms; i++) {
        if (Bmax_skip > 0 && B[i] > Bmax_skip) continue;
        float X = x_orth[i], Y = y_orth[i], Z = z_orth[i];
        float xf = f00*X + f01*Y + f02*Z;
        float yf =         f11*Y + f12*Z;
        float zf =                 f22*Z;
        // Wrap into [0,1)
        xf_all[nkeep] = xf - floorf(xf);
        yf_all[nkeep] = yf - floorf(yf);
        zf_all[nkeep] = zf - floorf(zf);
        B_all [nkeep] = B[i];
        el_all[nkeep] = elem[i];
        nkeep++;
    }
    fprintf(stderr, "  spread_and_fft: %d atoms kept (of %d), grid %dx%dx%d\n",
            nkeep, natoms, nx, ny, nz);

    // Allocate device memory
    size_t grid_size = (size_t)nx * ny * nz;
    float  *d_xf, *d_yf, *d_zf, *d_B;
    float  *d_rho;
    int    *d_el;

    cudaMalloc(&d_xf,  nkeep * sizeof(float));
    cudaMalloc(&d_yf,  nkeep * sizeof(float));
    cudaMalloc(&d_zf,  nkeep * sizeof(float));
    cudaMalloc(&d_B,   nkeep * sizeof(float));
    cudaMalloc(&d_el,  nkeep * sizeof(int));
    cudaMalloc(&d_rho, grid_size * sizeof(float));

    cudaMemset(d_rho, 0, grid_size * sizeof(float));

    cudaMemcpy(d_xf, xf_all, nkeep * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_yf, yf_all, nkeep * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_zf, zf_all, nkeep * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,  B_all,  nkeep * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_el, el_all, nkeep * sizeof(int),   cudaMemcpyHostToDevice);

    free(xf_all); free(yf_all); free(zf_all); free(B_all); free(el_all);

    // --- Step 1: precompute per-atom Gaussian data on device ---
    float *d_pre, *d_wk, *d_rcut2;
    cudaMalloc(&d_pre,   nkeep * NG1 * sizeof(float));
    cudaMalloc(&d_wk,    nkeep * NG1 * sizeof(float));
    cudaMalloc(&d_rcut2, nkeep *       sizeof(float));

    {
        int t = 128, b = (nkeep + t - 1) / t;
        precompute_atoms<<<b, t>>>(nkeep, d_B, d_el, d_pre, d_wk, d_rcut2);
        cudaDeviceSynchronize();
    }

    // --- Step 2: atom-centric spreading ---
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);

    {
        int threads = 128;
        int blocks  = (nkeep + threads - 1) / threads;
        spread_kernel_atom<<<blocks, threads>>>(
            nkeep, nx, ny, nz,
            g11, g22, g33, g12, g13, g23,
            schur33, rfx, rfy, rfz,
            d_xf, d_yf, d_zf,
            d_pre, d_wk, d_rcut2,
            d_rho
        );
    }

    cudaEventRecord(t1);
    cudaDeviceSynchronize();

    float ms_spread;
    cudaEventElapsedTime(&ms_spread, t0, t1);
    fprintf(stderr, "  Spreading time: %.1f ms\n", ms_spread);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR spread_kernel_atom: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaFree(d_pre); cudaFree(d_wk); cudaFree(d_rcut2);

    // Copy map to host if requested
    if (map_out != NULL) {
        cudaMemcpy(map_out, d_rho, grid_size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // FFT if output requested
    if (F_real != NULL || F_imag != NULL) {
        // D2Z FFT: input nx*ny*nz float64, output (nx/2+1)*ny*nz complex128
        // cuFFT convention: FFTW layout, nx*ny*nz real -> (nx/2+1)*ny*nz complex
        // Our grid is stored with X fastest (index = ix + nx*(iy+ny*iz))
        // cuFFT d2z with n={nz,ny,nx} (slowest to fastest) matches this

        cufftHandle plan;
        int dims[3] = {nz, ny, nx};
        cufftResult cr = cufftPlanMany(
            &plan, 3, dims,
            NULL, 1, 0,  // input: contiguous
            NULL, 1, 0,  // output: contiguous
            CUFFT_R2C, 1
        );
        if (cr != CUFFT_SUCCESS) {
            fprintf(stderr, "ERROR cufftPlanMany: %d\n", cr);
            return -2;
        }

        size_t fft_out_size = (size_t)(nx/2+1) * ny * nz;
        cufftComplex *d_Fc;
        cudaMalloc(&d_Fc, fft_out_size * sizeof(cufftComplex));

        cudaEventRecord(t0);
        cr = cufftExecR2C(plan, d_rho, d_Fc);
        cudaEventRecord(t1);
        cudaDeviceSynchronize();

        float ms_fft;
        cudaEventElapsedTime(&ms_fft, t0, t1);
        fprintf(stderr, "  FFT time: %.1f ms\n", ms_fft);

        if (cr != CUFFT_SUCCESS) {
            fprintf(stderr, "ERROR cufftExecR2C: %d\n", cr);
            return -3;
        }

        // Deinterleave on GPU then copy two contiguous float arrays to the
        // pre-allocated (pre-faulted) host buffers F_real / F_imag.
        // This avoids malloc'ing a large host h_Fc buffer whose pages would
        // cause first-access page faults inside cudaMemcpy (~1 µs/page).
        if (F_real != NULL && F_imag != NULL) {
            float *d_Fr, *d_Fi;
            cudaMalloc(&d_Fr, fft_out_size * sizeof(float));
            cudaMalloc(&d_Fi, fft_out_size * sizeof(float));

            {
                int dt = 256;
                int bt = (int)((fft_out_size + dt - 1) / dt);
                deinterleave_kernel<<<bt, dt>>>(fft_out_size, d_Fc, d_Fr, d_Fi);
            }

            cudaDeviceSynchronize();
            cudaMemcpy(F_real, d_Fr, fft_out_size * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(F_imag, d_Fi, fft_out_size * sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(d_Fr);
            cudaFree(d_Fi);
        }

        cudaFree(d_Fc);
        cufftDestroy(plan);

    }

    cudaFree(d_xf); cudaFree(d_yf); cudaFree(d_zf);
    cudaFree(d_B);  cudaFree(d_el); cudaFree(d_rho);
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    return nkeep;
}

} // extern "C"
