/*
 * sfcalc_gpu_collapse.cpp
 * =======================
 * C++ port of sfcalc_gpu_collapse.py — same algorithm, no Python overhead.
 * No CCP4 runtime dependency: uses vendored gemmi/symmetry.hpp (header-only,
 * MPL 2.0) for space-group tables; PDB reading and MTZ writing are hand-rolled.
 *
 * Usage: same command-line interface as the Python script.
 *   ./sfcalc_gpu_collapse  input.pdb  [dmin=1.5]  [rate=2.5]  [sg=P1]
 *       [super_mult=1,1,1]  [outmtz=collapsed.mtz]  [outI=supercell_I.mtz]
 *       [outmap=]  [bmax=0]  [noise=0.01]  [lib=sfcalc_gpu.so]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdarg>
#include <dlfcn.h>

// Vendored gemmi headers (header-only, MPL 2.0, no libgemmi_cpp needed).
// Only symmetry.hpp + fail.hpp are needed; everything else is hand-rolled.
#include "include/gemmi/symmetry.hpp"

// ---------------------------------------------------------------------------
// GPU library function (extern "C" in sfcalc_gpu.cu)
// ---------------------------------------------------------------------------
typedef int (*SpreadAndFft_fn)(
    int natoms,
    float *x, float *y, float *z, float *B, int *elem,
    int nx, int ny, int nz,
    float ax, float ay, float az,
    float alpha, float beta, float gamma,
    float Bmax_skip,
    float *map_out, float *F_real, float *F_imag
);

// ---------------------------------------------------------------------------
// Minimal UnitCell (replaces gemmi::UnitCell)
// ---------------------------------------------------------------------------
struct UnitCell {
    double a, b, c;
    double alpha, beta, gamma;  // degrees

    double volume() const {
        double ca = cos(alpha * M_PI/180.0);
        double cb = cos(beta  * M_PI/180.0);
        double cg = cos(gamma * M_PI/180.0);
        return a*b*c * sqrt(1.0 - ca*ca - cb*cb - cg*cg + 2.0*ca*cb*cg);
    }

    // Returns the reciprocal-space unit cell (a*, b*, c*, alpha*, beta*, gamma*)
    UnitCell reciprocal() const {
        double ca = cos(alpha * M_PI/180.0), sa = sin(alpha * M_PI/180.0);
        double cb = cos(beta  * M_PI/180.0), sb = sin(beta  * M_PI/180.0);
        double cg = cos(gamma * M_PI/180.0), sg = sin(gamma * M_PI/180.0);
        double V = volume();
        UnitCell r;
        r.a = b*c*sa / V;
        r.b = a*c*sb / V;
        r.c = a*b*sg / V;
        // cos(alpha*) = (cos(beta)*cos(gamma) - cos(alpha)) / (sin(beta)*sin(gamma))
        r.alpha = acos((cb*cg - ca) / (sb*sg)) * 180.0/M_PI;
        r.beta  = acos((ca*cg - cb) / (sa*sg)) * 180.0/M_PI;
        r.gamma = acos((ca*cb - cg) / (sa*sb)) * 180.0/M_PI;
        return r;
    }
};

// ---------------------------------------------------------------------------
// Minimal PDB parser (replaces gemmi::read_structure_file)
// Reads CRYST1 for cell parameters; ATOM/HETATM for atoms.
// Element is taken from cols 77-78 if present, else derived from atom name.
// ---------------------------------------------------------------------------
static int elem_idx(const char* sym) {
    // skip leading spaces
    while (*sym == ' ') sym++;
    if (sym[0]=='C' && sym[1]!='L' && sym[1]!='A' && sym[1]!='O' &&
        sym[1]!='R' && sym[1]!='S' && sym[1]!='U' && sym[1]!='D' &&
        sym[1]!='E' && sym[1]!='F' && sym[1]!='N' && sym[1]!='l' &&
        (sym[1]==' '||sym[1]=='\0'||sym[1]>='a'))   return 0; // C
    if (sym[0]=='H') return 1;  // H (also covers HG etc. — default C below)
    if (sym[0]=='N') return 2;  // N
    if (sym[0]=='O') return 3;  // O
    if (sym[0]=='P') return 4;  // P
    if (sym[0]=='S') return 5;  // S
    return 0;  // default C
}

// Safe column extraction (0-indexed start, length) into null-terminated buf
static void pdb_col(const char* line, int start, int len, char* buf) {
    int llen = (int)strlen(line);
    for (int i = 0; i < len; i++)
        buf[i] = (start+i < llen) ? line[start+i] : ' ';
    buf[len] = '\0';
}

static bool read_pdb(const char* path,
                     UnitCell& cell,
                     std::vector<float>& xs, std::vector<float>& ys,
                     std::vector<float>& zs, std::vector<float>& Bs,
                     std::vector<int>& els)
{
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return false; }

    bool got_cryst1 = false;
    char line[256];
    char buf[32];

    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "CRYST1", 6) == 0) {
            // a:7-15, b:16-24, c:25-33, alpha:34-40, beta:41-47, gamma:48-54
            pdb_col(line, 6, 9, buf);  cell.a     = atof(buf);
            pdb_col(line, 15, 9, buf); cell.b     = atof(buf);
            pdb_col(line, 24, 9, buf); cell.c     = atof(buf);
            pdb_col(line, 33, 7, buf); cell.alpha = atof(buf);
            pdb_col(line, 40, 7, buf); cell.beta  = atof(buf);
            pdb_col(line, 47, 7, buf); cell.gamma = atof(buf);
            got_cryst1 = true;
            continue;
        }
        if (strncmp(line, "ATOM  ", 6) != 0 && strncmp(line, "HETATM", 6) != 0)
            continue;

        pdb_col(line, 30, 8, buf); float x = (float)atof(buf);
        pdb_col(line, 38, 8, buf); float y = (float)atof(buf);
        pdb_col(line, 46, 8, buf); float z = (float)atof(buf);
        pdb_col(line, 60, 6, buf); float B = (float)atof(buf);

        // Element: cols 77-78 (0-indexed 76-77); fall back to atom name cols 12-13
        pdb_col(line, 76, 2, buf);
        int el;
        if (buf[0] != ' ' || buf[1] != ' ') {
            el = elem_idx(buf);
        } else {
            pdb_col(line, 12, 2, buf);
            // atom name: first char may be space for 1-char elements
            el = elem_idx(buf);
        }

        xs.push_back(x); ys.push_back(y); zs.push_back(z);
        Bs.push_back(B); els.push_back(el);
    }
    fclose(f);
    if (!got_cryst1) { fprintf(stderr, "ERROR: no CRYST1 in %s\n", path); return false; }
    return true;
}

// ---------------------------------------------------------------------------
// Minimal MTZ binary writer (replaces gemmi::Mtz)
// Writes a minimal but CCP4-compatible MTZ file.
// ---------------------------------------------------------------------------

// Write an 80-character header record (space-padded).
static void mtz_rec(FILE* f, const char* fmt, ...) {
    char buf[81] = {};
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    // pad to 80 with spaces
    int n = (int)strlen(buf);
    memset(buf + n, ' ', 80 - n);
    fwrite(buf, 1, 80, f);
}

static void write_mtz(const char* path,
                       const UnitCell& cell,
                       const gemmi::SpaceGroup* sg,
                       int ncol,
                       const char* const col_names[],   // ncol names
                       const char  col_types[],          // ncol type chars
                       const char* const col_datasets[], // ncol dataset names
                       int nrefl,
                       const float* data)               // nrefl * ncol floats
{
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ERROR: cannot write %s\n", path); return; }

    // Machine stamp: little-endian IEEE (same as CCP4 on x86)
    static const unsigned char stamp[4] = { 0x44, 0x41, 0x00, 0x00 };
    // Header location (1-indexed word count from start of file)
    int header_loc = 4 + nrefl * ncol;  // 3 pre-data words + data words

    // Write pre-data: MTZ magic, header location, machine stamp
    fwrite("MTZ ", 1, 4, f);
    fwrite(&header_loc, 4, 1, f);
    fwrite(stamp, 4, 1, f);

    // Write data
    fwrite(data, sizeof(float), (size_t)nrefl * ncol, f);

    // ------- Text header (80-char records) -------

    mtz_rec(f, "VERS MTZ:V1.1");
    mtz_rec(f, "TITLE sfcalc_gpu_collapse output");
    mtz_rec(f, "NCOL %8d %12d %8d", ncol, nrefl, 0);
    mtz_rec(f, "CELL  %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f",
            cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma);
    mtz_rec(f, "SORT    0    0    0    0    0");

    // SYMINF: nsym nsymp lattice spgno 'spgname'
    gemmi::GroupOps ops = sg->operations();
    int nsym  = (int)ops.order();
    int nsymp = (int)ops.sym_ops.size();
    char lattice = sg->hm[0];
    mtz_rec(f, "SYMINF %4d %4d %c %4d '%s'",
            nsym, nsymp, lattice, sg->number, sg->xhm());

    // SYMM records: all operators (sym_ops × cen_ops)
    for (const gemmi::Op& s : ops.sym_ops)
        for (const gemmi::Op::Tran& c : ops.cen_ops) {
            gemmi::Op combined = s;
            for (int i = 0; i < 3; i++)
                combined.tran[i] = ((s.tran[i] + c[i]) % gemmi::Op::DEN
                                    + gemmi::Op::DEN) % gemmi::Op::DEN;
            mtz_rec(f, "SYMM %s", combined.triplet().c_str());
        }

    // Resolution limits
    UnitCell rc = cell.reciprocal();
    double ca = cos(rc.alpha*M_PI/180), cb = cos(rc.beta*M_PI/180), cg = cos(rc.gamma*M_PI/180);
    double max_invd2 = 0.0;
    for (int i = 0; i < nrefl; i++) {
        double H = data[i*ncol+0], K = data[i*ncol+1], L = data[i*ncol+2];
        double invd2 = H*H*rc.a*rc.a + K*K*rc.b*rc.b + L*L*rc.c*rc.c
                     + 2.0*(H*K*rc.a*rc.b*cg + H*L*rc.a*rc.c*cb + K*L*rc.b*rc.c*ca);
        if (invd2 > max_invd2) max_invd2 = invd2;
    }
    mtz_rec(f, "RESO  %12.6f %12.6f", 0.0, max_invd2);
    mtz_rec(f, "VALM NaN");

    // Column min/max
    std::vector<float> cmin(ncol,  1e30f);
    std::vector<float> cmax(ncol, -1e30f);
    for (int i = 0; i < nrefl; i++)
        for (int c = 0; c < ncol; c++) {
            float v = data[i*ncol+c];
            if (v < cmin[c]) cmin[c] = v;
            if (v > cmax[c]) cmax[c] = v;
        }

    // Two datasets: HKL_base (id 1) and SFCALC_GPU (id 2)
    for (int c = 0; c < ncol; c++)
        mtz_rec(f, "COL %-30s %c %12.4f %12.4f %s",
                col_names[c], col_types[c], cmin[c], cmax[c], col_datasets[c]);

    mtz_rec(f, "DCELL  1 %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f",
            cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma);
    mtz_rec(f, "DWAVEL  1 %10.4f", 0.0);
    mtz_rec(f, "DCELL  2 %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f",
            cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma);
    mtz_rec(f, "DWAVEL  2 %10.4f", 1.0);
    mtz_rec(f, "PROJECT  1 DEFAULT");
    mtz_rec(f, "CRYSTAL  1 DEFAULT");
    mtz_rec(f, "DATASET  1 HKL_base");
    mtz_rec(f, "PROJECT  2 DEFAULT");
    mtz_rec(f, "CRYSTAL  2 DEFAULT");
    mtz_rec(f, "DATASET  2 SFCALC_GPU");
    mtz_rec(f, "BATCH");
    mtz_rec(f, "END");

    fclose(f);
    fprintf(stderr, "  Written: %s  (%d reflections)\n", path, nrefl);
}

// Convenience wrappers matching the old API
static void write_mtz_phased(const std::vector<int>& h, const std::vector<int>& k,
                               const std::vector<int>& l,
                               const std::vector<float>& amp, const std::vector<float>& phi,
                               const UnitCell& cell, const gemmi::SpaceGroup* sg,
                               const std::string& path) {
    int n = (int)h.size();
    std::vector<float> data(5*n);
    for (int i = 0; i < n; i++) {
        data[5*i+0] = (float)h[i];
        data[5*i+1] = (float)k[i];
        data[5*i+2] = (float)l[i];
        data[5*i+3] = amp[i];
        data[5*i+4] = phi[i];
    }
    static const char* names[] = {"H","K","L","FC","PHIC"};
    static const char  types[] = {'H','H','H','F','P'};
    static const char* dsets[] = {"HKL_base","HKL_base","HKL_base","SFCALC_GPU","SFCALC_GPU"};
    write_mtz(path.c_str(), cell, sg, 5, names, types, dsets, n, data.data());
}

static void write_mtz_intensity(const std::vector<int>& h, const std::vector<int>& k,
                                 const std::vector<int>& l,
                                 const std::vector<float>& intensity,
                                 const UnitCell& cell,
                                 const std::string& path) {
    const gemmi::SpaceGroup* sg_p1 = gemmi::find_spacegroup_by_name("P 1");
    int n = (int)h.size();
    std::vector<float> data(4*n);
    for (int i = 0; i < n; i++) {
        data[4*i+0] = (float)h[i];
        data[4*i+1] = (float)k[i];
        data[4*i+2] = (float)l[i];
        data[4*i+3] = intensity[i];
    }
    static const char* names[] = {"H","K","L","I"};
    static const char  types[] = {'H','H','H','J'};
    static const char* dsets[] = {"HKL_base","HKL_base","HKL_base","SFCALC_GPU"};
    write_mtz(path.c_str(), cell, sg_p1, 4, names, types, dsets, n, data.data());
}

// ---------------------------------------------------------------------------
// Element index (must match order in sfcalc_gpu.cu)
// ---------------------------------------------------------------------------
static int elem_idx_from_name(const std::string& name) {
    if (name == "C") return 0;
    if (name == "H") return 1;
    if (name == "N") return 2;
    if (name == "O") return 3;
    if (name == "P") return 4;
    if (name == "S") return 5;
    return 0;  // default C
}

// ---------------------------------------------------------------------------
// Grid helpers
// ---------------------------------------------------------------------------
static const double LEVEL_FACTOR = 1.41421356237; // sqrt(2)

static int good_fft_size(int n) {
    int best = n * 10;
    for (int i2 = 1; i2 <= n*2; i2 *= 2)
        for (int i3 = i2; i3 <= n*2; i3 *= 3)
            for (int i5 = i3; i5 <= n*2; i5 *= 5)
                if (i5 >= n) best = std::min(best, i5);
    return best;
}

struct Level { double d; int nx, ny, nz; };

static std::vector<Level> compute_levels(double dmin, double rate,
                                          double ax, double ay, double az,
                                          int min_pts = 4) {
    std::vector<Level> levels;
    double d = dmin;
    while (true) {
        double s  = d / (2.0 * rate);
        int nx = good_fft_size(std::max(min_pts, (int)ceil(ax / s)));
        int ny = good_fft_size(std::max(min_pts, (int)ceil(ay / s)));
        int nz = good_fft_size(std::max(min_pts, (int)ceil(az / s)));
        levels.push_back({d, nx, ny, nz});
        if (std::min({nx, ny, nz}) <= min_pts) break;
        d *= LEVEL_FACTOR;
    }
    return levels;
}

static double noise_wpx(double noise_frac, double nb2d = 4.0, double nb3d = 6.0) {
    double ln2      = log(2.0);
    double noise_2d = noise_frac * (nb2d / nb3d);
    return (4.0/9.0) * sqrt(-log(log(noise_2d + 1.0)) / ln2);
}

static void assign_levels(const std::vector<float>& B, double pixel_fine,
                           double noise_frac, int n_levels,
                           std::vector<int>& out_lev) {
    double ln2   = log(2.0);
    double log_f = log(LEVEL_FACTOR);
    double w_px  = noise_wpx(noise_frac);
    out_lev.resize(B.size());
    for (size_t i = 0; i < B.size(); i++) {
        double fwhm      = sqrt(ln2 * B[i]) / (2.0 * M_PI);
        double pixel_req = fwhm / w_px;
        double ratio     = pixel_req / pixel_fine;
        int lev = 0;
        if (ratio >= 1.0)
            lev = (int)floor(log(ratio) / log_f);
        lev = std::max(0, std::min(lev, n_levels - 1));
        out_lev[i] = lev;
    }
}

static double b_threshold_for_level(int L, double pixel_fine, double noise_frac) {
    if (L == 0) return 0.0;
    double ln2    = log(2.0);
    double w_px   = noise_wpx(noise_frac);
    double pix_L  = pixel_fine * pow(LEVEL_FACTOR, L);
    double fwhm_min = pix_L * w_px;
    return fwhm_min * fwhm_min * (4.0 * M_PI * M_PI) / ln2;
}

// ---------------------------------------------------------------------------
// Accumulate coarse FFT output into fine-grid arrays (frequency-domain insert)
// ---------------------------------------------------------------------------
static void add_to_fine(float* acc_re, float* acc_im,
                         int nx, int ny, int nz,
                         const float* c_re, const float* c_im,
                         int nx_c, int ny_c, int nz_c) {
    int nx_c2 = nx_c/2 + 1;
    int nx2   = nx/2 + 1;
    int Ln = nz_c/2 + 1, Lh = nz_c - Ln;
    int Kn = ny_c/2 + 1, Kh = ny_c - Kn;

    for (int iz = 0; iz < Ln; iz++) {
        int fz = iz;
        for (int iy = 0; iy < Kn; iy++) {
            size_t fi = ((size_t)fz*ny + iy) * nx2;
            size_t ci = ((size_t)iz*ny_c + iy) * nx_c2;
            for (int ix = 0; ix < nx_c2; ix++) {
                acc_re[fi+ix] += c_re[ci+ix];
                acc_im[fi+ix] += c_im[ci+ix];
            }
        }
        if (Kh) {
            for (int iy = 0; iy < Kh; iy++) {
                size_t fi = ((size_t)fz*ny + (ny-Kh+iy)) * nx2;
                size_t ci = ((size_t)iz*ny_c + (Kn+iy)) * nx_c2;
                for (int ix = 0; ix < nx_c2; ix++) {
                    acc_re[fi+ix] += c_re[ci+ix];
                    acc_im[fi+ix] += c_im[ci+ix];
                }
            }
        }
    }
    if (Lh) {
        for (int iz = 0; iz < Lh; iz++) {
            int fz = nz - Lh + iz;
            for (int iy = 0; iy < Kn; iy++) {
                size_t fi = ((size_t)fz*ny + iy) * nx2;
                size_t ci = ((size_t)(Ln+iz)*ny_c + iy) * nx_c2;
                for (int ix = 0; ix < nx_c2; ix++) {
                    acc_re[fi+ix] += c_re[ci+ix];
                    acc_im[fi+ix] += c_im[ci+ix];
                }
            }
            if (Kh) {
                for (int iy = 0; iy < Kh; iy++) {
                    size_t fi = ((size_t)fz*ny + (ny-Kh+iy)) * nx2;
                    size_t ci = ((size_t)(Ln+iz)*ny_c + (Kn+iy)) * nx_c2;
                    for (int ix = 0; ix < nx_c2; ix++) {
                        acc_re[fi+ix] += c_re[ci+ix];
                        acc_im[fi+ix] += c_im[ci+ix];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Blur correction: multiply F(h,k,l) by exp(+b_add * stol^2)
// ---------------------------------------------------------------------------
static void apply_blur(float* acc_re, float* acc_im,
                        int nx, int ny, int nz,
                        float b_add,
                        float gs11, float gs22, float gs33,
                        float gs12, float gs13, float gs23) {
    int nx2 = nx/2 + 1;
    for (int iz = 0; iz < nz; iz++) {
        float L = (iz <= nz/2) ? (float)iz : (float)(iz - nz);
        for (int iy = 0; iy < ny; iy++) {
            float K = (iy <= ny/2) ? (float)iy : (float)(iy - ny);
            size_t row = ((size_t)iz * ny + iy) * nx2;
            for (int ix = 0; ix < nx2; ix++) {
                float H = (float)ix;
                float stol2 = 0.25f * (H*H*gs11 + K*K*gs22 + L*L*gs33
                              + 2.f*(H*K*gs12 + H*L*gs13 + K*L*gs23));
                float corr = expf(b_add * stol2);
                acc_re[row+ix] *= corr;
                acc_im[row+ix] *= corr;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Build primitive-cell ASU reflection list
// ---------------------------------------------------------------------------
struct HKL { int h, k, l; };

static std::vector<HKL> build_prim_asu(const gemmi::SpaceGroup* sg,
                                        const UnitCell& cell,
                                        double dmin) {
    UnitCell rc = cell.reciprocal();
    int H_max = (int)ceil(1.0 / (dmin * rc.a));
    int K_max = (int)ceil(1.0 / (dmin * rc.b));
    int L_max = (int)ceil(1.0 / (dmin * rc.c));

    double rca = cos(rc.alpha * M_PI / 180.0);
    double rcb = cos(rc.beta  * M_PI / 180.0);
    double rcg = cos(rc.gamma * M_PI / 180.0);
    double gs11 = rc.a*rc.a, gs22 = rc.b*rc.b, gs33 = rc.c*rc.c;
    double gs12 = rc.a*rc.b*rcg, gs13 = rc.a*rc.c*rcb, gs23 = rc.b*rc.c*rca;
    double inv_dmin2 = 1.0 / (dmin * dmin);

    gemmi::ReciprocalAsu asu(sg);
    gemmi::GroupOps ops = sg->operations();

    std::vector<HKL> result;
    for (int H = -H_max; H <= H_max; H++) {
        for (int K = -K_max; K <= K_max; K++) {
            for (int L = -L_max; L <= L_max; L++) {
                if (!asu.is_in({H, K, L})) continue;
                double inv_d2 = H*H*gs11 + K*K*gs22 + L*L*gs33
                              + 2.0*(H*K*gs12 + H*L*gs13 + K*L*gs23);
                if (inv_d2 <= 0 || inv_d2 > inv_dmin2) continue;
                if (ops.is_systematically_absent({H, K, L})) continue;
                result.push_back({H, K, L});
            }
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Collapse supercell to primitive-cell ASU
// ---------------------------------------------------------------------------
static void collapse_to_prim_asu(const float* acc_re, const float* acc_im,
                                   int nx, int ny, int nz,
                                   int na, int nb, int nc,
                                   const gemmi::SpaceGroup* sg,
                                   const std::vector<HKL>& asu,
                                   std::vector<float>& F_re,
                                   std::vector<float>& F_im) {
    int nx2  = nx/2 + 1;
    int n    = (int)asu.size();
    F_re.assign(n, 0.f);
    F_im.assign(n, 0.f);

    gemmi::GroupOps ops = sg->operations();
    int den = gemmi::Op::DEN;

    for (const gemmi::Op& op : ops) {
        auto& rot  = op.rot;
        auto& tran = op.tran;

        for (int i = 0; i < n; i++) {
            int H = asu[i].h, K = asu[i].k, L = asu[i].l;

            // R^T * (H,K,L)
            int Hr = (rot[0][0]*H + rot[1][0]*K + rot[2][0]*L) / den;
            int Kr = (rot[0][1]*H + rot[1][1]*K + rot[2][1]*L) / den;
            int Lr = (rot[0][2]*H + rot[1][2]*K + rot[2][2]*L) / den;

            int SH = na * Hr, SK = nb * Kr, SL = nc * Lr;

            bool friedel = (SH < 0) || (SH == 0 && SK < 0)
                         || (SH == 0 && SK == 0 && SL < 0);
            if (friedel) { SH = -SH; SK = -SK; SL = -SL; }

            int ix = SH;
            int iy = (SK >= 0) ? SK : SK + ny;
            int iz = (SL >= 0) ? SL : SL + nz;

            if (ix >= nx2 || iy < 0 || iy >= ny || iz < 0 || iz >= nz) continue;

            size_t idx = (size_t)iz * ny * nx2 + iy * nx2 + ix;
            float re   =  acc_re[idx];
            float im   = friedel ? -acc_im[idx] : acc_im[idx];

            float phase = (float)(2.0 * M_PI *
                          (H * tran[0] + K * tran[1] + L * tran[2]) / (double)den);
            float cp = cosf(phase), sp = sinf(phase);

            F_re[i] += re * cp - im * sp;
            F_im[i] += re * sp + im * cp;
        }
    }
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------
struct Args {
    std::string pdb, outmtz, outI, outmap, lib, sg_name;
    double dmin, rate, bmax, noise;
    int na, nb, nc;
};

static Args parse_args(int argc, char** argv) {
    std::string exe(argv[0]);
    std::string dir = exe.substr(0, exe.rfind('/') + 1);

    Args a;
    a.outmtz   = "collapsed.mtz";
    a.outI     = "supercell_I.mtz";
    a.outmap   = "";
    a.lib      = dir + "sfcalc_gpu.so";
    a.sg_name  = "P 1";
    a.dmin     = 1.5;
    a.rate     = 2.5;
    a.bmax     = 0.0;
    a.noise    = 0.01;
    a.na = a.nb = a.nc = 1;

    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h" || arg == "-help") {
            printf(
"Usage: sfcalc_gpu_collapse <input.pdb> [key=value ...]\n"
"\n"
"Compute crystallographic structure factors from a P1 supercell PDB using GPU\n"
"FFT, then collapse to the primitive-cell ASU using space-group symmetry.\n"
"\n"
"Required:\n"
"  <input.pdb>          P1 supercell PDB with CRYST1 giving supercell dimensions\n"
"\n"
"Options (key=value, case-insensitive):\n"
"  sg=<name>            Space group of the PRIMITIVE cell  (default: P 1)\n"
"                         e.g. sg=\"P 21 21 21\"  sg=\"I 2 3\"  sg=19\n"
"  super_mult=na,nb,nc  Supercell multipliers along a, b, c  (default: 1,1,1)\n"
"                         Primitive cell = supercell / (na, nb, nc)\n"
"  dmin=<A>             Resolution limit in Angstroms  (default: 1.5)\n"
"  rate=<r>             Shannon oversampling rate for FFT grid  (default: 2.5)\n"
"  bmax=<B>             Skip atoms with B-factor > bmax; 0 = keep all  (default: 0)\n"
"  noise=<f>            Multi-grid B-factor coarsening tolerance  (default: 0.01)\n"
"                         Fraction of peak density; higher = coarser/faster\n"
"\n"
"Output files:\n"
"  outmtz=<file>        Primitive-cell ASU MTZ with FC and PHIC  (default: collapsed.mtz)\n"
"                         Use for computing |<F>|^2 by averaging complex F across frames\n"
"  outI=<file>          Supercell P1 intensity MTZ with I=|F|^2  (default: supercell_I.mtz)\n"
"                         Use for computing <|F|^2> by averaging I across frames\n"
"  outmap=<file>        Electron density map in CCP4 format (optional, default: none)\n"
"\n"
"Advanced:\n"
"  lib=<path>           Path to sfcalc_gpu.so  (default: directory of this executable)\n"
"\n"
"Diffuse scatter intensity: I_diffuse(H) = <|F(H)|^2> - |<F(H)>|^2\n"
"  Average outI across MD frames to get <|F|^2>.\n"
"  Average complex F from outmtz across frames, then square, to get |<F>|^2.\n"
"\n"
"Example (orthorhombic 2x2x2 supercell):\n"
"  sfcalc_gpu_collapse frame_001.pdb sg=\"P 21 21 21\" super_mult=2,2,2 dmin=2.0\n"
"      outmtz=frame_001_collapsed.mtz outI=frame_001_I.mtz\n"
);
            exit(0);
        }
        auto eq = arg.find('=');
        if (eq == std::string::npos) {
            if (arg.size() > 4 &&
                (arg.substr(arg.size()-4) == ".pdb" ||
                 arg.substr(arg.size()-4) == ".cif"))
                a.pdb = arg;
            continue;
        }
        std::string key = arg.substr(0, eq);
        std::string val = arg.substr(eq+1);
        for (auto& c : key) c = tolower(c);

        if (key == "dmin")            a.dmin   = atof(val.c_str());
        else if (key == "rate")       a.rate   = atof(val.c_str());
        else if (key == "outmtz" || key == "mtz")  a.outmtz = val;
        else if (key == "outi" || key == "outsq" || key == "outisq") a.outI = val;
        else if (key == "outmap" || key == "map")  a.outmap = val;
        else if (key == "bmax")       a.bmax   = atof(val.c_str());
        else if (key == "noise")      a.noise  = atof(val.c_str());
        else if (key == "lib")        a.lib    = val;
        else if (key == "sg" || key == "spacegroup" || key == "space_group")
            a.sg_name = val;
        else if (key == "super_mult" || key == "mult" || key == "multipliers") {
            std::string v = val;
            for (auto& c : v) if (c == 'x') c = ',';
            if (sscanf(v.c_str(), "%d,%d,%d", &a.na, &a.nb, &a.nc) != 3) {
                fprintf(stderr, "ERROR: super_mult must be na,nb,nc\n");
                exit(1);
            }
        }
    }
    return a;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    if (args.pdb.empty()) {
        fprintf(stderr, "Usage: sfcalc_gpu_collapse <input.pdb> [key=val ...]\n");
        fprintf(stderr, "Run with --help for full option list.\n");
        return 1;
    }

    double dmin  = args.dmin;
    double rate  = args.rate;
    double bmax  = args.bmax;
    double noise = args.noise;
    int na = args.na, nb = args.nb, nc = args.nc;

    const gemmi::SpaceGroup* sg = gemmi::find_spacegroup_by_name(args.sg_name);
    if (!sg) { fprintf(stderr, "ERROR: unknown space group '%s'\n", args.sg_name.c_str()); return 1; }

    // -----------------------------------------------------------------------
    // 1. Load PDB
    // -----------------------------------------------------------------------
    fprintf(stderr, "Reading %s ...\n", args.pdb.c_str());
    UnitCell cell{};
    std::vector<float> xs, ys, zs, Bs;
    std::vector<int>   els;
    if (!read_pdb(args.pdb.c_str(), cell, xs, ys, zs, Bs, els)) return 1;

    double ax = cell.a, ay = cell.b, az = cell.c;
    fprintf(stderr, "  Supercell: %.3f x %.3f x %.3f  angles: %.2f %.2f %.2f\n",
            ax, ay, az, cell.alpha, cell.beta, cell.gamma);

    UnitCell prim_cell{ax/na, ay/nb, az/nc, cell.alpha, cell.beta, cell.gamma};
    fprintf(stderr, "  super_mult: %d x %d x %d\n", na, nb, nc);
    fprintf(stderr, "  Primitive cell: %.3f x %.3f x %.3f\n",
            prim_cell.a, prim_cell.b, prim_cell.c);
    fprintf(stderr, "  Space group: %s\n", sg->xhm());

    int natoms = (int)xs.size();
    float B_min = *std::min_element(Bs.begin(), Bs.end());
    float B_max_val = *std::max_element(Bs.begin(), Bs.end());
    fprintf(stderr, "  Atoms: %d  B: %.2f .. %.2f\n", natoms, B_min, B_max_val);

    // bmax filter
    if (bmax > 0) {
        std::vector<float> xs2, ys2, zs2, Bs2;
        std::vector<int>   els2;
        int nskip = 0;
        for (int i = 0; i < natoms; i++) {
            if (Bs[i] > (float)bmax) { nskip++; continue; }
            xs2.push_back(xs[i]); ys2.push_back(ys[i]); zs2.push_back(zs[i]);
            Bs2.push_back(Bs[i]); els2.push_back(els[i]);
        }
        if (nskip) fprintf(stderr, "  Skipping %d atoms with B > %.1f\n", nskip, bmax);
        xs = xs2; ys = ys2; zs = zs2; Bs = Bs2; els = els2;
        natoms = (int)xs.size();
    }

    // Auto-blur
    float b_add      = (float)((dmin * rate) * (dmin * rate) / (M_PI * M_PI));
    float sigma_min  = sqrtf(b_add / (4.f * (float)(M_PI*M_PI)));
    float pixel_fine_v = (float)(dmin / (2.0 * rate));
    fprintf(stderr, "  Auto-blur: b_add = %.4f A^2  (sigma_min = %.3f A, pixel = %.3f A)\n",
            b_add, sigma_min, pixel_fine_v);
    for (int i = 0; i < natoms; i++) Bs[i] += b_add;

    // -----------------------------------------------------------------------
    // 2. Multi-grid levels
    // -----------------------------------------------------------------------
    auto levels    = compute_levels(dmin, rate, ax, ay, az);
    int n_levels   = (int)levels.size();
    double pix_fine = dmin / (2.0 * rate);
    std::vector<int> atom_lev;
    assign_levels(Bs, pix_fine, noise, n_levels, atom_lev);

    int nx = levels[0].nx, ny = levels[0].ny, nz = levels[0].nz;
    int nx2 = nx/2 + 1;
    double V_cell = cell.volume();
    double w_px   = noise_wpx(noise);

    int last_occ = 0;
    for (int L = 0; L < n_levels; L++)
        for (int i = 0; i < natoms; i++)
            if (atom_lev[i] == L) { last_occ = L; break; }

    fprintf(stderr, "  Multi-grid levels (noise<=%.1f%%, w_px=%.3f, step=sqrt(2)):\n",
            noise*100, w_px);
    for (int L = 0; L <= last_occ; L++) {
        int n_L = 0;
        for (int i = 0; i < natoms; i++) if (atom_lev[i] == L) n_L++;
        double B_lo = b_threshold_for_level(L,   pix_fine, noise);
        double B_hi = (L < n_levels-1) ? b_threshold_for_level(L+1, pix_fine, noise)
                                       : 1e30;
        double vfrac = (double)(levels[L].nx * levels[L].ny * levels[L].nz)
                     / (double)(nx * ny * nz) * 100.0;
        fprintf(stderr, "    L%d: pixel=%.3fA  grid %dx%dx%d (%.1f%%)  "
                "B=[%.1f,%.1f)  %d atoms\n",
                L, levels[L].d/rate/2,
                levels[L].nx, levels[L].ny, levels[L].nz,
                vfrac, B_lo, B_hi, n_L);
    }

    // -----------------------------------------------------------------------
    // 3. Load GPU library
    // -----------------------------------------------------------------------
    void* gpu_lib = dlopen(args.lib.c_str(), RTLD_NOW);
    if (!gpu_lib) {
        fprintf(stderr, "ERROR: cannot load %s: %s\n", args.lib.c_str(), dlerror());
        return 1;
    }
    SpreadAndFft_fn spread_and_fft =
        (SpreadAndFft_fn)dlsym(gpu_lib, "spread_and_fft");
    if (!spread_and_fft) {
        fprintf(stderr, "ERROR: spread_and_fft not found in %s\n", args.lib.c_str());
        return 1;
    }

    // -----------------------------------------------------------------------
    // 4. Spread + FFT each level → accumulate into fine-grid float32 arrays
    // -----------------------------------------------------------------------
    size_t acc_size = (size_t)nz * ny * nx2;
    std::vector<float> acc_re(acc_size, 0.f);
    std::vector<float> acc_im(acc_size, 0.f);

    for (int L = 0; L <= last_occ; L++) {
        std::vector<float> lx, ly, lz, lB;
        std::vector<int>   lel;
        for (int i = 0; i < natoms; i++) {
            if (atom_lev[i] != L) continue;
            lx.push_back(xs[i]); ly.push_back(ys[i]); lz.push_back(zs[i]);
            lB.push_back(Bs[i]); lel.push_back(els[i]);
        }
        int n_L = (int)lx.size();
        if (n_L == 0) continue;

        int NX = levels[L].nx, NY = levels[L].ny, NZ = levels[L].nz;
        int NX2 = NX/2 + 1;
        size_t fft_n = (size_t)NX2 * NY * NZ;

        std::vector<float> Fr(fft_n), Fi(fft_n);
        float* map_ptr = nullptr;
        // (map output not yet implemented — requires CCP4 map writer)

        int nkept = spread_and_fft(
            n_L,
            lx.data(), ly.data(), lz.data(), lB.data(), lel.data(),
            NX, NY, NZ,
            (float)ax, (float)ay, (float)az,
            (float)cell.alpha, (float)cell.beta, (float)cell.gamma,
            0.f,
            map_ptr, Fr.data(), Fi.data()
        );
        if (nkept < 0) {
            fprintf(stderr, "ERROR: spread_and_fft returned %d\n", nkept);
            return 1;
        }

        float norm_L = (float)(V_cell / ((double)NX * NY * NZ));
        for (size_t j = 0; j < fft_n; j++) {
            Fr[j] *=  norm_L;
            Fi[j] *= -norm_L;
        }

        if (L == 0) {
            for (size_t j = 0; j < acc_size; j++) {
                acc_re[j] += Fr[j];
                acc_im[j] += Fi[j];
            }
        } else {
            add_to_fine(acc_re.data(), acc_im.data(), nx, ny, nz,
                        Fr.data(), Fi.data(), NX, NY, NZ);
        }
    }

    // -----------------------------------------------------------------------
    // 5. Apply blur correction: F(H) *= exp(+b_add * stol^2)
    // -----------------------------------------------------------------------
    {
        UnitCell rc = cell.reciprocal();
        float cg = (float)cos(rc.gamma * M_PI / 180.0);
        float cb = (float)cos(rc.beta  * M_PI / 180.0);
        float ca = (float)cos(rc.alpha * M_PI / 180.0);
        float gs11 = (float)(rc.a*rc.a), gs22 = (float)(rc.b*rc.b), gs33 = (float)(rc.c*rc.c);
        float gs12 = (float)(rc.a*rc.b)*cg;
        float gs13 = (float)(rc.a*rc.c)*cb;
        float gs23 = (float)(rc.b*rc.c)*ca;
        apply_blur(acc_re.data(), acc_im.data(), nx, ny, nz,
                   b_add, gs11, gs22, gs33, gs12, gs13, gs23);
    }

    // -----------------------------------------------------------------------
    // 6. Extract supercell P1 ASU → write intensity MTZ
    // -----------------------------------------------------------------------
    fprintf(stderr, "Extracting supercell structure factors ...\n");

    UnitCell rc_sc = cell.reciprocal();
    double rca = cos(rc_sc.alpha * M_PI / 180.0);
    double rcb = cos(rc_sc.beta  * M_PI / 180.0);
    double rcg = cos(rc_sc.gamma * M_PI / 180.0);
    double m11 = rc_sc.a*rc_sc.a, m22 = rc_sc.b*rc_sc.b, m33 = rc_sc.c*rc_sc.c;
    double m12 = rc_sc.a*rc_sc.b*rcg, m13 = rc_sc.a*rc_sc.c*rcb, m23 = rc_sc.b*rc_sc.c*rca;
    double inv_dmin2 = 1.0 / (dmin * dmin);

    std::vector<int>   sc_h, sc_k, sc_l;
    std::vector<float> sc_I;

    for (int iz = 0; iz < nz; iz++) {
        int L = (iz <= nz/2) ? iz : iz - nz;
        for (int iy = 0; iy < ny; iy++) {
            int K = (iy <= ny/2) ? iy : iy - ny;
            for (int ix = 0; ix < nx2; ix++) {
                int H = ix;
                bool in_asu = (L > 0) || (L == 0 && K > 0) || (L == 0 && K == 0 && H > 0);
                if (!in_asu) continue;
                double inv_d2 = H*H*m11 + K*K*m22 + L*L*m33
                              + 2.0*(H*K*m12 + H*L*m13 + K*L*m23);
                if (inv_d2 <= 0 || inv_d2 > inv_dmin2) continue;

                size_t idx = (size_t)iz * ny * nx2 + iy * nx2 + ix;
                float re = acc_re[idx], im = acc_im[idx];
                sc_h.push_back(H); sc_k.push_back(K); sc_l.push_back(L);
                sc_I.push_back(re*re + im*im);
            }
        }
    }

    if (!args.outI.empty())
        write_mtz_intensity(sc_h, sc_k, sc_l, sc_I, cell, args.outI);

    // -----------------------------------------------------------------------
    // 7. Collapse to primitive-cell ASU → write phased MTZ
    // -----------------------------------------------------------------------
    bool simple_p1 = (na == 1 && nb == 1 && nc == 1 && std::string(sg->hm) == "P 1");

    if (simple_p1) {
        int n = (int)sc_h.size();
        std::vector<float> amp(n), phi(n);
        for (int i = 0; i < n; i++) {
            size_t idx = (size_t)
                ((sc_l[i] < 0 ? sc_l[i] + nz : sc_l[i])) * ny * nx2
                + ((sc_k[i] < 0 ? sc_k[i] + ny : sc_k[i])) * nx2
                + sc_h[i];
            float re = acc_re[idx], im = acc_im[idx];
            amp[i] = sqrtf(re*re + im*im);
            phi[i] = atan2f(im, re) * (float)(180.0 / M_PI);
        }
        write_mtz_phased(sc_h, sc_k, sc_l, amp, phi, cell, sg, args.outmtz);
    } else {
        fprintf(stderr, "Collapsing supercell to primitive-cell ASU (%s, %d operators) ...\n",
                sg->xhm(), (int)sg->operations().order());
        auto asu_refl = build_prim_asu(sg, prim_cell, dmin);
        fprintf(stderr, "  Primitive-cell ASU reflections: %d\n", (int)asu_refl.size());

        std::vector<float> F_re, F_im;
        collapse_to_prim_asu(acc_re.data(), acc_im.data(), nx, ny, nz,
                              na, nb, nc, sg, asu_refl, F_re, F_im);

        int n = (int)asu_refl.size();
        std::vector<float> amp(n), phi(n);
        std::vector<int> rh(n), rk(n), rl(n);
        for (int i = 0; i < n; i++) {
            rh[i] = asu_refl[i].h; rk[i] = asu_refl[i].k; rl[i] = asu_refl[i].l;
            amp[i] = sqrtf(F_re[i]*F_re[i] + F_im[i]*F_im[i]);
            phi[i] = atan2f(F_im[i], F_re[i]) * (float)(180.0 / M_PI);
        }
        write_mtz_phased(rh, rk, rl, amp, phi, prim_cell, sg, args.outmtz);
    }

    dlclose(gpu_lib);
    fprintf(stderr, "Done.\n");
    return 0;
}
