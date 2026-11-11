"""Soft-sphere kernels for the solvent mask's source: every atom a flat-top
sphere of half its Lennard-Jones diameter, fitted by the five Gaussians the density
placement already sums, so the pipeline and its gradients are unchanged.

WHY SPHERES. A geometric mask is the union of spheres of radius r_atom +
r_probe, then shrunk; what it answers is "how far from this atom can a water
sit". The atom-presence field (solvent_torch.mask_source_weights) got most of
the way there with one Gaussian per atom; a flat-top profile with a sharp
edge makes the summed field change fastest near the atoms' surfaces, and
lets the radius differ by atom TYPE, which is where the hydrogens the model
does not carry come in: a methyl carbon is a larger object than a carbonyl
carbon. NOTE what the mask is: a level set of the SUM of spheres, at the
calibrated cutoff, not their union. In the benchmark the cutoff sits near
one plateau (median 0.95 "sphere overlaps", 0.90-1.51), i.e. well inside a
lone sphere's surface (at 0.95 an isolated sphere reaches it only within
about d - 1.5 A, and above 1.0 not at all, so an isolated water need not
enter the mask). The radius is a shape and weighting parameter, not a fixed
distance to the boundary.

THE RADII. Each atom is a sphere of radius sigma_i / 2 (plus an optional
offset), sigma_i its OPLS united-atom Lennard-Jones diameter (Jorgensen and
Tirado-Rives, J. Am. Chem. Soc. 110, 1657, 1988), which carries its
hydrogens implicitly: what a deposited model without hydrogens needs, and
why a methyl carbon is a larger sphere than a carbonyl carbon. The water
contact distance, (sigma_i + sigma_water) / 2 with TIP3P's 3.15061 A, is
contact_radius(); it is the solvent-ACCESSIBLE surface, too far out for a
mask (sphere_radii). The values are SIGMA_UA below, checked against Tinker's
params/oplsua.prm on 2026-11-10 (a first version written from memory had
the alpha carbons at 3.85-3.905 instead of 3.80, aromatic carbons at 3.75
instead of 3.55, Arg CZ at 3.75 instead of 2.25 and water at 2.96 instead of
3.151; every number measured before that check used it).

    type   sigma (A)   atoms
    CHA    3.800       backbone CA (Pro and termini included)
    CH2A   3.800       Gly CA
    CH3B   3.910       Ala CB, Val/Leu methyls, Ile CG2, Thr CG2
    CH     3.850       Val/Ile/Thr CB, Leu CG
    CH2    3.905       other side-chain methylenes
    CH3    3.905       Ile CD1
    CH2X   3.800       Pro CD, Met CG
    CH3S   3.800       Met CE
    CAR    3.550       aromatic ring carbons of Phe/Tyr/Trp/His; OPLS-UA keeps
                       the aromatic H explicit (type 75), so a model without
                       hydrogens has its ring CH carbons a little small
    C      3.750       carbonyl and carboxylate C, Asn/Gln amide C
    CZR    2.250       Arg CZ
    N      3.250       every nitrogen
    O      2.960       carbonyl and carboxylate O
    OH     3.070       Ser/Thr/Tyr hydroxyl O
    S      3.550       Cys SG, Met SD (and MSE SE)
    OW     3.151       water O
    H      2.420       any explicit H (OPLS-UA's aromatic H; a deliberate
                       departure for polar H, which OPLS-UA gives sigma 0);
                       the sp3 carbons of a residue that carries hydrogens
                       then drop to the bare C (3.75), since their hydrogens
                       are atoms of their own. Decided per residue.
    Extrapolations beyond the source: MSE SE as S, the explicit-H rule
    above, and the element fallback's scale.

Anything else (ligands, nucleic acids, ions, unknown names) falls back to the
element: sigma_e = 2 r_vdw(e) * (3.75 / 3.40), i.e. gemmi's van der Waals
radii scaled so that a bare carbon lands on the OPLS carbon, which keeps one
scale across the table.

THE KERNEL is a Fermi profile 1 / (1 + exp((r - d) / w)) with edge width w
(default 0.5 A), fitted once per distinct radius by five centred Gaussians
with bounded amplitudes (|A| <= 20): for every radius the united-atom
table produces, plateau within 1.4% and the half-height crossing within
0.002 A; the large element-fallback radii fit worse (3.34 A: 3.9%, 0.012 A;
3.78 A: 7.6%, 0.024 A). Narrower
edges fit worse (w = 0.3: 20% plateau ripple). The fits for every radius the
default table and element fallback produce are precomputed in
SPHERE_FIT_TABLE, so a default run does no fitting at all; other radii are
fitted live (about 0.7 s each on a CPU) and cached per process.
"""
import functools

import numpy as np

SIGMA_WATER = 3.15061          # TIP3P oxygen, A
# OPLS united-atom Lennard-Jones sigma, A, by type, as Jorgensen and
# Tirado-Rives (J. Am. Chem. Soc. 110, 1657, 1988) give them for proteins and
# Tinker distributes them (params/oplsua.prm, atom types 1-80; the type
# numbers are quoted). Checked against that file on 2026-11-10.
SIGMA_UA = {
    "C":    3.750,   # 1 C peptide amide; 17 COO- carboxylate; Asn/Gln amide C
    "O":    2.960,   # 2 O peptide amide; 18 O- carboxylate
    "N":    3.250,   # 3, 12, 20, 40, 42, 46, 51, 54: every nitrogen
    "CHA":  3.800,   # 6, 14, 29, 30 CH (alpha), Pro and termini included
    "CH2A": 3.800,   # 5, 27, 28 CH2 (alpha) Gly
    "CH3B": 3.910,   # 7 CH3 (beta) Ala; Tinker's biotypes also give it to the
                     #   Val/Leu methyls, Ile CG2 and Thr CG2
    "CH":   3.850,   # 8 CH (beta) Val/Leu/Ile; 25 CHOH (beta) Thr
    "CH2":  3.905,   # 9 CH2 generic; 16, 19, 22, 31, 56, 57, 66
    "CH3":  3.905,   # 10 CH3 (delta) Ile
    "CAR":  3.550,   # 11, 26, 43-45, 48-50, 67-69, 73, 74 aromatic C and CH
    "CH2X": 3.800,   # 15 CH2 (delta) Pro; 34 CH2 (gamma) Met (37, cystine CB,
                     #   would be the same, but disulfides are not detected)
    "CH3S": 3.800,   # 36 CH3 (epsilon) Met
    "CZR":  2.250,   # 53 C (zeta) Arg
    "OH":   3.070,   # 23 OH Ser/Thr; 76 OH Tyr
    "S":    3.550,   # 32 SH Cys; 35 -S- Met; 38 -SS- cystine
    "OW":   3.151,   # 77 O water (TIP3P)
    "H":    2.420,   # 75 H(C) aromatic; used for every explicit H (see lj_sigma)
}
EDGE_WIDTH_DEFAULT = 0.5       # A

_AROMATIC_RES = {"PHE", "TYR", "TRP", "HIS", "HID", "HIE", "HIP"}
_STANDARD = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
             "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE",
             "HID", "HIE", "HIP"}
# side-chain carbons that are not plain CH2, by residue
_SIDE = {
    "ALA": {"CB": "CH3B"},
    "VAL": {"CB": "CH", "CG1": "CH3B", "CG2": "CH3B"},
    "LEU": {"CG": "CH", "CD1": "CH3B", "CD2": "CH3B"},
    "ILE": {"CB": "CH", "CG2": "CH3B", "CD1": "CH3", "CD": "CH3"},
    "THR": {"CB": "CH", "CG2": "CH3B"},
    "PRO": {"CD": "CH2X"},
    "MET": {"CG": "CH2X", "CE": "CH3S"},
    "MSE": {"CG": "CH2X", "CE": "CH3S"},
    "ASP": {"CG": "C"}, "GLU": {"CD": "C"}, "ASN": {"CG": "C"}, "GLN": {"CD": "C"},
    "ARG": {"CZ": "CZR"},
}
_HYDROXYL = {("SER", "OG"), ("THR", "OG1"), ("TYR", "OH")}


def bare_element(element):
    """The element symbol without a charge: cctbx scattering types carry it
    ('O1-', 'Zn2+'), and an unstripped 'O1-' would miss its type and fall
    back to gemmi's unknown element X."""
    import re

    m = re.match(r"\s*([A-Za-z]+)", element)
    return m.group(1) if m else element.strip()


def is_hydrogen(element):
    """True for hydrogen or deuterium, charged or not, in any case."""
    return bare_element(element).upper() in ("H", "D")


def united_atom_type(resname, name, element):
    """OPLS united-atom type (a SIGMA_UA key) for a protein atom or water,
    or None for anything the table does not cover (the caller falls back to
    the element). Cystine CB (3.80) is not told from cysteine CB (3.905):
    that needs the disulfide, which the labels do not carry."""
    res, nm, el = resname.strip().upper(), name.strip().upper(), bare_element(element).upper()
    if is_hydrogen(el):
        return "H"
    if res in ("HOH", "WAT", "DOD") and el == "O":
        return "OW"
    if res not in _STANDARD:
        return None
    if el == "N":
        return "N"
    if el in ("S", "SE"):
        return "S"
    if el == "O":
        return "OH" if (res, nm) in _HYDROXYL else "O"
    if el != "C":
        return None
    if nm == "C":
        return "C"
    if nm == "CA":
        return "CH2A" if res == "GLY" else "CHA"
    if nm in _SIDE.get(res, {}):
        return _SIDE[res][nm]
    if res in _AROMATIC_RES and nm not in ("CB",):
        return "CAR"
    return "CH2"


def lj_sigma(resname, name, element, explicit_hydrogens=False, vdw_radius=None,
             by_element=False):
    """The atom's own Lennard-Jones sigma, A (united-atom type, or the
    element fallback). With explicit hydrogens the heavy atoms are bare
    (C, N, O/OH, S) and the hydrogens get their own; without, the
    united-atom types carry the hydrogens. by_element skips the typing and
    uses the element fallback for every atom, N, S and H included."""
    t = None if by_element else united_atom_type(resname, name, element)
    if t is not None and explicit_hydrogens and t in ("CHA", "CH2A", "CH3B", "CH", "CH2",
                                                      "CH3", "CH2X", "CH3S"):
        t = "C"                    # the hydrogens are atoms of their own now
    if t is not None:
        sigma = SIGMA_UA[t]
    else:
        if vdw_radius is None:
            import gemmi
            vdw_radius = float(gemmi.Element(bare_element(element).capitalize()).vdw_r)
        sigma = 2.0 * vdw_radius * (3.75 / 3.40)
    return sigma


def contact_radius(resname, name, element, explicit_hydrogens=False, vdw_radius=None):
    """Water contact distance for one atom, A: (sigma_atom + sigma_water)/2,
    where a water oxygen's Lennard-Jones energy with the atom crosses zero.
    This is the radius of the SOLVENT-ACCESSIBLE surface (probe centres). The
    mask boundary a geometric mask draws is nearer the atoms: dilated by the
    probe and then shrunk back, i.e. about the atom's own radius sigma/2
    plus a little. sphere_radii() is what the mask uses."""
    return 0.5 * (lj_sigma(resname, name, element, explicit_hydrogens, vdw_radius) + SIGMA_WATER)


def sphere_radii(labels, elements, offset=0.0, explicit_hydrogens=None, by_element=False):
    """The mask's sphere radii: sigma/2 + offset per atom (offset in A, the
    knob that plays the geometric mask's probe-minus-shrink), from cctbx
    scatterer labels or, with by_element, the element fallback for every
    atom. A sum of spheres is NOT their union: inside a protein the spheres
    overlap many deep, so the field peaks far above one plateau and a cutoff
    calibrated to the solvent fraction only lands near a sphere's edge if
    the radii are about right. At the full contact radius (sigma/2 + 1.6 A)
    the union fills 7FPV's cell and the calibrated cutoff sat at nine
    plateaus, with no sphere surface anywhere near it; at sigma/2 it sits
    near one plateau. The offset is swept, not derived.

    explicit_hydrogens None (default) decides PER RESIDUE: a residue whose
    atoms include a hydrogen has its sp3 carbons bare, every other residue
    keeps the united-atom types, so hydrogens on a ligand or a water do not
    shrink the protein. True or False forces it for every atom."""
    parsed = []
    for lab in labels:
        # cctbx labels are 'pdb=" CA AMET A   1 "': atom name in columns 1-4
        # of the quoted part, altloc 5, residue name 6-8, the rest chain,
        # number and insertion code
        q = lab.split('"')[1] if '"' in lab else lab
        parsed.append((q[0:4] if len(q) >= 4 else "", q[5:8] if len(q) >= 8 else "", q[5:]))
    if explicit_hydrogens is None:
        with_h = {key for (_, _, key), el in zip(parsed, elements)
                  if is_hydrogen(el)}
        per_atom = [key in with_h for (_, _, key) in parsed]
    else:
        per_atom = [bool(explicit_hydrogens)] * len(parsed)
    out = np.empty(len(labels))
    for i, ((name, resname, _), el) in enumerate(zip(parsed, elements)):
        out[i] = 0.5 * lj_sigma(resname, name, el, per_atom[i],
                                by_element=by_element) + offset
    if (out <= 0.0).any():
        raise ValueError("offset %g leaves a sphere radius <= 0 (smallest %.3f A)"
                         % (offset, out.min()))
    return out


def fermi(r, radius, width):
    return 1.0 / (1.0 + np.exp((np.asarray(r) - radius) / width))


# Precomputed by the fit below (8 starts) for every radius the default table
# and element fallback produce at offset 0 and edge 0.5 A, keyed by the
# radius rounded to 0.001 A. Anything else (an offset, another edge, an
# element not listed) is fitted live and cached. Regenerate with
# tools/fit_sphere_table.py if the table, the edge or the fit changes.
SPHERE_FIT_TABLE = {
    1.125: ((-0.123690, 0.180846, 20.000000, -19.966887, 0.800576), (1.347682, 0.256293, 5.000000, 4.993344, 0.564052)),
    1.169: ((0.187870, -0.170048, -19.963654, 20.000000, 0.844807), (0.253127, 1.284803, 4.994694, 5.000000, 0.553331)),
    1.210: ((-0.218471, 0.195228, 13.860597, -13.821689, 0.890369), (1.222301, 0.250410, 5.000000, 4.994061, 0.544433)),
    1.246: ((0.259240, 0.051028, 0.000682, -0.292150, 0.891415), (0.270518, 5.000000, 0.020000, 1.205359, 0.559318)),
    1.313: ((0.492853, 0.044689, 0.032382, -3.458502, 3.809181), (0.338483, 5.000000, 0.162565, 0.821684, 0.753349)),
    1.324: ((0.478845, 0.044200, 0.022662, -5.203025, 5.579573), (0.326680, 5.000000, 0.148127, 0.797393, 0.750030)),
    1.346: ((0.043420, 0.008998, 0.449146, -8.421447, 8.845381), (5.000000, 0.111132, 0.305883, 0.764563, 0.733436)),
    1.390: ((0.399699, 0.042606, -9.709411, 10.197882, 0.000974), (0.279229, 5.000000, 0.724163, 0.695112, 0.020000)),
    1.480: ((1.191069, 0.024664, 0.228549, -11.225060, 10.722235), (0.475186, 5.000000, 0.229079, 1.263080, 1.282078)),
    1.533: ((10.952809, 0.078917, -10.528644, 0.441657, 0.000904), (0.646781, 2.639525, 0.679297, 0.265388, 0.020000)),
    1.535: ((9.949091, 0.079989, -9.526490, 0.442325, 0.000904), (0.644603, 2.616054, 0.680586, 0.265217, 0.020000)),
    1.544: ((11.888622, 0.084790, -11.474378, 0.446671, 0.000906), (0.645135, 2.518188, 0.675509, 0.264646, 0.020000)),
    1.575: ((10.594965, 0.105832, -10.214015, 0.461586, 0.000915), (0.635652, 2.213366, 0.671443, 0.262659, 0.020000)),
    1.621: ((10.172360, 0.157843, -9.878495, 0.499770, 0.001594), (0.627966, 1.834908, 0.668194, 0.262846, 0.040788)),
    1.625: ((10.471807, 0.164206, -10.188811, 0.504461, 0.001729), (0.628263, 1.805441, 0.667597, 0.263109, 0.043757)),
    1.676: ((9.248535, 0.301261, -9.180334, 0.582300, 0.005573), (0.627473, 1.464261, 0.677287, 0.269422, 0.084456)),
    1.687: ((8.372097, 0.356592, -8.383113, 0.605186, 0.007391), (0.627148, 1.396436, 0.684005, 0.271785, 0.093699)),
    1.699: ((8.378817, 0.438167, -8.502912, 0.634758, 0.010202), (0.630767, 1.324242, 0.689883, 0.275080, 0.103994)),
    1.710: ((7.625314, 0.548863, -7.893201, 0.665149, 0.013700), (0.632549, 1.257294, 0.700536, 0.278629, 0.113206)),
    1.743: ((0.791655, 1.716598, -8.823619, 7.244555, 0.032945), (0.294145, 1.039626, 0.751101, 0.657687, 0.139535)),
    1.775: ((0.827365, 9.112820, -20.000000, 10.996936, 0.027280), (0.289479, 0.885449, 0.781420, 0.678950, 0.131306)),
    1.798: ((-20.000000, 10.946192, 0.011218, 0.772468, 9.236177), (0.761776, 0.655947, 0.101300, 0.273927, 0.864758)),
    1.831: ((-20.000000, 0.002300, 11.071928, 9.189659, 0.704487), (0.735783, 0.046748, 0.628369, 0.839436, 0.257332)),
    1.864: ((-20.000000, 0.000996, 0.670853, 8.997221, 11.301613), (0.713905, 0.020000, 0.248501, 0.819747, 0.607602)),
    1.875: ((0.664423, 9.023862, -19.999981, 11.282198, 0.000946), (0.246457, 0.813538, 0.707949, 0.601218, 0.020000)),
    1.897: ((0.650136, 8.945800, -20.000000, 11.376170, 0.000841), (0.242273, 0.801813, 0.695287, 0.589032, 0.020000)),
    1.900: ((0.648006, 8.961781, -20.000000, 11.362536, 0.000826), (0.241689, 0.800124, 0.693764, 0.587302, 0.020000)),
    1.908: ((0.642181, 8.946104, -19.999992, 11.384606, 0.000784), (0.240122, 0.795879, 0.689317, 0.582892, 0.020000)),
    1.919: ((0.633638, 9.009043, -20.000000, 11.331005, 0.000726), (0.237927, 0.789751, 0.683844, 0.576603, 0.020000)),
    1.925: ((0.628901, 8.896102, -19.999983, 11.449091, 0.000693), (0.236725, 0.787025, 0.679849, 0.573673, 0.020000)),
    1.930: ((0.624600, 9.030399, -20.000000, 11.319466, 0.000665), (0.235694, 0.783848, 0.678138, 0.570502, 0.020000)),
    1.952: ((0.605202, 8.861956, -20.000000, 11.508851, 0.000534), (0.231117, 0.773066, 0.665417, 0.559160, 0.020000)),
    1.955: ((0.602227, 8.986133, -20.000000, 11.387859, 0.000515), (0.230465, 0.771015, 0.664737, 0.557154, 0.020000)),
    1.985: ((17.226012, -17.858030, 18.847968, -17.791862, 0.554142), (0.719870, 0.644561, 0.560541, 0.644434, 0.220741)),
    2.007: ((-18.326091, -18.168533, 19.534415, 17.389401, 0.550480), (0.634539, 0.634698, 0.552706, 0.710365, 0.218256)),
    2.029: ((17.046276, -18.011265, 19.339383, -17.938035, 0.544739), (0.702379, 0.625580, 0.543271, 0.625578, 0.215634)),
    2.040: ((19.268644, 17.157452, 0.541601, -18.612676, -17.373223), (0.538709, 0.697850, 0.214314, 0.621397, 0.621461)),
    2.063: ((-17.711492, -17.506975, 18.827561, 16.840684, 0.533473), (0.612693, 0.612457, 0.528569, 0.689693, 0.211455)),
    2.096: ((19.177470, 16.858926, 0.520877, -18.333753, -17.238232), (0.516122, 0.676676, 0.207338, 0.598977, 0.599198)),
    2.118: ((-17.242364, -17.044092, 18.456554, 16.307131, 0.509404), (0.590880, 0.591205, 0.506051, 0.669820, 0.204379)),
    2.162: ((-17.083899, -16.955671, 18.260667, 16.283074, 0.485090), (0.574530, 0.574927, 0.488850, 0.653486, 0.198423)),
    2.184: ((18.413436, 16.392782, 0.471691, -17.128237, -17.159123), (0.480954, 0.645071, 0.195373, 0.566657, 0.566310)),
    2.228: ((-17.229641, 18.131764, 0.440028, -16.396994, 16.047948), (0.550546, 0.464253, 0.188887, 0.550679, 0.630152)),
    2.316: ((18.056015, 0.367094, 16.093185, -16.616976, -16.901198), (0.433707, 0.174955, 0.599928, 0.520903, 0.520489)),
    2.382: ((-17.158383, 18.447566, 0.307549, -17.183879, 16.588990), (0.499406, 0.413366, 0.163607, 0.499709, 0.577344)),
    2.504: ((-19.999999, 10.367010, 0.581241, -5.830578, 15.867860), (0.537940, 0.394496, 0.180664, 0.921522, 0.752106)),
    2.548: ((-17.468402, -16.644927, 18.544741, 16.422536, 0.157509), (0.451505, 0.451950, 0.366381, 0.529562, 0.129957)),
    2.746: ((17.321263, 14.669991, 3.629828, -19.389181, -15.255718), (0.789437, 0.790134, 0.240829, 0.635576, 0.911289)),
    2.956: ((-5.795445, 19.702705, 4.981062, -11.100569, -6.814327), (0.507411, 0.668781, 0.234347, 0.507935, 0.861198)),
    3.033: ((-15.473379, 16.827350, 5.550526, -19.850935, 13.918479), (0.765474, 0.654276, 0.231595, 0.487320, 0.654953)),
    3.342: ((11.892662, 12.675366, 10.804241, -20.000000, -14.412729), (0.582761, 0.582955, 0.227480, 0.353211, 0.684208)),
    3.783: ((-5.568120, -15.389160, 19.257574, -17.013345, 19.636880), (0.673191, 0.282086, 0.204738, 0.281411, 0.442603)),
}


def sphere_coefficients(radius, width=EDGE_WIDTH_DEFAULT):
    """(A, lam) for a soft sphere: from SPHERE_FIT_TABLE when the radius and
    edge are the defaults' (no fit, no scipy, nothing to wait for), else a
    live gaussian_fit, cached per process."""
    key = round(float(radius), 3)
    if abs(width - EDGE_WIDTH_DEFAULT) < 1e-9 and key in SPHERE_FIT_TABLE:
        return SPHERE_FIT_TABLE[key]
    return gaussian_fit(key, width)


@functools.lru_cache(maxsize=None)
def gaussian_fit(radius, width=EDGE_WIDTH_DEFAULT, n_terms=5, amp_max=20.0, seeds=2):
    """(A, lam) of n_terms centred Gaussians fitting the Fermi sphere, by
    bounded least squares weighted by r (the volume weight, softened at the
    origin); the best of several starts. Cached per (radius, width)."""
    from scipy.optimize import least_squares

    r = np.linspace(0.0, radius + 3.0, 300)
    target = fermi(r, radius, width)
    weight = np.sqrt(r * r + 0.05)

    def model(p):
        A, lam = p[:n_terms], np.exp(p[n_terms:])
        return (A[:, None] * np.exp(-lam[:, None] * r[None, :] ** 2)).sum(0)

    best = None
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        p0 = np.concatenate([rng.uniform(-1, 2, n_terms), np.log(rng.uniform(0.05, 1.5, n_terms))])
        res = least_squares(lambda p: (model(p) - target) * weight, p0, max_nfev=1500,
                            bounds=([-amp_max] * n_terms + [np.log(0.02)] * n_terms,
                                    [amp_max] * n_terms + [np.log(5.0)] * n_terms))
        if best is None or res.cost < best.cost:
            best = res
    A, lam = best.x[:n_terms], np.exp(best.x[n_terms:])
    return tuple(A.tolist()), tuple(lam.tolist())


def sphere_kernels(radii, grid_shape, orth_matrix_np, width=EDGE_WIDTH_DEFAULT,
                   taper_width=0.1, device="cpu", dtype=None):
    """Per-atom (A, lam) and the offset structure compute_density wants, for
    atoms of the given sphere radii: one offset group (the kernels are keyed
    by radius, not element), sized for the largest sphere. Returns
    (atom_A, atom_lam, elem_offsets, atom_radius_ang, taper_width,
    element_idx) with element_idx all zeros."""
    import torch

    from .density_torch import precompute_element_offsets

    dtype = dtype or torch.float64
    radii = np.asarray(radii, dtype=float)
    # one fit per distinct radius, gathered back per atom
    distinct, inverse = np.unique(radii, return_inverse=True)
    coeffs = [sphere_coefficients(d, width) for d in distinct]
    A = np.array([c[0] for c in coeffs])[inverse]
    lam = np.array([c[1] for c in coeffs])[inverse]
    # the sphere's reach: the Fermi tail is below 1% by d + 5 w; the taper
    # (density_torch's, the same raised cosine) then takes it to zero
    reach = radii + 5.0 * width
    spacing = np.linalg.norm(np.asarray(orth_matrix_np), axis=0) / np.asarray(grid_shape, dtype=float)
    r_max = float(reach.max()) + taper_width
    offsets = precompute_element_offsets(
        torch.tensor(np.ceil(r_max / spacing).astype(np.int64)[None, :], device=device),
        torch.tensor([r_max], device=device, dtype=dtype),
        torch.tensor(np.asarray(orth_matrix_np), device=device, dtype=dtype), grid_shape, device)
    n = len(radii)
    return (torch.tensor(A, device=device, dtype=dtype), torch.tensor(lam, device=device, dtype=dtype),
            offsets, torch.tensor(reach + taper_width, device=device, dtype=dtype), taper_width,
            torch.zeros(n, dtype=torch.long, device=device))
