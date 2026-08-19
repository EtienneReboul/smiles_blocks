"""Build a publication SVG comparing t-SMILES, fragSMILES, SAFE and SPE.

Dependencies
------------
    pip install rdkit          # the only third-party requirement

The t-SMILES / fragSMILES / SAFE strings shown in the figure are baked in as
constants (each was generated once with the respective reference
implementation and verified to round-trip). The SPE row is computed live by a
self-contained tokeniser (no `SmilesPE` package needed); it uses the published
ChEMBL SPE vocabulary `SPE_ChEMBL.txt`, expected in `data/external/` and
downloaded automatically on first run if absent.

Usage
-----
    python make_figure.py           # writes the SVG to the OUT path below

To use a different reference molecule, change SMI and regenerate the three
baked-in strings with the reference tools (see the project notes); the SPE row
updates on its own.
"""
import codecs
import re
import urllib.request
import warnings

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from smiles_blocks.files import EXTERNAL_DATA_DIR, FIGURES_DIR

warnings.filterwarnings('ignore')

OUT_DIR = FIGURES_DIR / 'fragmentation_notation_comparison'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SMI = 'COc1ccc(CNC(=O)c2ccccc2Cl)cc1'
mol = Chem.MolFromSmiles(SMI)
rdDepictor.SetPreferCoordGen(True)
rdDepictor.Compute2DCoords(mol)

# ---- palette (colour-blind safe, print friendly) -------------------------
PAL = ['#2c6fb5', '#d9822b', '#3f8f5b', '#a44a9c', '#c0504d', '#5f8fb0',
       '#8a6d3b', '#4d4d4d']
def hex2rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

# ---- fragment definitions (verified against each reference implementation)
CUTS = {
    'tsmiles':  [(6, 7), (7, 8)],
    'safe':     [(1, 2), (5, 6), (6, 7), (7, 8), (8, 10)],
    'frag':     [(0, 1), (1, 2), (5, 6), (6, 7), (7, 8), (8, 10), (15, 16)],
}

def atom_groups(cuts):
    """Return list of atom-index tuples produced by cutting `cuts`."""
    bidx = [mol.GetBondBetweenAtoms(i, j).GetIdx() for i, j in cuts]
    fm = Chem.rdmolops.FragmentOnBonds(mol, bidx, addDummies=False)
    return list(Chem.GetMolFrags(fm))

# --------------------------------------------------------------------------
# SMILES Pair Encoding (SPE), self-contained.
# No `SmilesPE` package required: the tokenizer below is a faithful
# reimplementation of SmilesPE.tokenizer.SPE_Tokenizer and was verified to
# produce byte-identical output. It needs the published ChEMBL SPE vocabulary
# (Li & Fourches 2021); the file lives in data/external/ and, if it is not
# found, is downloaded once from the authors' GitHub repository.
# --------------------------------------------------------------------------
SPE_VOCAB = EXTERNAL_DATA_DIR / 'SPE_ChEMBL.txt'
SPE_VOCAB_URL = ('https://raw.githubusercontent.com/XinhaoLi74/SmilesPE/'
                 'master/SPE_ChEMBL.txt')

if not SPE_VOCAB.exists():
    print(f'[SPE] vocabulary not found, downloading to {SPE_VOCAB} ...')
    SPE_VOCAB.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(SPE_VOCAB_URL, SPE_VOCAB)

_SMI_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/"
    r"|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])")

def atomwise_tokenizer(smi):
    """Atom-level SMILES tokeniser (Br/Cl and bracket atoms stay intact)."""
    return _SMI_PATTERN.findall(smi)

class SPETokenizer:
    """Standalone SPE tokeniser; matches SmilesPE.SPE_Tokenizer exactly."""
    def __init__(self, vocab_path):
        with codecs.open(vocab_path, encoding='utf-8') as fh:
            codes = [tuple(l.strip('\r\n ').split(' ')) for l in fh]
        codes = [c for c in codes if len(c) == 2]
        self.bpe = {c: i for i, c in reversed(list(enumerate(codes)))}
        self.cache = {}

    def tokenize(self, smi):
        if smi in self.cache:
            return list(self.cache[smi])
        word = atomwise_tokenizer(smi)
        while len(word) > 1:
            pairs = [(self.bpe[p], i, p)
                     for i, p in enumerate(zip(word, word[1:])) if p in self.bpe]
            if not pairs:
                break
            bigram = min(pairs)[2]
            positions = [i for _, i, p in pairs if p == bigram]
            merged, new_word, i = ''.join(bigram), [], 0
            for j in positions:
                if j < i:
                    continue
                new_word.extend(word[i:j]); new_word.append(merged); i = j + 2
            new_word.extend(word[i:]); word = new_word
        self.cache[smi] = tuple(word)
        return list(word)

spe = SPETokenizer(SPE_VOCAB)
SPE_TOKS = spe.tokenize(SMI)
ATOMWISE = atomwise_tokenizer(SMI)

def spe_atom_groups():
    """Map each SPE token to the RDKit atom indices it covers."""
    non_atom = set('()=#-+/\\.@%') | set('0123456789')
    groups, ai, pos = [], 0, 0
    for tok in SPE_TOKS:
        grp, consumed = [], 0
        while consumed < len(tok):
            aw = ATOMWISE[pos]
            if not (aw[0] in non_atom):          # it denotes a real atom
                grp.append(ai); ai += 1
            consumed += len(aw); pos += 1
        groups.append(grp)
    return groups

# ---- molecule rendering --------------------------------------------------
def light(h, f=0.74):
    r, g, b = hex2rgb(h)
    return (r + (1-r)*f, g + (1-g)*f, b + (1-b)*f)

def mol_svg(groups, w, h, plain=False):
    """Draw the molecule with each fragment on a soft coloured blob."""
    acols, bcols, hatoms, hbonds = {}, {}, [], []
    if not plain:
        for gi, grp in enumerate(groups):
            col = light(PAL[gi % len(PAL)])
            for a in grp:
                acols[a] = col; hatoms.append(a)
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i in acols and j in acols and acols[i] == acols[j]:
                bcols[b.GetIdx()] = acols[i]; hbonds.append(b.GetIdx())
    d = rdMolDraw2D.MolDraw2DSVG(w, h)
    o = d.drawOptions()
    o.addStereoAnnotation = False
    o.bondLineWidth = 1.5
    o.fillHighlights = True
    o.highlightBondWidthMultiplier = 20
    o.highlightRadius = 0.33
    o.clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, mol, highlightAtoms=hatoms, highlightAtomColors=acols,
        highlightBonds=hbonds, highlightBondColors=bcols)
    d.FinishDrawing()
    svg = d.GetDrawingText()
    body = re.sub(r'^.*?<svg[^>]*>', '', svg, flags=re.S)
    body = body.replace('</svg>', '')
    body = re.sub(r"<rect[^>]*fill:#FFFFFF[^>]*/>", '', body)
    return body, w, h

# scissor marks on cut bonds, drawn in molecule coordinate space
def cut_marks(cuts, w, h):
    d = rdMolDraw2D.MolDraw2DSVG(w, h)
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    out = []
    for i, j in cuts:
        p1 = d.GetDrawCoords(i)
        p2 = d.GetDrawCoords(j)
        mx, my = (p1.x + p2.x)/2, (p1.y + p2.y)/2
        dx, dy = p2.x - p1.x, p2.y - p1.y
        L = (dx*dx + dy*dy) ** .5 or 1
        nx, ny = -dy/L*7, dx/L*7
        out.append(
            f'<line x1="{mx-nx:.1f}" y1="{my-ny:.1f}" x2="{mx+nx:.1f}" '
            f'y2="{my+ny:.1f}" stroke="#c0392b" stroke-width="2.2" '
            f'stroke-linecap="round" stroke-dasharray="3,2"/>')
    return ''.join(out)

# ---- text helpers --------------------------------------------------------
def esc(s):
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))

def token_run(tokens, x, y, size=12.4, gap=5.0, colours=None, maxw=592):
    """Lay out coloured token chips, wrapping as needed. Returns (svg, y_end)."""
    parts, cx, cy = [], x, y
    cw = size * 0.605
    for k, t in enumerate(tokens):
        tw = max(len(t) * cw + 9, 15)
        if cx + tw > x + maxw:
            cx = x; cy += size + 11
        col = (colours[k] if colours else '#333')
        parts.append(
            f'<rect x="{cx:.1f}" y="{cy-size+2.5:.1f}" width="{tw:.1f}" '
            f'height="{size+6:.1f}" rx="2.6" fill="{col}" fill-opacity="0.13" '
            f'stroke="{col}" stroke-opacity="0.55" stroke-width="0.7"/>')
        parts.append(
            f'<text x="{cx+tw/2:.1f}" y="{cy+3.6:.1f}" text-anchor="middle" '
            f'font-family="DejaVu Sans Mono, Consolas, monospace" '
            f'font-size="{size}" fill="{col}">{esc(t)}</text>')
        cx += tw + gap
    return ''.join(parts), cy

# ==========================================================================
W, PAD = 1180, 34
rows = []

# --- row content ----------------------------------------------------------
ts_groups = atom_groups(CUTS['tsmiles'])
safe_groups = atom_groups(CUTS['safe'])
frag_groups = atom_groups(CUTS['frag'])
spe_groups = spe_atom_groups()

TS_TOKENS = ['[2*]CC1=CC=C(OC)C=C1', '&', '[1*]N[2*]', '&',
             '[1*]C(=O)C1=CC=CC=C1Cl', '&', '&', '&']
TS_COLS = [PAL[0], '#8a8a8a', PAL[1], '#8a8a8a', PAL[2], '#8a8a8a',
           '#8a8a8a', '#8a8a8a']

FRAG_TOKENS = ['C', 'O', '<0>c1ccccc1<3>', 'C', 'N', 'C=O',
               '<0>c1ccccc1<5>', 'Cl']
SAFE_TOKENS = ['c14ccc7cc1', 'c16ccccc1Cl', 'C36=O', 'CO4', 'C57', 'N35']
SAFE_ORDER = [1, 5, 4, 0, 2, 3]   # SAFE block k -> fragment group (verified)

METHODS = [
    dict(key='tsmiles', name='t-SMILES',
         cite='Wu et al., Nat. Commun. 2024',
         principle='Fragment the graph, arrange the fragments as a <tspan font-style="italic">full binary tree</tspan> (FBT), then serialise by',
         principle2='breadth-first traversal. Two extra symbols carry the topology.',
         groups=ts_groups, cuts=CUTS['tsmiles'],
         tokens=TS_TOKENS, colours=TS_COLS,
         note='<tspan font-weight="bold">&amp;</tspan> = placeholder node (no real fragment) \u00b7 <tspan font-weight="bold">^</tspan> = fragment separator \u00b7 joins are dummy atoms <tspan font-weight="bold">[n*]</tspan> matched in pairs.',
         note2='Topology lives in the tree, not in nested brackets \u2014 this shortens long-range grammatical dependencies.',
         tree=True),
    dict(key='frag', name='fragSMILES',
         cite='Mastrolorito et al., Commun. Chem. 2025',
         principle='Cleave <tspan font-style="italic">all exocyclic single bonds</tspan>, collapse the molecule to a <tspan font-style="italic">reduced graph</tspan> whose nodes are',
         principle2='canonical fragment SMILES, then write that graph as a string.',
         groups=frag_groups, cuts=CUTS['frag'],
         tokens=FRAG_TOKENS, colours=[PAL[i % len(PAL)] for i in range(8)],
         note='<tspan font-weight="bold">&lt;i&gt;</tspan> tags name the connecting atom of a fragment, so each fragment is encoded identically in every context (no redundancy).',
         note2='Chirality is written as an explicit <tspan font-weight="bold">R</tspan>/<tspan font-weight="bold">S</tspan> suffix, making it invariant to traversal order \u2014 unlike @/@@ in SMILES.'),
    dict(key='safe', name='SAFE',
         cite='Noutahi et al., Digital Discovery 2024',
         principle='Fragment (BRICS by default) and rewrite each fragment as a block; attachment points become',
         principle2='<tspan font-style="italic">ring-closure digits</tspan> and blocks are separated by dots.',
         groups=safe_groups, cuts=CUTS['safe'],
         tokens=SAFE_TOKENS,
         colours=[PAL[SAFE_ORDER[k] % len(PAL)] for k in range(6)],
         note='The string stays a <tspan font-weight="bold">syntactically valid SMILES</tspan> \u2014 any SMILES parser reads it, so no new tokeniser or decoder is required.',
         note2='Block order is unconstrained, which makes scaffold decoration and linker design expressible as simple sequence in-filling.'),
    dict(key='spe', name='SMILES Pair Encoding (SPE)',
         cite='Li &amp; Fourches, J. Chem. Inf. Model. 2021',
         principle='No chemical fragmentation at all: learn the most frequent adjacent-symbol merges from a corpus',
         principle2='(byte-pair encoding on atom-level SMILES tokens) and store them as a vocabulary.',
         groups=spe_groups, cuts=[],
         tokens=SPE_TOKS, colours=[PAL[k % len(PAL)] for k in range(len(SPE_TOKS))],
         note='Data-driven, not rule-driven: tokens are frequent <tspan font-style="italic">substrings</tspan>, so they need not be chemically meaningful.',
         note2='Here <tspan font-family="DejaVu Sans Mono, monospace" font-weight="bold">Cl)cc1</tspan> merges the chlorine with a branch-close and two atoms of the <tspan font-style="italic">other</tspan> ring \u2014 a valid token spanning no single substructure.',
         hatch=True),
]

def fbt(cx, y0):
    """Compact full binary tree for the 3-fragment decomposition."""
    N = {
        'F1': (cx,       y0,        PAL[0], 'F1'),
        'F2': (cx - 68,  y0 + 36,   PAL[1], 'F2'),
        'a':  (cx + 68,  y0 + 36,   '#9a9a9a', '&'),
        'F3': (cx - 104, y0 + 72,   PAL[2], 'F3'),
        'b':  (cx - 32,  y0 + 72,   '#9a9a9a', '&'),
        'c':  (cx - 126, y0 + 108,  '#9a9a9a', '&'),
        'd':  (cx - 82,  y0 + 108,  '#9a9a9a', '&'),
    }
    E = [('F1','F2','L'), ('F1','a','R'), ('F2','F3','L'),
         ('F2','b','R'), ('F3','c','L'), ('F3','d','R')]
    out = []
    for u, v, lab in E:
        x1, y1, _, _ = N[u]; x2, y2, _, _ = N[v]
        out.append(f'<line x1="{x1}" y1="{y1+11}" x2="{x2}" y2="{y2-11}" '
                   f'stroke="#b4b4b4" stroke-width="1.2"/>')
        mx, my = (x1+x2)/2, (y1+y2)/2
        out.append(f'<text x="{mx + (-8 if lab=="L" else 8)}" y="{my+4}" '
                   f'font-size="8.4" fill="#999" text-anchor="middle">{lab}</text>')
    for k, (x, yy, col, lab) in N.items():
        out.append(f'<circle cx="{x}" cy="{yy}" r="11.5" fill="{col}" '
                   f'fill-opacity="{0.16 if lab!="&" else 0.10}" '
                   f'stroke="{col}" stroke-width="1.2"/>')
        out.append(f'<text x="{x}" y="{yy+4}" text-anchor="middle" '
                   f'font-size="10.4" font-weight="bold" fill="{col}">'
                   f'{"&amp;" if lab=="&" else lab}</text>')
    out.append(f'<text x="{cx}" y="{y0+140}" text-anchor="middle" '
               f'font-size="10.6" fill="#666">full binary tree \u2014 '
               f'breadth-first read-out gives the string above</text>')
    return ''.join(out)

# ---------------------------------------------------------------- assemble
svg = []
MW, MH = 300, 172
ROW_H = 224
TS_ROW_H = 348
HEAD_H = 214
H = HEAD_H + ROW_H * 3 + TS_ROW_H + 66

svg.append(
    f'<svg xmlns="http://www.w3.org/2000/svg" '
    f'xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">')
svg.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

# ---- header
svg.append(f'<text x="{PAD}" y="34" font-size="19.5" font-weight="bold" '
           f'fill="#111">How four fragment-based notations tokenise the same '
           f'molecule</text>')
svg.append(f'<text x="{PAD}" y="55" font-size="12.6" fill="#555">Reference '
           f'compound: 2-chloro-<tspan font-style="italic">N</tspan>-'
           f'(4-methoxybenzyl)benzamide. Dashed red marks indicate cleaved '
           f'bonds; colours match fragments to their tokens.</text>')

body, w, h = mol_svg([], MW, MH, plain=True)
svg.append(f'<g transform="translate({PAD},70)">{body}</g>')

hx = PAD + MW + 26
svg.append(f'<text x="{hx}" y="94" font-size="12.8" font-weight="bold" '
           f'fill="#111">Input SMILES</text>')
svg.append(f'<text x="{hx}" y="116" font-size="13.4" '
           f'font-family="DejaVu Sans Mono, Consolas, monospace" '
           f'fill="#222">{esc(SMI)}</text>')
aw, _ = token_run(ATOMWISE, hx, 152, size=11.2, gap=3.0,
                  colours=['#666']*len(ATOMWISE), maxw=W-hx-PAD)
svg.append(f'<text x="{hx}" y="140" font-size="11.6" fill="#555">'
           f'Atom-level tokenisation \u2014 {len(ATOMWISE)} tokens, the baseline '
           f'every method below compresses:</text>')
svg.append(aw)
svg.append(f'<line x1="{PAD}" y1="{HEAD_H-12}" x2="{W-PAD}" y2="{HEAD_H-12}" '
           f'stroke="#d5d5d5" stroke-width="1"/>')

# ---- method rows
y = HEAD_H
for mi, M in enumerate(METHODS):
    if mi:
        svg.append(f'<line x1="{PAD}" y1="{y-12}" x2="{W-PAD}" y2="{y-12}" '
                   f'stroke="#e6e6e6" stroke-width="1"/>')
    body, w, h = mol_svg(M['groups'], MW, MH)
    marks = cut_marks(M['cuts'], MW, MH) if M['cuts'] else ''
    svg.append(f'<g transform="translate({PAD},{y+16})">{body}{marks}</g>')

    tx = PAD + MW + 26
    svg.append(f'<text x="{tx}" y="{y+30}" font-size="15.6" '
               f'font-weight="bold" fill="#111">{M["name"]}'
               f'<tspan font-size="11.2" font-weight="normal" fill="#777">'
               f'   {M["cite"]}</tspan></text>')
    svg.append(f'<text x="{tx}" y="{y+51}" font-size="12.1" fill="#333">'
               f'{M["principle"]}</text>')
    svg.append(f'<text x="{tx}" y="{y+67}" font-size="12.1" fill="#333">'
               f'{M["principle2"]}</text>')

    lbl = 'Resulting string' if M['key'] != 'spe' else 'Resulting tokens'
    svg.append(f'<text x="{tx}" y="{y+92}" font-size="11.3" '
               f'font-weight="bold" fill="#666">{lbl} '
               f'<tspan font-weight="normal">({len(M["tokens"])} tokens)'
               f'</tspan></text>')
    run, yend = token_run(M['tokens'], tx, y+114, colours=M['colours'],
                          maxw=W - tx - PAD)
    svg.append(run)

    ny = max(yend + 26, y + 150)
    svg.append(f'<text x="{tx}" y="{ny}" font-size="11.4" fill="#444">'
               f'{M["note"]}</text>')
    svg.append(f'<text x="{tx}" y="{ny+16}" font-size="11.4" fill="#444">'
               f'{M["note2"]}</text>')
    if M.get('tree'):
        svg.append(fbt(PAD + MW/2, y + 214))
        y += TS_ROW_H
    else:
        y += ROW_H

svg.append('</svg>')

out = OUT_DIR / 'fragmentation_methods_figure.svg'
out.write_text('\n'.join(svg))
print('written', out)
print('SPE tokens', SPE_TOKS)
print('SPE groups', spe_atom_groups())
print('spe groups', spe_groups)
