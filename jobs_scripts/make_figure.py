"""Compare how t-SMILES, fragSMILES, SAFE and SPE tokenise the same molecule.

t-SMILES / fragSMILES / SAFE strings below are baked in, taken from each
method's reference implementation. SPE is computed live with a small
from-scratch tokenizer (no `SmilesPE` package needed) against the published
ChEMBL SPE vocabulary in data/external/SPE_ChEMBL.txt (downloaded on first
run if missing).
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

PAL = ['#2c6fb5', '#d9822b', '#3f8f5b', '#a44a9c', '#c0504d', '#5f8fb0',
       '#8a6d3b', '#4d4d4d']


def hex2rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def light(h, f=0.74):
    r, g, b = hex2rgb(h)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


# bonds each method cuts, as (atom_i, atom_j) pairs
CUTS = {
    'tsmiles': [(6, 7), (7, 8)],
    'frag':    [(0, 1), (1, 2), (5, 6), (6, 7), (7, 8), (8, 10), (15, 16)],
    'safe':    [(1, 2), (5, 6), (6, 7), (7, 8), (8, 10)],
}


def atom_groups(cuts):
    bond_idx = [mol.GetBondBetweenAtoms(i, j).GetIdx() for i, j in cuts]
    frags = Chem.rdmolops.FragmentOnBonds(mol, bond_idx, addDummies=False)
    return list(Chem.GetMolFrags(frags))


# ---- SPE tokenizer, self-contained (no SmilesPE dependency) ---------------
SPE_VOCAB = EXTERNAL_DATA_DIR / 'SPE_ChEMBL.txt'
SPE_VOCAB_URL = ('https://raw.githubusercontent.com/XinhaoLi74/SmilesPE/'
                  'master/SPE_ChEMBL.txt')

if not SPE_VOCAB.exists():
    print(f'[SPE] vocabulary not found, downloading to {SPE_VOCAB} ...')
    SPE_VOCAB.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(SPE_VOCAB_URL, SPE_VOCAB)

_ATOM_RE = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/"
    r"|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])")


def atomwise_tokenizer(smi):
    return _ATOM_RE.findall(smi)


class SPETokenizer:
    """Reimplementation of SmilesPE.tokenizer.SPE_Tokenizer."""

    def __init__(self, vocab_path):
        with codecs.open(vocab_path, encoding='utf-8') as fh:
            codes = [tuple(l.strip('\r\n ').split(' ')) for l in fh]
        codes = [c for c in codes if len(c) == 2]
        self.bpe = {c: i for i, c in reversed(list(enumerate(codes)))}

    def tokenize(self, smi):
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
        return word


spe = SPETokenizer(SPE_VOCAB)
SPE_TOKENS = spe.tokenize(SMI)
ATOMWISE = atomwise_tokenizer(SMI)


def spe_atom_groups():
    """Map each SPE token to the RDKit atom indices it covers."""
    non_atom = set('()=#-+/\\.@%') | set('0123456789')
    groups, ai, pos = [], 0, 0
    for tok in SPE_TOKENS:
        grp, consumed = [], 0
        while consumed < len(tok):
            aw = ATOMWISE[pos]
            if aw[0] not in non_atom:
                grp.append(ai); ai += 1
            consumed += len(aw); pos += 1
        groups.append(grp)
    return groups


# ---- molecule rendering ----------------------------------------------------
def mol_svg(groups, w, h):
    """Draw the molecule, each fragment group on its own soft colour."""
    acols, bcols, hatoms, hbonds = {}, {}, [], []
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
    o.highlightRadius = 0.32
    o.clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, mol, highlightAtoms=hatoms, highlightAtomColors=acols,
        highlightBonds=hbonds, highlightBondColors=bcols)
    d.FinishDrawing()
    body = re.sub(r'^.*?<svg[^>]*>', '', d.GetDrawingText(), flags=re.S)
    body = body.replace('</svg>', '')
    return re.sub(r'<rect[^>]*fill:#FFFFFF[^>]*/>', '', body)


def cut_marks(cuts, w, h):
    """Dashed mark centred on each cut bond's actual drawn line segment.

    RDKit shortens a bond's line where it meets a labelled atom (e.g. 'HN',
    'Cl'), so the true visual midpoint isn't the atom-centre midpoint -
    it's read back from the rendered SVG path for that bond.
    """
    d = rdMolDraw2D.MolDraw2DSVG(w, h)
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    svg = d.GetDrawingText()
    out = []
    for i, j in cuts:
        bidx = mol.GetBondBetweenAtoms(i, j).GetIdx()
        segs = re.findall(rf"class='bond-{bidx} [^']*' d='M ([\d.-]+),([\d.-]+) L ([\d.-]+),([\d.-]+)'", svg)
        xs = [float(v) for x1, y1, x2, y2 in segs for v in (x1, x2)]
        ys = [float(v) for x1, y1, x2, y2 in segs for v in (y1, y2)]
        mx, my = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        p1, p2 = d.GetDrawCoords(i), d.GetDrawCoords(j)
        dx, dy = p2.x - p1.x, p2.y - p1.y
        L = (dx * dx + dy * dy) ** 0.5 or 1
        nx, ny = -dy / L * 7, dx / L * 7
        out.append(
            f'<line x1="{mx - nx:.1f}" y1="{my - ny:.1f}" x2="{mx + nx:.1f}" '
            f'y2="{my + ny:.1f}" stroke="#c0392b" stroke-width="2.2" '
            f'stroke-linecap="round"/>')
    return ''.join(out)


# ---- text helpers -----------------------------------------------------------
def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def token_run(tokens, cx, y, size=12.4, gap=5.0, colours=None, maxw=592):
    """Lay out coloured token chips, each wrapped line centred on cx. Returns (svg, y_end)."""
    cw = size * 0.605
    widths = [max(len(t) * cw + 9, 15) for t in tokens]

    lines, cur, cur_w = [], [], 0
    for k, tw in enumerate(widths):
        extra = tw if not cur else tw + gap
        if cur and cur_w + extra > maxw:
            lines.append(cur); cur, cur_w = [], 0
            extra = tw
        cur.append(k); cur_w += extra
    if cur:
        lines.append(cur)

    parts, cy = [], y
    for li, line in enumerate(lines):
        if li:
            cy += size + 11
        line_w = sum(widths[k] for k in line) + gap * (len(line) - 1)
        px = cx - line_w / 2
        for k in line:
            t, tw = tokens[k], widths[k]
            col = colours[k] if colours else '#333'
            parts.append(
                f'<rect x="{px:.1f}" y="{cy - size + 2.5:.1f}" width="{tw:.1f}" '
                f'height="{size + 6:.1f}" rx="2.6" fill="{col}" fill-opacity="0.13" '
                f'stroke="{col}" stroke-opacity="0.55" stroke-width="0.7"/>')
            parts.append(
                f'<text x="{px + tw / 2:.1f}" y="{cy + 3.6:.1f}" text-anchor="middle" '
                f'font-family="DejaVu Sans Mono, Consolas, monospace" '
                f'font-size="{size}" fill="{col}">{esc(t)}</text>')
            px += tw + gap
    return ''.join(parts), cy


# ==============================================================================
W, PAD, MW, MH = 1180, 30, 220, 132
NCOLS, GUTTER = 2, 40
COL_W = (W - 2 * PAD - GUTTER) / NCOLS

TS_TOKENS = ['[2*]CC1=CC=C(OC)C=C1', '&', '[1*]N[2*]', '&',
             '[1*]C(=O)C1=CC=CC=C1Cl', '&', '&', '&']
TS_COLS = [PAL[0], '#8a8a8a', PAL[1], '#8a8a8a', PAL[2], '#8a8a8a',
           '#8a8a8a', '#8a8a8a']

FRAG_TOKENS = ['C', 'O', '<0>c1ccccc1<3>', 'C', 'N', 'C=O',
               '<0>c1ccccc1<5>', 'Cl']

SAFE_TOKENS = ['c14ccc7cc1', 'c16ccccc1Cl', 'C36=O', 'CO4', 'C57', 'N35']
SAFE_ORDER = [1, 5, 4, 0, 2, 3]  # SAFE block k -> fragment group


def dot_join(tokens, colours):
    """Interleave a '.' chip between blocks, as in the real block-separated string."""
    out_tok, out_col = [tokens[0]], [colours[0]]
    for t, c in zip(tokens[1:], colours[1:]):
        out_tok.append('.'); out_col.append('#8a8a8a')
        out_tok.append(t); out_col.append(c)
    return out_tok, out_col


FRAG_COLS = [PAL[i % len(PAL)] for i in range(8)]
SAFE_COLS = [PAL[SAFE_ORDER[k] % len(PAL)] for k in range(6)]
FRAG_TOKENS, FRAG_COLS = dot_join(FRAG_TOKENS, FRAG_COLS)
SAFE_TOKENS, SAFE_COLS = dot_join(SAFE_TOKENS, SAFE_COLS)

METHODS = [
    dict(name='t-SMILES',
         cuts=CUTS['tsmiles'], groups=atom_groups(CUTS['tsmiles']),
         tokens=TS_TOKENS, colours=TS_COLS),
    dict(name='fragSMILES',
         cuts=CUTS['frag'], groups=atom_groups(CUTS['frag']),
         tokens=FRAG_TOKENS, colours=FRAG_COLS),
    dict(name='SAFE',
         cuts=CUTS['safe'], groups=atom_groups(CUTS['safe']),
         tokens=SAFE_TOKENS, colours=SAFE_COLS),
    dict(name='SPE',
         cuts=[], groups=spe_atom_groups(),
         tokens=SPE_TOKENS, colours=[PAL[k % len(PAL)] for k in range(len(SPE_TOKENS))]),
]

def method_panel(letter, colx, col_w, y, M):
    """Render one method panel (molecule + name + tokens) at (colx, y). Returns bottom y."""
    cx = colx + col_w / 2
    body.append(f'<text x="{colx}" y="{y + 13}" font-size="13" font-weight="700" '
                f'fill="#111">{letter}</text>')
    marks = cut_marks(M['cuts'], MW, MH) if M['cuts'] else ''
    body.append(f'<g transform="translate({colx + (col_w - MW) / 2:.1f},{y + 20})">'
                f'{mol_svg(M["groups"], MW, MH)}{marks}</g>')
    title_y = y + 20 + MH + 20
    body.append(f'<text x="{cx}" y="{title_y}" font-size="13.5" font-weight="700" '
                f'text-anchor="middle" fill="#111">{M["name"]}</text>')
    run, yend = token_run(M['tokens'], cx, title_y + 20, size=10.6, gap=4,
                           colours=M['colours'], maxw=col_w - 20)
    body.append(run)
    return max(yend + 14, y + 20 + MH + 14)


body, y = [], PAD

# ---- panel (a): reference molecule, full width -----------------------------
FULL_CX = PAD + (W - 2 * PAD) / 2
body.append(f'<text x="{PAD}" y="{y + 13}" font-size="13" font-weight="700" '
            f'fill="#111">a</text>')
body.append(f'<g transform="translate({FULL_CX - MW / 2:.1f},{y + 20})">{mol_svg([], MW, MH)}</g>')
body.append(f'<text x="{FULL_CX}" y="{y + 20 + MH + 20}" font-size="13.5" font-weight="700" '
            f'text-anchor="middle" fill="#111">SMILES</text>')
body.append(f'<text x="{FULL_CX}" y="{y + 20 + MH + 40}" font-size="12" text-anchor="middle" '
            f'font-family="DejaVu Sans Mono, Consolas, monospace" fill="#222">{esc(SMI)}</text>')
run, yend = token_run(ATOMWISE, FULL_CX, y + 20 + MH + 64, size=10.4, gap=3,
                       colours=['#777'] * len(ATOMWISE), maxw=W - 2 * PAD - 40)
body.append(run)
y = max(yend + 14, y + 20 + MH + 14)
body.append(f'<line x1="{PAD}" y1="{y}" x2="{W - PAD}" y2="{y}" stroke="#e2e2e2"/>')
y += 22

# ---- panels (b)-(e): 2x2 grid of methods -----------------------------------
for row_letters, row_methods in [('bc', METHODS[:2]), ('de', METHODS[2:])]:
    bottoms = []
    for col, (letter, M) in enumerate(zip(row_letters, row_methods)):
        colx = PAD + col * (COL_W + GUTTER)
        bottoms.append(method_panel(letter, colx, COL_W, y, M))
    y = max(bottoms)
    if row_methods is not METHODS[2:]:
        body.append(f'<line x1="{PAD}" y1="{y}" x2="{W - PAD}" y2="{y}" stroke="#e2e2e2"/>')
    y += 22

H = y + PAD - 22
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',
       f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
svg.extend(body)
svg.append('</svg>')

out = OUT_DIR / 'fragmentation_methods_figure.svg'
out.write_text('\n'.join(svg))
print('written', out)
