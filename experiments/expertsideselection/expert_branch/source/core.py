# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from common import *
import sys, platform
BASE = Path(__file__).resolve().parents[2]
OUT = BASE / 'outputs/v284M_r1'
OLD = Path(_p1_path('path_942393e607f5'))
R2 = Path(_p1_path('path_7e04651960a0'))
DOMAIN = ('RedPajamaArXiv', 'RedPajamaGithub', 'RedPajamaWikipedia', 'RedPajamaStackExchange')
torch.set_num_threads(4)

def write(n, x):
    (OUT / n).parent.mkdir(parents=True, exist_ok=True)
    (OUT / n).write_text(json.dumps(x, indent=2, allow_nan=False, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)) + '\n')

def csvwrite(n, rows):
    with (OUT / n).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def rngkey(*args):
    key = 'v284M-expert-selection|' + '|'.join(map(str, args))
    return (np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:16], 'little')), key)

def unit(rng, n):
    x = rng.standard_normal(n)
    return x / np.linalg.norm(x)

def direction(f, g, seed, r):
    d, D = DIM[f]
    m = D // g
    rng, key = rngkey(f, g, seed, r)
    a = unit(rng, m)
    b = unit(rng, d)
    e = unit(rng, d)
    b2 = unit(rng, d)
    nb = 3
    v = {'a': a, 'b': b / math.sqrt(nb), 'e': e / math.sqrt(nb), 'b2': b2 / math.sqrt(nb), 'bias': a / math.sqrt(nb), 'key': key}
    parent = rng.standard_normal(4)
    parent -= parent.mean()
    parent /= np.linalg.norm(parent)
    child = rng.standard_normal(g)
    child -= child.mean()
    if g > 1:
        child /= np.linalg.norm(child)
    v['parent'] = parent
    v['child'] = child
    return v

def local_params(s, f, g, c):
    m = DIM[f][1] // g
    sl = slice(c * m, (c + 1) * m)
    if f == 'S1':
        return (s['dense_h_to_4h.weight'][sl].double(), s['dense_h_to_4h.bias'][sl].double(), s['dense_4h_to_h.weight'][:, sl].double() * g)
    return (s['gate_proj.weight'][sl].double(), s['up_proj.weight'][sl].double(), s['down_proj.weight'][:, sl].double() * g)

def local_direction(v, f):
    a = torch.from_numpy(v['a'])
    b = torch.from_numpy(v['b'])
    e = torch.from_numpy(v['e'])
    b2 = torch.from_numpy(v['b2'])
    return (a[:, None] * b[None, :], torch.from_numpy(v['bias']) if f == 'S1' else a[:, None] * b2[None, :], e[:, None] * a[None, :])

def prep(h, s, f):
    h = torch.from_numpy(h).double()
    s = {k: v.double() for k, v in s.items()}
    if f == 'S1':
        z = F.linear(h, s['dense_h_to_4h.weight'], s['dense_h_to_4h.bias'])
        pdf = torch.exp(-z * z / 2) / math.sqrt(2 * math.pi)
        phi = F.gelu(z, approximate='none')
        a = 0.5 * (1 + torch.erf(z / math.sqrt(2))) + z * pdf
        second = (2 - z * z) * pdf
        return {k: v.numpy() for k, v in {'h': h, 'phi': phi, 'a': a, 'second': second, 'V': s['dense_4h_to_h.weight']}.items()}
    z = F.linear(h, s['gate_proj.weight'])
    u = F.linear(h, s['up_proj.weight'])
    sig = torch.sigmoid(z)
    act = z * sig
    first = sig + z * sig * (1 - sig)
    second = sig * (1 - sig) * (2 + z * (1 - 2 * sig))
    return {k: v.numpy() for k, v in {'h': h, 'phi': act * u, 'a': first * u, 'b': act, 'u': u, 'first': first, 'second': second, 'V': s['down_proj.weight']}.items()}

def derivative(p, f, g, v, src, tgt):
    d, D = DIM[f]
    m = D // g
    h = p['h']
    a = v['a']
    dz = (h @ v['b'])[:, None] * a[None, :]
    if f == 'S1':
        dz += v['bias'][None, :]
    else:
        du = (h @ v['b2'])[:, None] * a[None, :]
    dot = np.sum(h[src] * h[tgt], axis=1) + (1 if f == 'S1' else 0)
    n = len(src)
    full = np.empty((n, g))
    restricted = np.empty_like(full)
    for c in range(g):
        sl = slice(c * m, (c + 1) * m)
        phi = p['phi'][:, sl]
        basea = p['a'][:, sl]
        V = g * p['V'][:, sl]
        norm = np.sum(V * V, axis=0) / d
        dnorm = 2 * (v['e'] @ V) * a / d
        if f == 'S1':
            dphi = basea * dz
            da = p['second'][:, sl] * dz
        else:
            baseb = p['b'][:, sl]
            dphi = basea * dz + baseb * du
            da = p['second'][:, sl] * p['u'][:, sl] * dz + p['first'][:, sl] * du
            db = p['first'][:, sl] * dz
        for lo in range(0, n, 256):
            ix = src[lo:lo + 256]
            iy = tgt[lo:lo + 256]
            output = np.sum(dphi[ix] * phi[iy] + phi[ix] * dphi[iy], axis=1) / m
            z = basea[ix] * basea[iy]
            dzp = da[ix] * basea[iy] + basea[ix] * da[iy]
            if f == 'S2':
                z += baseb[ix] * baseb[iy]
                dzp += db[ix] * baseb[iy] + baseb[ix] * db[iy]
            full[lo:lo + 256, c] = output + dot[lo:lo + 256] * np.sum(dzp * norm + z * dnorm, axis=1) / m
            restricted[lo:lo + 256, c] = output
    return (full, restricted)

def torch_kernel(hx, hy, params, f):
    W, U, V = params
    d = V.shape[0]
    m = V.shape[1]
    dot = (hx * hy).sum(1) + (1 if f == 'S1' else 0)
    zx = F.linear(hx, W, U if f == 'S1' else None)
    zy = F.linear(hy, W, U if f == 'S1' else None)
    if f == 'S1':
        phix = F.gelu(zx, approximate='none')
        phiy = F.gelu(zy, approximate='none')

        def prime(z):
            return 0.5 * (1 + torch.erf(z / math.sqrt(2))) + z * torch.exp(-z * z / 2) / math.sqrt(2 * math.pi)
        z = prime(zx) * prime(zy)
    else:
        ux = F.linear(hx, U)
        uy = F.linear(hy, U)
        sx = torch.sigmoid(zx)
        sy = torch.sigmoid(zy)
        ax = zx * sx
        ay = zy * sy
        phix = ax * ux
        phiy = ay * uy
        dx = sx + zx * sx * (1 - sx)
        dy = sy + zy * sy * (1 - sy)
        z = dx * ux * dy * uy + ax * ay
    return (phix * phiy).sum(1) / m + dot * (z * (V * V).sum(0) / d).sum(1) / m

def scalar_derivative(x, y, params, vs, f):
    params = tuple((t.detach().clone().requires_grad_() for t in params))
    W, U, V = params
    d, m = V.shape

    def forward(h):
        return F.linear(F.gelu(F.linear(h, W, U), approximate='none'), V) if f == 'S1' else F.linear(F.silu(F.linear(h, W)) * F.linear(h, U), V)
    fx, fy = (forward(x), forward(y))
    terms = []
    for j in range(d):
        gx = torch.autograd.grad(fx[j], params, retain_graph=True, create_graph=True)
        gy = torch.autograd.grad(fy[j], params, retain_graph=True, create_graph=True)
        k = sum(((a * b).sum() for a, b in zip(gx, gy)))
        dk = torch.autograd.grad(k, params, retain_graph=True)
        terms.append(float(sum(((a * b).sum() for a, b in zip(dk, vs))).detach()))
    return math.fsum(terms) / (d * m)

def qfield(h, w, g, s, src, tgt, pref=None):
    w = w.reshape(4, g, -1)[:, 0]
    z = h @ w.T
    z -= z.mean(1, keepdims=True)

    def co(s):
        t = s * z
        t -= t.max(1, keepdims=True)
        p = np.exp(t)
        p /= p.sum(1, keepdims=True)
        p = np.repeat(p / g, g, axis=1)
        return p[src] * p[tgt]
    u = co(1)
    uc = u - u.mean(1, keepdims=True)
    pref = float(np.mean(np.sum(uc * uc, axis=1) / (4 * g - 1))) if pref is None else pref
    assert pref > 0
    u = co(s)
    return ((u - u.mean(1, keepdims=True)) / math.sqrt(pref), u, math.sqrt(pref), pref)

def aggregate(x, group):
    return np.array([x[group == i].mean() for i in range(12)])

def actual(dk, u, root, group):
    return aggregate((np.sum(u * dk, axis=1) - u.sum(1) * dk.mean(1)) / root, group)

def field_for(d, f, g, v, cls):
    return (d[:, None, :] * v['parent'][None, :, None] / math.sqrt(g) if cls == 'parent' else d[:, None, :] * v['child'][None, None, :] / 2 * np.ones((1, 4, 1))).reshape(-1, 4 * g)
