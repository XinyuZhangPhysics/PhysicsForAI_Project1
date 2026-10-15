# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from common import *

def prepared(h, s, fam):
    h = torch.as_tensor(h, dtype=torch.float64)
    s = {k: v.double() for k, v in s.items()}
    if fam == 'S1':
        z = F.linear(h, s['dense_h_to_4h.weight'], s['dense_h_to_4h.bias'])
        phi = F.gelu(z, approximate='none')
        der = 0.5 * (1 + torch.erf(z / math.sqrt(2))) + z * torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        return {'h': h.numpy(), 'phi': phi.numpy(), 'a': der.numpy(), 'V': s['dense_4h_to_h.weight'].numpy()}
    z = F.linear(h, s['gate_proj.weight'])
    u = F.linear(h, s['up_proj.weight'])
    sig = torch.sigmoid(z)
    act = z * sig
    der = sig + z * sig * (1 - sig)
    return {'h': h.numpy(), 'phi': (act * u).numpy(), 'a': (der * u).numpy(), 'b': act.numpy(), 'V': s['down_proj.weight'].numpy()}

def kernel(p, fam, g, src, tgt, probe=None):
    d, D = DIM[fam]
    m = D // g
    n = len(src)
    phi = p['phi']
    h = p['h']
    dot = np.einsum('nd,nd->n', h[src], h[tgt])
    dot = dot + 1 if fam == 'S1' else dot
    norms = np.sum((g * p['V']) ** 2, axis=0) / d if probe is None else np.square(probe @ (g * p['V']))
    output = np.empty((n, g))
    input = np.empty_like(output)
    for c in range(g):
        sl = slice(c * m, (c + 1) * m)
        for lo in range(0, n, 256):
            ix = src[lo:lo + 256]
            iy = tgt[lo:lo + 256]
            output[lo:lo + 256, c] = np.sum(phi[ix, sl] * phi[iy, sl], axis=1) / m
            z = p['a'][ix, sl] * p['a'][iy, sl]
            if fam == 'S2':
                z = z + p['b'][ix, sl] * p['b'][iy, sl]
            input[lo:lo + 256, c] = dot[lo:lo + 256] * np.sum(z * norms[sl], axis=1) / m
    return (output + input, output, input)

def scalar_gradient_reference(x, y, s, fam, g, c):
    d, D = DIM[fam]
    m = D // g
    sl = slice(c * m, (c + 1) * m)
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    if fam == 'S1':
        w = s['dense_h_to_4h.weight'][sl].double().clone().requires_grad_()
        b = s['dense_h_to_4h.bias'][sl].double().clone().requires_grad_()
        v = (s['dense_4h_to_h.weight'][:, sl].double() * g).clone().requires_grad_()
        params = (w, b, v)

        def f(h):
            return F.linear(F.gelu(F.linear(h, w, b), approximate='none'), v)
    else:
        gate = s['gate_proj.weight'][sl].double().clone().requires_grad_()
        up = s['up_proj.weight'][sl].double().clone().requires_grad_()
        v = (s['down_proj.weight'][:, sl].double() * g).clone().requires_grad_()
        params = (gate, up, v)

        def f(h):
            return F.linear(F.silu(F.linear(h, gate)) * F.linear(h, up), v)
    fx = f(x)
    fy = f(y)
    terms = []
    downs = []
    for j in range(d):
        gx = torch.autograd.grad(fx[j], params, retain_graph=True)
        gy = torch.autograd.grad(fy[j], params, retain_graph=True)
        terms.append(sum((float(torch.sum(a * b)) for a, b in zip(gx, gy))))
        downs.append(float(torch.sum(gx[-1] * gy[-1])))
    return (math.fsum(terms) / (d * m), math.fsum(downs) / (d * m))

def down_reference(phi_x, phi_y, s, fam, g, c):
    d, D = DIM[fam]
    m = D // g
    sl = slice(c * m, (c + 1) * m)
    key = 'dense_4h_to_h.weight' if fam == 'S1' else 'down_proj.weight'
    v = (s[key][:, sl].double() * g).clone().requires_grad_()
    x = torch.as_tensor(phi_x, dtype=torch.float64)
    y = torch.as_tensor(phi_y, dtype=torch.float64)
    fx = F.linear(x, v)
    fy = F.linear(y, v)
    terms = []
    for j in range(d):
        gx = torch.autograd.grad(fx[j], v, retain_graph=True)[0]
        gy = torch.autograd.grad(fy[j], v, retain_graph=True)[0]
        terms.append(float(torch.sum(gx * gy)))
    return math.fsum(terms) / (d * m)
