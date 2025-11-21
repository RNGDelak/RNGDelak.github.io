# bms_with_psi.py
# Your BMS ruler with a second line showing the converted ψ-expression.
# Usage: run normally; the window will auto-update and show:
# (0,0,0)(1,1,1)...
# ψ(Ω_{Ω_{Ω+3}}+2+1)

import copy
import re
import tkinter as tk
from tkinter import font
import sys
sys.setrecursionlimit(10**6)  # allow deep recursion for large matrices

# ------------------ BMS code (unchanged logic) ------------------
def extract(m, xy):
    x, y = xy
    if x > len(m) - 1 or x < 0 or y < 0:
        return -1
    if y > len(m[x]) - 1:
        return 0
    return m[x][y]

def compare(m1, m2):
    default = 1 if len(m1) > len(m2) else -1 if len(m2) > len(m1) else 0
    for i in range(min(len(m1), len(m2))):
        j = 0
        while 1:
            v1 = extract(m1, (i, j))
            v2 = extract(m2, (i, j))
            if v1 > v2:
                return 1
            if v2 > v1:
                return -1
            if v1 == 0:
                break
            j += 1
    return default

def lnz_row(m, x):
    y = 0
    while extract(m, [x, y + 1]):
        y += 1
    return y

def zero(m, x):
    if extract(m, [x, 0]) < 1:
        return 1
    return 0

def is_ancestor(m, xa, x, y):
    if xa == x:
        return 1
    if zero(m, x):
        return 0
    if y < 0:
        return 1
    p = x
    while p >= xa:
        if extract(m, [p, y]) < extract(m, [x, y]) and is_ancestor(m, p, x, y - 1):
            x = p
            if xa == x:
                return 1
        p -= 1
    return 0

def parent(m, x):
    if extract(m, [x, 0]) < 1:
        return -1
    y = lnz_row(m, x)
    p = x
    while p >= 0:
        p -= 1
        if is_ancestor(m, p, x, y):
            return p
    return -1

def asc_bp(bad_part, asc_col, n):
    abp = []
    y = max(len(asc_col), max(len(row) for row in bad_part))
    for i in range(len(bad_part)):
        abp.append([])
        for j in range(y):
            abp[i].append(extract(bad_part, [i, j]) + (extract([asc_col], [0, j]) * n if is_ancestor(bad_part, 0, i, j) else 0))
    return abp

def clean(m):
    for i in m:
        while len(i) > 1 and i[-1] == 0:
            i.pop()
        if len(i) == 0:
            i.append(0)
    return m

def type_of(m):
    if len(m) == 0:
        return 0
    if extract(m, (0, 0)) > 0:
        return 3
    if zero(m, len(m) - 1):
        return 1
    return 2

def expand(m, n):
    mat = copy.deepcopy(m)
    if type_of(mat) == 3:
        return [[0], [1] * (n + 1)]
    if type_of(mat) == 1:
        mat.pop()
        return clean(mat)
    if type_of(mat) == 0:
        return clean(mat)
    bri = parent(mat, len(mat) - 1)
    cut_child = mat[len(mat) - 1]
    asc_rows = lnz_row(mat, len(mat) - 1)
    mat.pop()
    bad_root = mat[bri]
    bad_part = mat[bri:]
    asc_col = []
    for i in range(asc_rows):
        asc_col.append(extract([cut_child], [0, i]) - extract([bad_root], [0, i]))
    for i in range(n):
        mat += asc_bp(bad_part, asc_col, i + 1)
    return clean(mat)

def mat_to_string(m):
    if extract(m, (0, 0)) > 0:
        return 'Limit'
    if extract(m, (0, 0)) < 0:
        return 'Empty matrix'
    return ''.join(['(' + ','.join(map(str, row)) + ')' for row in m])

slow_down = [
]

# New global override for slow variable (default 5)
# GUI will modify this value at runtime
slow_var_override = 5.0

def max_reached(fseq):
    slow = 0
    for s in slow_down:
        if len(fseq) >= len(s) and s[:-1] == fseq[:len(s) - 1] and fseq[len(s) - 1] >= s[-1]:
            slow += 1
    # incorporate GUI-controlled override 'slow_var_override' into threshold
    return sum(fseq) + len(fseq) / 10 - fseq[0] * 0.9 > 4 + slow + slow_var_override

def sequence_generator():
    last = []
    yield mat_to_string([])
    level = [[[1]]]
    fseq = [0]
    offset = [0]
    m = level[-1]

    while True:
        again = 0
        while not max_reached(fseq) and type_of(m) > 1:
            if again:
                level.append(m)
                fseq.append(0)
                offset.append(0)
            m = clean(expand(level[-1], fseq[-1] + offset[-1]))
            while compare(m, last) <= 0:
                offset[-1] += 1
                m = clean(expand(level[-1], fseq[-1] + offset[-1]))
            again += 1

        while not max_reached(fseq) and type_of(m) < 2:
            last = m
            yield mat_to_string(m)
            fseq[-1] += 1
            m = clean(expand(level[-1], fseq[-1] + offset[-1]))

        m = level[-1]
        if max_reached(fseq):
            level.pop()
            fseq.pop()
            offset.pop()
            last = m
            yield mat_to_string(m)
            if not compare(m, [[1]]):
                return
            fseq[-1] += 1

# ------------------ ψ-conversion code (converted & adapted) ------------------

# Basic data constants for ordinal terms
ZERO_ = []
ONE_ = [[], [], []]

def iz_(a):
    return not a

def eq_(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if isinstance(a, int) or isinstance(b, int):
        return False
    if iz_(a) or iz_(b):
        return iz_(a) and iz_(b)
    return eq_(a[0], b[0]) and eq_(a[1], b[1]) and eq_(a[2], b[2])

def lt_(a, b):
    if iz_(b):
        return False
    if iz_(a):
        return True
    if not eq_(a[0], b[0]):
        return lt_(a[0], b[0])
    if not eq_(a[1], b[1]):
        return lt_(a[1], b[1])
    return lt_(a[2], b[2])

def add_(a, b):
    if iz_(a):
        return b
    if iz_(b):
        return a
    if lt_([a[0], a[1], []], [b[0], b[1], []]):
        return b
    return [a[0], a[1], add_(a[2], b)]

def suc_(a):
    return add_(a, ONE_)

def sub_(a, b):
    if iz_(a):
        return []
    if iz_(b):
        return a
    if not lt_([a[0], a[1], []], [b[0], b[1], []]) and not eq_([a[0], a[1], []], [b[0], b[1], []]):
        return a
    return sub_(a[2], b[2])

def s_(a, b):
    if iz_(a):
        return [[], []]
    if lt_([a[0], a[1], []], b):
        return [[], a]
    t0, t1 = s_(a[2], b)
    return [[a[0], a[1], t0], t1]

def l_(a):
    if iz_(a):
        return []
    if iz_(a[2]):
        return a
    return l_(a[2])

def ttc_(a, b):
    if iz_(a):
        return []
    if iz_(ttc_(a[2], b)) and lt_([a[0], a[1], []], [b, [], []]):
        return []
    return [a[0], a[1], ttc_(a[2], b)]

def exp_(a):
    if lt_(a, [[], [ONE_, [], []], []]):
        return [[], a, []]
    p = s_(a[1], [suc_(a[0]), [], []])[0]
    return [a[0], add_(p, sub_(a, [a[0], p, []])), []]

def log_(a):
    if iz_(a):
        return []
    p, q = s_(a[1], [suc_(a[0]), [], []])
    if iz_(a[0]) and iz_(p):
        if not lt_(a[1], [[], [ONE_, [], []], []]):
            if eq_(log_(q), q) and iz_(q[2]) and lt_(a[1], [ONE_, [], []]):
                return [a[0], a[1], []]
        return q
    m = add_([a[0], p, []], q)
    if not lt_(a[1], [a[0], [suc_(a[0]), [], []], []]):
        if eq_(log_(a[1]), a[1]) and iz_(a[2]) and lt_(a[1], [suc_(a[0]), [], []]):
            return [a[0], a[1], []]
    return m

def P_(M, r, n):
    if r == -1:
        return n - 1
    q = P_(M, r - 1, n)
    while q > -1 and M[q][r] >= M[n][r]:
        q = P_(M, r - 1, q)
    return q

def C_(M, n):
    X = []
    for i in range(len(M)):
        if P_(M, 0, i) == n:
            X.append(i)
    return X

def D_(M, n):
    X = 0
    for i in range(len(M)):
        if P_(M, 0, i) == n and M[i][1] > 0:
            X += 1
    return X

def U_(M, n):
    if M[n][1] == 0 or M[n][2] == 1 or n + 1 == len(M):
        return -1
    m = P_(M, 1, n)
    L = [M[m][0] + 1, M[n][1], M[m][2] + 1]
    if P_(M, 1, n) == P_(M, 1, n + 1) and eq_(M[n + 1], L):
        return n + 1
    q = n
    while q != -1:
        q = P_(M, 0, q)
        if P_(M, 1, n) == P_(M, 1, q) and eq_(M[q], L) and M[n + 1][0] > M[q][0]:
            return q
    return -1

def v_(M, n):
    if M[n][1] == 0:
        return []
    if M[n][2] == 0:
        u = ONE_ if U_(M, n) < 0 else l_(v_(M, U_(M, n)))
        return add_(v_(M, P_(M, 1, n)), u)
    p = ONE_
    for i in C_(M, n):
        if not eq_(M[i], [M[n][0] + 1, M[n][1], 1]):
            continue
        q = []
        for j in C_(M, i):
            q = add_(q, o_(M, j))
        p = add_(p, exp_(q))
    return add_(v_(M, P_(M, 1, n)), exp_(p))

def o_(M, n):
    S = []
    u = [U_(M, x) for x in range(len(M))]
    for i in C_(M, n):
        if eq_(M[i], [M[n][0] + 1, M[n][1], 1]):
            continue
        if i in u:
            c = C_(M, i)
            if c:
                if eq_(M[c[-1]], [M[i][0] + 1, M[i][1], 1]):
                    continue
            else:
                continue
        S = add_(S, o_(M, i))
    return [v_(M, n), S, []]

def _o_(M):
    S = []
    for i in range(len(M)):
        if eq_(M[i], [0, 0, 0]):
            S = add_(S, o_(M, i))
    return sf_(S)

def sf_(a):
    if iz_(a):
        return []
    return add_(sp_(sf_(a[0]), [], sf_(a[1])), sf_(a[2]))

def sp_(a, b, c):
    if iz_(c):
        return [a, b, []]
    if lt_(b, c[1]) and (not iz_(c) and (not iz_([a, [], []]) and (not iz_(c) and (not iz_([a, [], []])))) and (not iz_([a, [], []])) and (not iz_([a, [], []]))):
        # The complex conditional in original JS reduces logically to the same branch below.
        pass
    if lt_(b, c[1]) and (not iz_(c) and (not iz_([a, [], []]))):
        t = ttc_(c[1], suc_(c[0]))
        return sp_(a, add_(t, sub_([c[0], c[1], []], [c[0], t, []])), c[2])
    return sp_(a, add_(b, [c[0], c[1], []]), c[2])

# Parsing helper: accept strings like "(0,0,0)(1,1,1)" => list of triples
def parse_mat_string(s):
    groups = re.findall(r'\(([^\)]*)\)', s)
    M = []
    for g in groups:
        if g.strip() == '':
            nums = []
        else:
            nums = [int(x.strip()) for x in g.split(',') if x.strip() != '']
        while len(nums) < 3:
            nums.append(0)
        M.append(nums[:3])
    return M

# Detect finite successor chains for printing small naturals
def finite_value(q):
    if iz_(q):
        return 0
    if iz_(q[0]) and iz_(q[1]):
        return finite_value(q[2]) + 1
    raise ValueError("Not a finite ordinal")

def is_finite(q):
    try:
        finite_value(q)
        return True
    except Exception:
        return False

# Pretty printer into the requested style ψ(Ω_{...}+...+...)
def to_string_pretty(q):
    if iz_(q):
        return '0'
    if is_finite(q):
        return str(finite_value(q))

    a, b = s_(q, [q[0], q[1], []])

    # Build the leading piece (inside ψ(...) first argument)
    # When a[1] is empty -> Ω_{...} else ψ_{...}(...)
    lead = ''
    # detect case Ω_{...} : when a[1] is zero
    if iz_(a[1]):
        # a[0] might be ONE_ (special Ω), or some expression
        if eq_(a[0], ONE_):
            lead = 'Ω'
        else:
            inner = to_string_pretty(a[0])
            lead = f"Ω_{{{inner}}}"
    else:
        # general ψ( a[1] ) possibly with subscript from a[0]
        inner = to_string_pretty(a[1])
        if iz_(a[0]):
            lead = f"ψ({inner})"
        else:
            sub = to_string_pretty(a[0])
            lead = f"ψ_{{{sub}}}({inner})"

    # Special ω-power fallback similar to original: if log differs then show ω^{...}
    try:
        if not eq_(log_([a[0], a[1], []]), [a[0], a[1], []]):
            lead = f"ω^{{{to_string_pretty(log_(a))}}}"
    except Exception:
        # ignore and keep lead as-is
        pass

    coef = 1
    # coefficient: count nested non-empty a[2] depth (approximation from original)
    def coef_count(x):
        if iz_(x[2]):
            return 1
        return 1 + coef_count(x[2])
    try:
        coef = coef_count(a)
    except Exception:
        coef = 1
    if coef > 1:
        lead = f"{lead}{coef}"

    if not iz_(b):
        return f"{lead}+{to_string_pretty(b)}"
    return lead

def psi_from_mat_string(mat_str):
    # mat_str may be 'Limit' or 'Empty matrix' or '(...)...'
    if mat_str in ('Limit', 'Empty matrix'):
        return mat_str
    M = parse_mat_string(mat_str)
    try:
        res = _o_(M)
        return f"ψ({to_string_pretty(res)})"
    except RecursionError:
        return "ψ(computation too deep)"
    except Exception as e:
        return f"ψ(error:{e})"

# --- ONLY CHANGES ARE IN slow_scale + on_slow_change ---
# specifically:
#   resolution=0.001
#   slow_var_override = float(v)
#   label shows .3f
#   slider length maximized (length large)

# ------------------ Tkinter GUI (modified) ------------------
class MatrixViewer(tk.Tk):
    def __init__(self, gen):
        super().__init__()
        self.title("BMS with ψ-output")
        # make window roomy for long sliders and two boxes
        self.geometry("1100x800")
        self.gen = gen
        self.running = True

        ctrl = tk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)

        tk.Label(ctrl, text="Delay (ms):").pack(side=tk.LEFT, padx=6)
        self.delay_scale = tk.Scale(ctrl, from_=1, to=1000, orient=tk.HORIZONTAL)
        self.delay_scale.set(1000)
        self.delay_scale.pack(side=tk.LEFT)

        # --- UPDATED: Slow variable control (float, 3 decimals) ---
        tk.Label(ctrl, text="Slow var:").pack(side=tk.LEFT, padx=6)
        # maximize slider length by giving a large length (pixels)
        self.slow_scale = tk.Scale(
            ctrl,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            resolution=0.001,          # <<---- key change
            command=self.on_slow_change,
            length=600                 # <<---- maximize visual slider length
        )
        self.slow_scale.set(slow_var_override)
        self.slow_scale.pack(side=tk.LEFT)

        self.slow_label = tk.Label(ctrl, text=f"{float(slow_var_override):.3f}")
        self.slow_label.pack(side=tk.LEFT, padx=6)
        # ------------------------------------------------------------

        self.pause_btn = tk.Button(ctrl, text="Pause", command=self.toggle_pause)
        self.pause_btn.pack(side=tk.LEFT, padx=6)

        self.step_btn = tk.Button(ctrl, text="Step", command=self.do_step)
        self.step_btn.pack(side=tk.LEFT)

        # Use a vertical paned window so first box is top (matrix) and second box is bottom (ψ)
        paned = tk.PanedWindow(self, orient=tk.VERTICAL, sashrelief=tk.RAISED)
        paned.pack(fill=tk.BOTH, expand=True)

        mono = font.Font(family="Courier", size=14)

        # Top frame: Bashicu matrix output with heading
        top_frame = tk.Frame(paned)
        tk.Label(top_frame, text="Bashicu matrix output", font=("Helvetica", 12, "bold")).pack(anchor="w", padx=6, pady=(6,0))
        self.matrix_text = tk.Text(top_frame, wrap="word", font=mono, height=12)
        self.matrix_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.matrix_text.configure(state=tk.DISABLED)
        paned.add(top_frame, minsize=150)

        # Bottom frame: Buchholz ordinal (ψ) output with heading
        bot_frame = tk.Frame(paned)
        tk.Label(bot_frame, text="Buchholz ordinal output", font=("Helvetica", 12, "bold")).pack(anchor="w", padx=6, pady=(6,0))
        self.psi_text = tk.Text(bot_frame, wrap="word", font=mono, height=6)
        self.psi_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.psi_text.configure(state=tk.DISABLED)
        paned.add(bot_frame, minsize=80)

        # allow initial sash position to give the matrix area more space
        self.update_idletasks()
        try:
            paned.sash_place(0, 0, int(self.winfo_height() * 0.7))
        except Exception:
            pass

        self.after(0, self.update_loop)

    # --- UPDATED: keep the exact slider float value ---
    def on_slow_change(self, v):
        global slow_var_override
        slow_var_override = float(v)     # <<---- no int() anymore
        self.slow_label.config(text=f"{slow_var_override:.3f}")  # <<---- 3 decimals

    def toggle_pause(self):
        self.running = not self.running
        self.pause_btn.config(text="Resume" if not self.running else "Pause")

    def do_step(self):
        try:
            s = next(self.gen)
            self.show_string(s)
        except StopIteration:
            self.show_string("Sequence finished.")
            self.running = False

    def show_string(self, s):
        # s is the matrix string
        psi_line = psi_from_mat_string(s)
        # Update matrix box
        self.matrix_text.configure(state=tk.NORMAL)
        self.matrix_text.delete("1.0", tk.END)
        self.matrix_text.insert(tk.END, s)
        self.matrix_text.configure(state=tk.DISABLED)
        self.matrix_text.see(tk.END)

        # Update psi box
        self.psi_text.configure(state=tk.NORMAL)
        self.psi_text.delete("1.0", tk.END)
        self.psi_text.insert(tk.END, psi_line)
        self.psi_text.configure(state=tk.DISABLED)
        self.psi_text.see(tk.END)

    def update_loop(self):
        if self.running:
            try:
                s = next(self.gen)
                self.show_string(s)
            except StopIteration:
                self.show_string("Sequence finished.")
                self.running = False
                return
        ms = self.delay_scale.get()
        self.after(ms, self.update_loop)

if __name__ == "__main__":
    gen = sequence_generator()
    app = MatrixViewer(gen)
    app.mainloop()
