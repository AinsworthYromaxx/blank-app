"""
v4_quantum_3d_plots.py
======================
Publication-quality 3D Plotly visualizations for the V4 Quantum Portfolio.

Three figures expose the quantum wave-function mechanics to any audience:

  fig1  Wave Interference Surface
        Z = |ψ|²(φ_momentum, φ_safety)  — animated phase slider
        Red = constructive interference, Blue = destructive cancellation
        All portfolio holdings float as glowing spheres on the surface.

  fig2  Phase Space Trajectories
        (φ_momentum, φ_sortino, φ_safety) paths for all portfolio holdings
        Smooth convergence from disordered start → "quantum stability zone"
        Colour encodes final |ψ|² conviction.

  fig3  Energy Landscape  (Hamiltonian)
        X = annual volatility, Y = expected return (momentum proxy)
        Z = −|ψ|²  ("potential energy" — portfolio sits at ground-state minimum)
        Full universe: sector-coloured point cloud.  Portfolio: gold stars.

Quick start
-----------
>>> from v4_quantum_3d_plots import generate_3d_visuals
>>> fig1, fig2, fig3 = generate_3d_visuals(metrics_df, portfolio_df)
>>> fig1.show(); fig2.show(); fig3.show()
"""
from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("plotly not installed — run: pip install plotly kaleido", stacklevel=2)


def _ensure_plotly():
    """Re-attempt plotly import if module was loaded before plotly was installed."""
    global go, make_subplots, HAS_PLOTLY  # noqa: PLW0603
    if HAS_PLOTLY:
        return True
    try:
        import plotly.graph_objects as _go
        from plotly.subplots import make_subplots as _ms
        go = _go
        make_subplots = _ms
        HAS_PLOTLY = True
        return True
    except ImportError:
        return False

try:
    from scipy.interpolate import griddata as _griddata
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ─────────────────────────────────────────────────────────────────────────────
# Visual constants
# ─────────────────────────────────────────────────────────────────────────────

SECTOR_COLORS: dict[str, str] = {
    'Technology':             '#4a9eff',
    'Healthcare':             '#72d572',
    'Financial Services':     '#ffb347',
    'Consumer Cyclical':      '#ff6b9d',
    'Consumer Discretionary': '#ff6b9d',
    'Consumer Defensive':     '#c3a6ff',
    'Consumer Staples':       '#c3a6ff',
    'Industrials':            '#40e0d0',
    'Real Estate':            '#f9ca24',
    'Communication Services': '#f0e68c',
    'Utilities':              '#7ec8e3',
    'Energy':                 '#ff9ff3',
    'Basic Materials':        '#a0c4ff',
    'Materials':              '#a0c4ff',
}
_DEFAULT_COLOR = '#aaaaaa'

# Dark presentation theme shared by all three figures
_DARK = dict(
    template='plotly_dark',
    paper_bgcolor='#0e1117',
    plot_bgcolor='#0e1117',
    font=dict(family='Inter, Arial, sans-serif', color='#e0e0e0', size=12),
    margin=dict(l=10, r=10, t=70, b=10),
)

# Diverging colour scale: Prussian blue → white → crimson
_INTERFERENCE_COLORSCALE = [
    [0.00, '#1a1a5e'],
    [0.15, '#2244aa'],
    [0.30, '#4488dd'],
    [0.45, '#aaccff'],
    [0.50, '#ffffff'],
    [0.60, '#ffcc66'],
    [0.75, '#ee4444'],
    [0.90, '#aa1111'],
    [1.00, '#660000'],
]

_DEFAULT_BW = {'momentum': 0.30, 'safety': 0.30, 'sortino': 0.25, 'value': 0.15}


# ─────────────────────────────────────────────────────────────────────────────
# Quantum math helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rank_normalize(series: pd.Series) -> pd.Series:
    """Percentile rank → [0, 1] (same as notebook _normalize())."""
    return series.rank(pct=True, na_option='bottom').clip(0.0, 1.0)


def _extract_phi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive normalised φ ∈ [0,1] for each stock from raw metric columns.

    Returns DataFrame with columns [momentum, safety, sortino, value]
    aligned to df's index.
    """
    out = pd.DataFrame(index=df.index)

    out['momentum'] = _rank_normalize(df['momentum']) if 'momentum' in df.columns else pd.Series(0.5, index=df.index)
    out['sortino']  = _rank_normalize(df['sortino'])  if 'sortino'  in df.columns else pd.Series(0.5, index=df.index)

    if 'win_rate' in df.columns and 'max_dd' in df.columns:
        wr = _rank_normalize(df['win_rate'])
        dd = _rank_normalize(-df['max_dd'])
        out['safety'] = (wr + dd) / 2.0
    elif 'win_rate' in df.columns:
        out['safety'] = _rank_normalize(df['win_rate'])
    else:
        out['safety'] = pd.Series(0.5, index=df.index)

    if 'vol_60' in df.columns:
        inv_vol = 1.0 / df['vol_60'].replace(0, np.nan).fillna(df['vol_60'].median())
        out['value'] = _rank_normalize(inv_vol)
    else:
        out['value'] = pd.Series(0.5, index=df.index)

    return out.fillna(0.5)


def _psi_sq_2d(
    phi_m: np.ndarray, phi_s: np.ndarray,
    phi_sortino: float, phi_value: float,
    bw: dict, a: float, b: float,
) -> np.ndarray:
    """
    Compute |ψ|² over a 2-D grid of (φ_momentum, φ_safety).

    Parameters
    ----------
    phi_m, phi_s : 2-D arrays from np.meshgrid — same shape
    phi_sortino, phi_value : scalars, held fixed (surface slice)
    bw : basis weight dict
    a, b : phase parameters from CAL-A

    Returns
    -------
    2-D ndarray of |ψ|² values.
    """
    psi_r = np.zeros_like(phi_m, dtype=float)
    psi_i = np.zeros_like(phi_m, dtype=float)
    factors = [
        (phi_m,       bw.get('momentum', 0.30)),
        (phi_s,       bw.get('safety',   0.30)),
        (phi_sortino, bw.get('sortino',  0.25)),
        (phi_value,   bw.get('value',    0.15)),
    ]
    for phi, w in factors:
        theta  = a * np.pi * (1.0 - phi) ** b
        psi_r += np.sqrt(w) * phi * np.cos(theta)
        psi_i += np.sqrt(w) * phi * np.sin(theta)
    return psi_r ** 2 + psi_i ** 2


def _psi_sq_stocks(phi_df: pd.DataFrame, bw: dict, a: float, b: float) -> np.ndarray:
    """Vectorised |ψ|² for every stock row in phi_df."""
    psi_r = np.zeros(len(phi_df))
    psi_i = np.zeros(len(phi_df))
    for col, w in [
        ('momentum', bw.get('momentum', 0.30)),
        ('safety',   bw.get('safety',   0.30)),
        ('sortino',  bw.get('sortino',  0.25)),
        ('value',    bw.get('value',    0.15)),
    ]:
        phi   = phi_df[col].values if col in phi_df.columns else np.full(len(phi_df), 0.5)
        theta = a * np.pi * (1.0 - phi) ** b
        psi_r += np.sqrt(w) * phi * np.cos(theta)
        psi_i += np.sqrt(w) * phi * np.sin(theta)
    return psi_r ** 2 + psi_i ** 2


def _scene(title_x: str = 'φ_momentum', title_y: str = 'φ_safety',
           title_z: str = '|ψ|²', eye: dict | None = None) -> dict:
    """Return a standardised Plotly scene dict."""
    eye = eye or dict(x=1.55, y=-1.55, z=1.35)
    return dict(
        aspectmode='cube',
        camera=dict(eye=eye, up=dict(x=0, y=0, z=1)),
        xaxis=dict(title=title_x, showgrid=True, gridcolor='#333355', gridwidth=1,
                   backgroundcolor='#0e1117', color='#aaccff'),
        yaxis=dict(title=title_y, showgrid=True, gridcolor='#333355', gridwidth=1,
                   backgroundcolor='#0e1117', color='#aaccff'),
        zaxis=dict(title=title_z, showgrid=True, gridcolor='#333355', gridwidth=1,
                   backgroundcolor='#0e1117', color='#ffcc99'),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Wave Interference Surface
# ─────────────────────────────────────────────────────────────────────────────

def fig1_wave_interference_surface(
    metrics_df:    pd.DataFrame,
    portfolio_df:  pd.DataFrame,
    best_a:        float = 1.0,
    best_b:        float = 1.0,
    basis_weights: dict  = None,
    n_grid:        int   = 55,
    n_frames:      int   = 8,
) -> 'go.Figure':
    """
    3D wave interference surface with animated phase rotation slider.

    The surface shows |ψ|²(φ_momentum, φ_safety) with φ_sortino = φ_value = 0.5
    (median cross-section).  All portfolio holdings are pinned as glowing spheres.
    Animate the 'a' parameter slider to watch the interference ridges shift live.

    Returns a Plotly Figure with animation frames.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly required: pip install plotly")

    bw = basis_weights or _DEFAULT_BW.copy()

    # ── Normalise universe phi values ─────────────────────────────────────────
    phi_universe = _extract_phi(metrics_df)
    ticker_col   = 'ticker' if 'ticker' in metrics_df.columns else None
    if ticker_col:
        phi_universe.index = metrics_df[ticker_col].values

    # ── Surface grid ──────────────────────────────────────────────────────────
    phi_axis = np.linspace(0.0, 1.0, n_grid)
    M, S     = np.meshgrid(phi_axis, phi_axis)
    phi_sort_med = float(phi_universe['sortino'].median())
    phi_val_med  = float(phi_universe['value'].median())

    # ── All portfolio holdings ─────────────────────────────────────────────────
    weight_col  = 'weight'  if 'weight'  in portfolio_df.columns else None
    sector_col  = 'sector'  if 'sector'  in portfolio_df.columns else None
    psi_sq_col  = 'psi_sq'  if 'psi_sq'  in portfolio_df.columns else None
    t_col       = 'ticker'  if 'ticker'  in portfolio_df.columns else None

    holdings = portfolio_df.copy()

    h_tickers = holdings[t_col].tolist() if t_col else list(range(len(holdings)))
    h_sectors = (holdings[sector_col].tolist() if sector_col
                 else ['Unknown'] * len(holdings))

    # Map holdings to their phi coordinates
    h_phi_m = [float(phi_universe.loc[tk, 'momentum'])
               if tk in phi_universe.index else 0.5
               for tk in h_tickers]
    h_phi_s = [float(phi_universe.loc[tk, 'safety'])
               if tk in phi_universe.index else 0.5
               for tk in h_tickers]
    h_weights = (holdings[weight_col].tolist() if weight_col else [1.0] * len(holdings))

    def _holding_z(a_val):
        """Compute |ψ|² for each holding at given 'a'."""
        return [float(_psi_sq_2d(np.array([[m]]), np.array([[s]]),
                                  phi_sort_med, phi_val_med, bw, a_val, best_b)[0, 0])
                for m, s in zip(h_phi_m, h_phi_s)]

    def _holding_hover(a_val):
        h_z = _holding_z(a_val)
        return [
            f"<b>{tk}</b><br>Sector: {sec}<br>φ_momentum: {m:.2f}<br>"
            f"φ_safety: {s:.2f}<br>|ψ|²: {z:.4f}<br>Weight: {w:.1%}"
            for tk, sec, m, s, z, w in
            zip(h_tickers, h_sectors, h_phi_m, h_phi_s, h_z, h_weights)
        ]

    h_colors = [SECTOR_COLORS.get(sec, _DEFAULT_COLOR) for sec in h_sectors]

    # ── Animation frames  ─────────────────────────────────────────────────────
    a_vals   = np.linspace(0.5, 2.0, n_frames)
    frames   = []
    for a_val in a_vals:
        z_surf = _psi_sq_2d(M, S, phi_sort_med, phi_val_med, bw, a_val, best_b)
        h_z    = _holding_z(a_val)
        frames.append(go.Frame(
            name=f'{a_val:.2f}',
            data=[
                go.Surface(
                    z=z_surf, x=phi_axis, y=phi_axis,
                    colorscale=_INTERFERENCE_COLORSCALE,
                    cmin=0, cmax=float(z_surf.max()),
                    lighting=dict(ambient=0.6, diffuse=0.85, roughness=0.4,
                                  specular=0.9, fresnel=0.3),
                    lightposition=dict(x=100, y=200, z=1000),
                    opacity=0.88, showscale=False,
                ),
                go.Scatter3d(
                    x=h_phi_m, y=h_phi_s, z=h_z,
                    text=_holding_hover(a_val),
                    hovertemplate='%{text}<extra></extra>',
                    mode='markers',
                    marker=dict(
                        size=[5 + 10 * w / max(h_weights) for w in h_weights],
                        color=h_colors,
                        opacity=0.95,
                        line=dict(color='white', width=2),
                        symbol='circle',
                    ),
                ),
            ],
            traces=[0, 1],
        ))

    # ── Initial figure (best_a) ───────────────────────────────────────────────
    z_init = _psi_sq_2d(M, S, phi_sort_med, phi_val_med, bw, best_a, best_b)
    h_z0   = _holding_z(best_a)

    surf_trace = go.Surface(
        z=z_init, x=phi_axis, y=phi_axis,
        colorscale=_INTERFERENCE_COLORSCALE,
        cmin=0, cmax=float(z_init.max()),
        colorbar=dict(title=dict(text='|ψ|²', side='right'),
                      thickness=14, len=0.6, x=1.02,
                      tickfont=dict(size=10, color='#aaaaaa')),
        lighting=dict(ambient=0.6, diffuse=0.85, roughness=0.4,
                      specular=0.9, fresnel=0.3),
        lightposition=dict(x=100, y=200, z=1000),
        opacity=0.88,
        hovertemplate=(
            'φ_momentum=%{x:.2f}<br>φ_safety=%{y:.2f}<br>|ψ|²=%{z:.4f}'
            '<extra></extra>'
        ),
    )

    # Show text labels only for top-10 by weight
    _top10_tickers = set(
        (portfolio_df.nlargest(10, weight_col)[t_col].tolist() if weight_col and t_col
         else h_tickers[:10])
    )
    _text_labels = [tk if tk in _top10_tickers else '' for tk in h_tickers]

    marker_trace = go.Scatter3d(
        x=h_phi_m, y=h_phi_s, z=h_z0,
        name='Portfolio Holdings',
        text=_text_labels,
        customdata=_holding_hover(best_a),
        hovertemplate='%{customdata}<extra></extra>',
        mode='markers+text',
        textposition='top center',
        textfont=dict(size=9, color='white'),
        marker=dict(
            size=[5 + 10 * w / max(h_weights) for w in h_weights],
            color=h_colors, opacity=0.95,
            line=dict(color='white', width=1.5),
        ),
    )

    # ── Slider & play button ──────────────────────────────────────────────────
    sliders = [dict(
        active=int(np.argmin(np.abs(a_vals - best_a))),
        currentvalue=dict(prefix='Phase a = ', font=dict(size=13, color='#aaccff')),
        pad=dict(t=40, b=10),
        steps=[dict(
            method='animate',
            label=f'{v:.1f}',
            args=[[f'{v:.2f}'],
                  dict(mode='immediate', frame=dict(duration=400, redraw=True),
                       transition=dict(duration=200))]
        ) for v in a_vals],
    )]

    updatemenus = [dict(
        type='buttons', showactive=False,
        y=1.12, x=0.18, xanchor='right',
        buttons=[
            dict(label='▶ Play',
                 method='animate',
                 args=[None, dict(frame=dict(duration=600, redraw=True),
                                  fromcurrent=True, transition=dict(duration=300))]),
            dict(label='⏸ Pause',
                 method='animate',
                 args=[[None], dict(mode='immediate', frame=dict(duration=0),
                                    transition=dict(duration=0))]),
        ],
        font=dict(size=12), bgcolor='#222244', bordercolor='#4466aa',
    )]

    fig = go.Figure(data=[surf_trace, marker_trace], frames=frames)
    fig.update_layout(
        **_DARK,
        title=dict(
            text=(
                '<b>Quantum Harmony: Factor Interference Creates Alpha</b><br>'
                f'<sup>θ(φ) = a·π·(1−φ)^{best_b:.2f} | '
                'Red = constructive peak · Blue = destructive cancellation | '
                'Spheres = portfolio holdings</sup>'
            ),
            x=0.5, xanchor='center', font=dict(size=16, color='#fff'),
        ),
        scene=_scene('φ_momentum  →', 'φ_safety  →', '|ψ|²  probability density',
                     eye=dict(x=1.55, y=-1.55, z=1.35)),
        sliders=sliders,
        updatemenus=updatemenus,
        height=680,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Phase Space Trajectories
# ─────────────────────────────────────────────────────────────────────────────

def fig2_phase_trajectories(
    metrics_df:    pd.DataFrame,
    portfolio_df:  pd.DataFrame,
    best_a:        float = 1.0,
    best_b:        float = 1.0,
    basis_weights: dict  = None,
    n_steps:       int   = 90,
) -> 'go.Figure':
    """
    3D phase-space trajectories for all portfolio holdings.

    Each trajectory starts at a "disordered" state (low φ) and converges
    to the stock's current (φ_mom, φ_sortino, φ_safety) position, tracing
    the path of "quantum collapse" to an ordered state.

    Colour encodes final |ψ|² conviction (bright = high conviction).
    A translucent "quantum stability zone" sphere marks the (1,1,1) corner.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly required: pip install plotly")

    bw = basis_weights or _DEFAULT_BW.copy()

    # ── Extract φ for holdings ────────────────────────────────────────────────
    phi_universe = _extract_phi(metrics_df)
    ticker_col   = 'ticker' if 'ticker' in metrics_df.columns else None
    if ticker_col:
        phi_universe.index = metrics_df[ticker_col].values

    t_col      = 'ticker'  if 'ticker'  in portfolio_df.columns else None
    weight_col = 'weight'  if 'weight'  in portfolio_df.columns else None
    sector_col = 'sector'  if 'sector'  in portfolio_df.columns else None
    psi_col    = 'psi_sq'  if 'psi_sq'  in portfolio_df.columns else None

    holdings = portfolio_df.copy()
    if weight_col:
        holdings = holdings.sort_values(weight_col, ascending=False).reset_index(drop=True)
    h_tickers = holdings[t_col].tolist() if t_col else [f'H{i}' for i in range(len(holdings))]
    h_sectors = holdings[sector_col].tolist() if sector_col else ['Unknown'] * len(holdings)
    h_weights = holdings[weight_col].values if weight_col else np.ones(len(holdings))

    psi_sq_final = (_psi_sq_stocks(
        phi_universe.loc[[t for t in h_tickers if t in phi_universe.index]],
        bw, best_a, best_b)
        if any(t in phi_universe.index for t in h_tickers)
        else np.full(len(h_tickers), 0.5))

    # Viridis colorscale mapped to |ψ|² range
    psi_min, psi_max = float(psi_sq_final.min()), float(psi_sq_final.max()) + 1e-9

    def _psi_color(psi: float) -> str:
        """Map |ψ|² → hex colour on a Plasma-like scale."""
        t = (psi - psi_min) / (psi_max - psi_min)
        # Plasma: dark purple → blue → teal → yellow → bright white
        stops = [(0, '#0d0221'), (0.25, '#5b2d8e'), (0.5, '#de7065'),
                 (0.75, '#fdc328'), (1.0, '#f0f921')]
        t = np.clip(t, 0, 1)
        for i in range(len(stops) - 1):
            t0, c0 = stops[i];   t1, c1 = stops[i + 1]
            if t0 <= t <= t1:
                frac = (t - t0) / (t1 - t0)
                r0, g0, b0 = int(c0[1:3],16), int(c0[3:5],16), int(c0[5:7],16)
                r1, g1, b1 = int(c1[1:3],16), int(c1[3:5],16), int(c1[5:7],16)
                r = int(r0 + frac*(r1-r0))
                g = int(g0 + frac*(g1-g0))
                b3= int(b0 + frac*(b1-b0))
                return f'#{r:02x}{g:02x}{b3:02x}'
        return stops[-1][1]

    traces = []
    t_param = np.linspace(0, 1, n_steps)

    for idx, tk in enumerate(h_tickers):
        # Final (target) phi values
        if tk in phi_universe.index:
            fm = float(phi_universe.loc[tk, 'momentum'])
            fs = float(phi_universe.loc[tk, 'sortino'])
            fsa= float(phi_universe.loc[tk, 'safety'])
        else:
            fm = fs = fsa = 0.5

        psi_val = psi_sq_final[idx] if idx < len(psi_sq_final) else 0.5
        sec     = h_sectors[idx]
        color   = _psi_color(psi_val)

        # ── Smooth convergence path (damped oscillatory approach) ─────────────
        # Start: random "disordered" initial state
        np.random.seed(idx * 17 + 7)
        x0 = np.random.uniform(0.05, 0.25)
        y0 = np.random.uniform(0.05, 0.25)
        z0 = np.random.uniform(0.05, 0.25)

        # Bézier control point: midpoint with upward bulge
        cx = (x0 + fm) / 2 + np.random.uniform(-0.1, 0.1)
        cy = (y0 + fs) / 2 + np.random.uniform(-0.1, 0.1)
        cz = (z0 + fsa)/ 2 + 0.18

        # Quadratic Bézier + damped oscillation
        damp = np.exp(-3.5 * t_param)
        osc  = np.sin(t_param * 4 * np.pi) * 0.06

        xs = ((1-t_param)**2 * x0 + 2*(1-t_param)*t_param * cx + t_param**2 * fm
              + osc * damp)
        ys = ((1-t_param)**2 * y0 + 2*(1-t_param)*t_param * cy + t_param**2 * fs
              + osc * damp)
        zs = ((1-t_param)**2 * z0 + 2*(1-t_param)*t_param * cz + t_param**2 * fsa
              + osc * damp)

        xs = np.clip(xs, 0.0, 1.0)
        ys = np.clip(ys, 0.0, 1.0)
        zs = np.clip(zs, 0.0, 1.0)

        # ── Path line ─────────────────────────────────────────────────────────
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines',
            name=tk,
            line=dict(color=color, width=4),
            opacity=0.75,
            showlegend=False,
            hoverinfo='skip',
        ))

        # ── Terminal marker ───────────────────────────────────────────────────
        w_scaled = float(h_weights[idx]) if weight_col else 0.02
        # Only show text label for top-10 by weight
        _show_label = idx < 10 if weight_col else True
        traces.append(go.Scatter3d(
            x=[fm], y=[fs], z=[fsa],
            mode='markers+text' if _show_label else 'markers',
            name=tk,
            text=[tk] if _show_label else [''],
            textposition='top center',
            textfont=dict(size=9, color='white'),
            hovertemplate=(
                f'<b>{tk}</b><br>Sector: {sec}<br>'
                f'φ_momentum: {fm:.3f}<br>φ_sortino: {fs:.3f}<br>'
                f'φ_safety: {fsa:.3f}<br>|ψ|²: {psi_val:.4f}<br>'
                f'Weight: {w_scaled:.1%}<extra></extra>'
            ),
            marker=dict(
                size=6 + 10 * w_scaled / (float(h_weights.max()) if weight_col else 0.05),
                color=color,
                opacity=0.95,
                line=dict(color='white', width=1.5),
            ),
        ))

    # ── Quantum stability zone marker (translucent corner sphere) ─────────────
    traces.append(go.Scatter3d(
        x=[0.95], y=[0.95], z=[0.95],
        mode='markers+text',
        name='Quantum Ground State',
        text=['⭐ Ground State'],
        textposition='bottom center',
        textfont=dict(size=11, color='#ffd700'),
        marker=dict(size=28, color='rgba(255,215,0,0.18)',
                    line=dict(color='gold', width=3)),
        hovertemplate='<b>Quantum Stability Zone</b><br>All φ → 1 ideal state<extra></extra>',
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        **_DARK,
        title=dict(
            text=(
                '<b>Factor Convergence: Path to Ground State</b><br>'
                '<sup>All holdings trace paths from disorder → quantum stability zone '
                '| Brightness = |ψ|² conviction</sup>'
            ),
            x=0.5, xanchor='center', font=dict(size=16, color='#fff'),
        ),
        scene=_scene(
            'φ_momentum  →', 'φ_sortino  →', 'φ_safety  →',
            eye=dict(x=1.55, y=1.55, z=1.05),
        ),
        height=660,
        legend=dict(x=0.02, y=0.95, bgcolor='rgba(0,0,0,0.4)',
                    bordercolor='#444466', borderwidth=1),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Energy Landscape (Hamiltonian)
# ─────────────────────────────────────────────────────────────────────────────

def fig3_energy_landscape(
    metrics_df:    pd.DataFrame,
    portfolio_df:  pd.DataFrame,
    best_a:        float = 1.0,
    best_b:        float = 1.0,
    basis_weights: dict  = None,
) -> 'go.Figure':
    """
    3D energy landscape treating −|ψ|² as potential energy.

    Every candidate stock is a particle at (annual_vol, expected_return).
    Portfolio holdings sit at the ground-state minimum (lowest energy well).
    Colour = sector.  Portfolio = gold stars rising from the energy surface.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly required: pip install plotly")

    bw = basis_weights or _DEFAULT_BW.copy()

    # ── Universe metrics ──────────────────────────────────────────────────────
    df = metrics_df.copy()
    ticker_col = 'ticker' if 'ticker' in df.columns else None
    if ticker_col:
        df = df.set_index(ticker_col)

    phi_u   = _extract_phi(df)
    psi_all = _psi_sq_stocks(phi_u, bw, best_a, best_b)
    energy  = -psi_all                        # lower energy = stronger selection

    # Annual volatility
    if 'vol_60' in df.columns:
        ann_vol = df['vol_60'].values * np.sqrt(252) * 100
    else:
        ann_vol = np.random.uniform(10, 45, len(df))

    # Expected return proxy: momentum mapped to plausible return range
    if 'momentum' in df.columns:
        ann_ret = (phi_u['momentum'].values * 80.0 - 10.0)   # −10%  to +70%
    else:
        ann_ret = np.random.uniform(-10, 50, len(df))

    sectors = df['sector'].values if 'sector' in df.columns else np.full(len(df), 'Unknown')
    colors  = [SECTOR_COLORS.get(s, _DEFAULT_COLOR) for s in sectors]
    tickers = df.index.tolist()

    # ── Interpolated energy surface (requires scipy) ──────────────────────────
    traces = []
    if HAS_SCIPY:
        vol_grid = np.linspace(float(np.nanpercentile(ann_vol, 2)),
                               float(np.nanpercentile(ann_vol, 98)), 48)
        ret_grid = np.linspace(float(np.nanpercentile(ann_ret, 2)),
                               float(np.nanpercentile(ann_ret, 98)), 48)
        VG, RG = np.meshgrid(vol_grid, ret_grid)
        try:
            EG = _griddata(
                np.column_stack([ann_vol, ann_ret]),
                energy,
                (VG, RG), method='cubic',
            )
            EG = np.where(np.isnan(EG),
                          _griddata(np.column_stack([ann_vol, ann_ret]),
                                    energy, (VG, RG), method='nearest'),
                          EG)
            traces.append(go.Surface(
                x=vol_grid, y=ret_grid, z=EG,
                colorscale=_INTERFERENCE_COLORSCALE,
                opacity=0.52,
                showscale=False,
                hovertemplate=(
                    'Vol: %{x:.1f}%<br>Return: %{y:.1f}%<br>'
                    'Energy −|ψ|²: %{z:.3f}<extra></extra>'
                ),
                lighting=dict(ambient=0.7, diffuse=0.8, roughness=0.5, specular=0.6),
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor='#ffffff',
                           project_z=False, width=1.5),
                ),
            ))
        except Exception:
            pass  # scipy interpolation failed — proceed with scatter only

    # ── Full universe scatter ─────────────────────────────────────────────────
    hover_universe = [
        f'<b>{tk}</b><br>Sector: {sec}<br>'
        f'Vol: {v:.1f}%<br>Ret: {r:.1f}%<br>Energy: {e:.4f}<br>|ψ|²: {-e:.4f}'
        for tk, sec, v, r, e in zip(tickers, sectors, ann_vol, ann_ret, energy)
    ]
    traces.append(go.Scatter3d(
        x=ann_vol, y=ann_ret, z=energy,
        mode='markers',
        name='Universe',
        text=hover_universe,
        hovertemplate='%{text}<extra></extra>',
        marker=dict(
            size=4,
            color=colors,
            opacity=0.55,
            line=dict(color='rgba(255,255,255,0.08)', width=0.5),
        ),
    ))

    # ── Portfolio holdings — gold stars at ground-state bottom ────────────────
    t_col   = 'ticker'  if 'ticker'  in portfolio_df.columns else None
    w_col   = 'weight'  if 'weight'  in portfolio_df.columns else None
    sec_col = 'sector'  if 'sector'  in portfolio_df.columns else None

    if t_col:
        p_tickers = portfolio_df[t_col].values
    else:
        p_tickers = np.array([f'H{i}' for i in range(len(portfolio_df))])

    p_index = {tk: i for i, tk in enumerate(tickers)}

    p_vol  = np.array([ann_vol[p_index[tk]] if tk in p_index else float(ann_vol.mean())
                       for tk in p_tickers])
    p_ret  = np.array([ann_ret[p_index[tk]] if tk in p_index else float(ann_ret.mean())
                       for tk in p_tickers])
    p_e    = np.array([energy[p_index[tk]]  if tk in p_index else float(energy.min())
                       for tk in p_tickers])
    p_wts  = portfolio_df[w_col].values if w_col else np.ones(len(p_tickers)) * 0.02
    p_secs = portfolio_df[sec_col].values if sec_col else ['Portfolio'] * len(p_tickers)

    hover_port = [
        f'<b>⭐ {tk}</b><br>Sector: {sec}<br>'
        f'Vol: {v:.1f}%<br>Ret: {r:.1f}%<br>'
        f'Energy: {e:.4f}<br>|ψ|²: {-e:.4f}<br>Weight: {w:.1%}'
        for tk, sec, v, r, e, w in zip(p_tickers, p_secs, p_vol, p_ret, p_e, p_wts)
    ]

    traces.append(go.Scatter3d(
        x=p_vol, y=p_ret, z=p_e,
        mode='markers',
        name='Portfolio (ground state)',
        text=hover_port,
        hovertemplate='%{text}<extra></extra>',
        marker=dict(
            size=[9 + 18 * w / max(float(p_wts.max()), 1e-6) for w in p_wts],
            symbol='diamond',
            color='gold',
            opacity=1.0,
            line=dict(color='white', width=2),
        ),
    ))

    # ── Sector legend traces (invisible, for legend) ──────────────────────────
    seen_sectors: set[str] = set()
    for sec, col in SECTOR_COLORS.items():
        if sec in sectors and sec not in seen_sectors:
            seen_sectors.add(sec)
            traces.append(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers', name=sec,
                marker=dict(size=8, color=col, opacity=0.9),
                showlegend=True,
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        **_DARK,
        title=dict(
            text=(
                '<b>Quantum Portfolio Energy Minimization</b><br>'
                '<sup>Z = −|ψ|² potential energy | '
                'Portfolio (⭐ gold) sits at the ground-state minimum | '
                'Sector colours on universe point cloud</sup>'
            ),
            x=0.5, xanchor='center', font=dict(size=16, color='#fff'),
        ),
        scene=_scene(
            'Annual Volatility (%)', 'Expected Return proxy (%)',
            '−|ψ|²  (potential energy)',
            eye=dict(x=2.0, y=-2.0, z=1.5),
        ),
        height=680,
        legend=dict(
            x=1.01, y=0.85, bgcolor='rgba(0,0,0,0.35)',
            bordercolor='#444466', borderwidth=1, font=dict(size=10),
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Master entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_3d_visuals(
    metrics_df:    pd.DataFrame,
    portfolio_df:  pd.DataFrame,
    best_a:        float = 1.0,
    best_b:        float = 1.0,
    basis_weights: dict  = None,
    save_dir:      str   = None,
) -> 'tuple[go.Figure, go.Figure, go.Figure]':
    """
    Generate all three publication-quality 3D quantum visualizations.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Full universe metrics with columns:
        ticker, sector, momentum, sortino, win_rate, max_dd, vol_60
        (at least 20 rows recommended; 100+ for energy landscape surface)
    portfolio_df : pd.DataFrame
        Portfolio holdings with columns: ticker, sector, weight, psi_sq
    best_a, best_b : float
        Calibrated phase parameters from CAL-A
    basis_weights : dict, optional
        Regime-adjusted basis weights from CAL-C
        Default: {'momentum':0.30, 'safety':0.30, 'sortino':0.25, 'value':0.15}
    save_dir : str, optional
        If provided, saves HTML + PNG (requires kaleido) for each figure.

    Returns
    -------
    tuple of (fig1, fig2, fig3) — Plotly Figure objects
    """
    if not _ensure_plotly():
        raise ImportError("plotly required: pip install plotly kaleido")

    bw = basis_weights or _DEFAULT_BW.copy()

    # Ensure 'ticker' is a column (handle index-as-ticker)
    for frame in [metrics_df, portfolio_df]:
        if frame is not None and 'ticker' not in frame.columns and frame.index.name == 'ticker':
            frame.reset_index(inplace=True)

    fig1 = fig1_wave_interference_surface(
        metrics_df, portfolio_df, best_a, best_b, bw)
    fig2 = fig2_phase_trajectories(
        metrics_df, portfolio_df, best_a, best_b, bw)
    fig3 = fig3_energy_landscape(
        metrics_df, portfolio_df, best_a, best_b, bw)

    if save_dir:
        save_visuals(fig1, fig2, fig3, save_dir)

    return fig1, fig2, fig3


def save_visuals(
    fig1: 'go.Figure',
    fig2: 'go.Figure',
    fig3: 'go.Figure',
    save_dir: str,
    also_png: bool = True,
) -> None:
    """
    Save the three figures as interactive HTML and optionally PNG.

    PNG export requires kaleido: pip install kaleido
    """
    os.makedirs(save_dir, exist_ok=True)
    figs = {
        'v4_3d_interference_surface':  fig1,
        'v4_3d_phase_trajectories':    fig2,
        'v4_3d_energy_landscape':      fig3,
    }
    for name, fig in figs.items():
        html_path = os.path.join(save_dir, f'{name}.html')
        fig.write_html(html_path, include_plotlyjs='cdn', full_html=True, auto_open=False)
        print(f'  HTML saved: {html_path}')
        if also_png:
            png_path = os.path.join(save_dir, f'{name}.png')
            try:
                fig.write_image(png_path, width=1400, height=800, scale=2)
                print(f'   PNG saved: {png_path}')
            except Exception as exc:
                print(f'   PNG skipped ({name}): {exc} — install kaleido: pip install kaleido')
