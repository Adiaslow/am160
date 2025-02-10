"""
Implementation of SINDy analysis for dynamical systems with comprehensive derivative computation
and coefficient analysis capabilities.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pysindy import SINDy
from pysindy.optimizers import STLSQ

def lorenz(_: float, u: np.ndarray, sigma: float, rho: float,
           beta: float) -> np.ndarray:
    """Returns a list containing the three functions of the Lorenz equation.

    The Lorenz equations have constant coefficients (that don't depend on t),
    but we still receive t as the first parameter because that's how the
    integrator works.
    """
    x = u[0]
    y = u[1]
    z = u[2]
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z

    return np.hstack((dx_dt, dy_dt, dz_dt))

def add_noise_to_derivatives(uprime: np.ndarray, variance: float) -> np.ndarray:
    """Adds Gaussian noise to derivatives."""
    noise = np.random.normal(0, np.sqrt(variance), uprime.shape)
    return uprime + noise

def compute_derivatives_finite_diff(u: np.ndarray, dt: float, method: str = 'central') -> np.ndarray:
    """
    Compute derivatives using finite difference methods.
    
    Args:
        u: Array of shape (n_samples, n_variables)
        dt: Time step
        method: 'forward', 'backward', or 'central'
    
    Returns:
        Array of derivatives
    """
    if method == 'forward':
        du_dt = (u[1:] - u[:-1]) / dt
        du_dt = np.vstack([du_dt, du_dt[-1]])
    elif method == 'backward':
        du_dt = (u[1:] - u[:-1]) / dt
        du_dt = np.vstack([du_dt[0], du_dt])
    elif method == 'central':
        du_dt = np.zeros_like(u)
        du_dt[1:-1] = (u[2:] - u[:-2]) / (2 * dt)
        du_dt[0] = (u[1] - u[0]) / dt
        du_dt[-1] = (u[-1] - u[-2]) / dt
    else:
        raise ValueError("Method must be 'forward', 'backward', or 'central'")
    
    return du_dt

def compute_derivatives_spline(u: np.ndarray, t: np.ndarray, window_size: int = 4) -> np.ndarray:
    """
    Compute derivatives using cubic spline interpolation.
    
    Args:
        u: Array of shape (n_samples, n_variables)
        t: Time points
        window_size: Number of consecutive points to use for spline fitting
    
    Returns:
        Array of derivatives
    """
    n_samples, n_vars = u.shape
    derivatives = np.zeros_like(u)
    
    for var in range(n_vars):
        for i in range(0, n_samples - window_size + 1):
            t_window = t[i:i + window_size]
            u_window = u[i:i + window_size, var]
            cs = CubicSpline(t_window, u_window)
            if i + window_size//2 < n_samples:
                derivatives[i + window_size//2, var] = cs.derivative()(t[i + window_size//2])
    
    derivatives[:window_size//2] = derivatives[window_size//2]
    derivatives[-(window_size//2):] = derivatives[-(window_size//2 + 1)]
    
    return derivatives

def create_lorenz_grid(u, noise_vars=[0.01, 0.1, 0.5]):
    """Creates 2x2 grid of Lorenz attractors with different noise levels."""
    colormap = 'plasma'
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}],
               [{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=('Original', f'Noise σ²={noise_vars[0]}',
                       f'Noise σ²={noise_vars[1]}', f'Noise σ²={noise_vars[2]}'),
        horizontal_spacing=0.0,
        vertical_spacing=0.0,
    )

    trajectories = [u] + [add_noise_to_derivatives(u, var) for var in noise_vars]

    for idx, trajectory in enumerate(trajectories):
        row = (idx // 2) + 1
        col = (idx % 2) + 1

        fig.add_trace(
            go.Scatter3d(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                z=trajectory[:, 2],
                mode='lines',
                line=dict(
                    color=np.arange(len(trajectory)),
                    colorscale=colormap,
                    width=2
                )
            ),
            row=row, col=col
        )
        
        points_to_plot = 100
        skip = len(trajectory) // points_to_plot
        
        fig.add_trace(
            go.Scatter3d(
                x=trajectory[::skip, 0],
                y=trajectory[::skip, 1],
                z=trajectory[::skip, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=np.arange(points_to_plot),
                    colorscale=colormap,
                    showscale=True if idx == 0 else False
                )
            ),
            row=row, col=col
        )

    camera = dict(eye=dict(x=2, y=-2, z=2))
    fig.update_layout(
        title='Lorenz Attractors with Different Noise Levels',
        showlegend=False,
        width=1200,
        height=1200
    )

    for i in range(1, 5):
        fig.update_scenes(
            camera=camera,
            aspectmode='cube',
            row=(i-1)//2 + 1,
            col=(i-1)%2 + 1
        )

    return fig

def compare_derivative_methods(u: np.ndarray, t: np.ndarray) -> dict:
    """
    Compare different derivative computation methods for SINDy.
    
    Args:
        u: Array of shape (n_samples, n_variables)
        t: Time points
    
    Returns:
        Dictionary containing SINDy models for each method
    """
    dt = t[1] - t[0]
    methods = {
        'forward_diff': compute_derivatives_finite_diff(u, dt, 'forward'),
        'backward_diff': compute_derivatives_finite_diff(u, dt, 'backward'),
        'central_diff': compute_derivatives_finite_diff(u, dt, 'central'),
        'spline': compute_derivatives_spline(u, t)
    }
    
    models = {}
    for method_name, du_dt in methods.items():
        optimizer = STLSQ(threshold=0.1, alpha=0.05)
        model = SINDy(optimizer=optimizer)
        model.fit(u, t=dt, x_dot=du_dt)
        models[method_name] = model
    
    return models

def analyze_coefficients(models: dict) -> dict:
    """
    Analyze the accuracy and characteristics of discovered coefficients for each method.
    
    Returns a dictionary with analysis metrics for each method.
    """
    true_coefs = {
        'dx/dt': np.array([0, -10, 10, 0, 0, 0, 0, 0, 0, 0]),
        'dy/dt': np.array([0, 28, -1, 0, 0, 0, -1, 0, 0, 0]),
        'dz/dt': np.array([0, 0, 0, -8/3, 0, 1, 0, 0, 0, 0])
    }
    
    analysis = {}
    terms = ['1', 'x', 'y', 'z', 'x²', 'xy', 'xz', 'y²', 'yz', 'z²']
    
    for method_name, model in models.items():
        pred_coefs = model.coefficients()
        method_analysis = {
            'relative_errors': [],
            'spurious_terms': [],
            'missed_terms': [],
            'accuracy_score': 0.0
        }
        
        for i, (var, true_coef) in enumerate(true_coefs.items()):
            pred_coef = pred_coefs[i]
            
            true_nonzero = np.where(np.abs(true_coef) > 1e-10)[0]
            if len(true_nonzero) > 0:
                rel_errors = np.abs(pred_coef[true_nonzero] - true_coef[true_nonzero]) / np.abs(true_coef[true_nonzero])
                method_analysis['relative_errors'].extend(list(zip(
                    [var] * len(true_nonzero),
                    [terms[j] for j in true_nonzero],
                    rel_errors
                )))
            
            true_zero = np.where(np.abs(true_coef) <= 1e-10)[0]
            spurious = np.where(np.abs(pred_coef[true_zero]) > 0.1)[0]
            method_analysis['spurious_terms'].extend([
                (var, terms[true_zero[j]], pred_coef[true_zero[j]])
                for j in spurious
            ])
            
            pred_zero = np.where(np.abs(pred_coef) <= 0.1)[0]
            missed = np.intersect1d(pred_zero, true_nonzero)
            method_analysis['missed_terms'].extend([
                (var, terms[j], true_coef[j])
                for j in missed
            ])
        
        true_matrix = np.vstack([true_coefs[var] for var in true_coefs.keys()])
        error = np.linalg.norm(pred_coefs - true_matrix) / np.linalg.norm(true_matrix)
        method_analysis['accuracy_score'] = 100 * (1 - error)
        
        analysis[method_name] = method_analysis
    
    return analysis

def visualize_derivative_comparison(u: np.ndarray, t: np.ndarray, true_derivatives: np.ndarray):
    """
    Visualize how different derivative computation methods compare to true derivatives.
    Returns a Plotly figure showing the comparison.
    """
    dt = t[1] - t[0]
    methods = {
        'Forward Difference': compute_derivatives_finite_diff(u, dt, 'forward'),
        'Backward Difference': compute_derivatives_finite_diff(u, dt, 'backward'),
        'Central Difference': compute_derivatives_finite_diff(u, dt, 'central'),
        'Cubic Spline': compute_derivatives_spline(u, t)
    }
    
    fig = make_subplots(rows=3, cols=1,
                       subplot_titles=['dx/dt', 'dy/dt', 'dz/dt'])
    
    colors = ['purple', 'blue', 'green', 'orange']
    t_slice = slice(0, 100)
    
    for var in range(3):
        for (method_name, derivatives), color in zip(methods.items(), colors):
            fig.add_trace(
                go.Scatter(x=t[t_slice], y=derivatives[t_slice, var],
                          name=method_name if var == 0 else None,
                          line=dict(color=color),
                          showlegend=var == 0),
                row=var+1, col=1
            )
        
        fig.add_trace(
            go.Scatter(x=t[t_slice], y=true_derivatives[t_slice, var],
                      name='True' if var == 0 else None,
                      line=dict(color='red', dash='dash'),
                      showlegend=var == 0),
            row=var+1, col=1
        )
    
    fig.update_layout(
        height=900,
        width=1200,
        title='Comparison of Derivative Computation Methods',
        showlegend=True
    )
    
    return fig

def visualize_coefficients(results, noise_levels):
    """Creates three side-by-side plots of discovered coefficients."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=['dx/dt', 'dy/dt', 'dz/dt'])
    
    true_coefs = {
        'dx/dt': np.array([0, -10, 10, 0, 0, 0, 0, 0, 0, 0]),
        'dy/dt': np.array([0, 28, -1, 0, 0, 0, -1, 0, 0, 0]),
        'dz/dt': np.array([0, 0, 0, -8/3, 0, 1, 0, 0, 0, 0])
    }
    
    variables = ['dx/dt', 'dy/dt', 'dz/dt']
    terms = ['1', 'x', 'y', 'z', 'x²', 'xy', 'xz', 'y²', 'yz', 'z²']
    colors = ['blue', 'green', 'purple']
    
    for var_idx, var in enumerate(variables):
        for noise_idx, (noise, model) in enumerate(zip(noise_levels, results)):
            fig.add_trace(go.Scatter(
                name=f'σ²={noise}',
                x=terms,
                y=model.coefficients()[var_idx],
                mode='lines+markers',
                line=dict(color=colors[noise_idx]),
                showlegend=var_idx == 0
            ), row=1, col=var_idx+1)
        
        fig.add_trace(go.Scatter(
            name='True Values',
            x=terms,
            y=true_coefs[var],
            mode='lines+markers',
            line=dict(color='red', width=2, dash='dash'),
            showlegend=var_idx == 0
        ), row=1, col=var_idx+1)

    fig.update_layout(
        title='SINDy Coefficient Values with Different Noise Levels',
        width=1200,
        height=500
    )
    
    return fig

def calculate_accuracy(model, true_coefs):
    """Calculate accuracy of SINDy model compared to true coefficients."""
    pred_coefs = np.array(model.coefficients())
    true_coefs = np.array([true_coefs['dx/dt'], true_coefs['dy/dt'], true_coefs['dz/dt']])
    
    error = np.linalg.norm(pred_coefs - true_coefs) / np.linalg.norm(true_coefs)
    accuracy = max(0, 100 * (1 - error))
    return accuracy

def plot_noise_accuracy(t, u, noise_range=np.linspace(0, 10, 100)):
    """Creates accuracy plot for different noise levels."""
    dt = t[1] - t[0]
    accuracies = []
    
    true_coefs = {
        'dx/dt': np.array([0, -10, 10, 0, 0, 0, 0, 0, 0, 0]),
        'dy/dt': np.array([0, 28, -1, 0, 0, 0, -1, 0, 0, 0]),
        'dz/dt': np.array([0, 0, 0, -8/3, 0, 1, 0, 0, 0, 0])
    }
    
    for noise in noise_range:
        u_noisy = add_noise_to_derivatives(u, noise)
        optimizer = STLSQ(threshold=0.1, alpha=0.05)
        model = SINDy(optimizer=optimizer)
        model.fit(u_noisy, t=dt)
        accuracies.append(calculate_accuracy(model, true_coefs))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=noise_range,
        y=accuracies,
        mode='lines+markers',
        name='Model Accuracy'
    ))
    
    fig.update_layout(
        title='SINDy Model Accuracy vs Noise Level',
        xaxis_title='Noise Variance (σ²)',
        yaxis_title='Accuracy (%)',
        width=1200,
        height=500
    )
    
    return fig

def visualize_coefficient_analysis(analysis: dict):
    """
    Create comprehensive visualization of coefficient analysis results.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Overall Accuracy Score',
            'Relative Errors in Identified Terms',
            'Number of Spurious Terms',
            'Number of Missed Terms'
        )
    )
    
    methods = list(analysis.keys())
    accuracies = [analysis[method]['accuracy_score'] for method in methods]
    
    fig.add_trace(
        go.Bar(x=methods, y=accuracies, name='Accuracy Score',
               marker_color=['purple', 'blue', 'green', 'orange']),
        row=1, col=1
    )
    
    colors = ['purple', 'blue', 'green', 'orange']
    for method, color in zip(methods, colors):
        errors = analysis[method]['relative_errors']
        if errors:
            variables, terms, values = zip(*errors)
            fig.add_trace(
                go.Box(y=values, name=method, boxpoints='all',
                      marker_color=color,
                      line_color=color),
                row=1, col=2
            )
    
    spurious_counts = [len(analysis[method]['spurious_terms']) for method in methods]
    fig.add_trace(
        go.Bar(x=methods, y=spurious_counts, name='Spurious Terms',
               marker_color=['purple', 'blue', 'green', 'orange']),
        row=2, col=1
    )
    
    missed_counts = [len(analysis[method]['missed_terms']) for method in methods]
    fig.add_trace(
        go.Bar(x=methods, y=missed_counts, name='Missed Terms',
               marker_color=['purple', 'blue', 'green', 'orange']),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        width=1200,
        title='Comprehensive Analysis of SINDy Coefficients by Derivative Method',
        showlegend=False
    )
    
    return fig

def print_detailed_analysis(analysis: dict):
    """
    Print detailed analysis results in a formatted way.
    """
    print("DETAILED ANALYSIS OF SINDY COEFFICIENTS BY DERIVATIVE METHOD")
    print("=" * 80)
    
    for method, results in analysis.items():
        print(f"\n{method.upper()}")
        print("-" * 40)
        
        print(f"Overall Accuracy Score: {results['accuracy_score']:.2f}%")
        
        if results['relative_errors']:
            print("\nRelative Errors in Identified Terms:")
            for var, term, error in results['relative_errors']:
                print(f"  {var}, {term}: {error:.3f}")
        
        if results['spurious_terms']:
            print("\nSpurious Terms (False Positives):")
            for var, term, coef in results['spurious_terms']:
                print(f"  {var}, {term}: {coef:.3f}")
        
        if results['missed_terms']:
            print("\nMissed Terms (False Negatives):")
            for var, term, true_coef in results['missed_terms']:
                print(f"  {var}, {term}: should be {true_coef:.3f}")
        
        print("\n" + "=" * 80)

if __name__ == "__main__":
    SIGMA = 10
    RHO = 28
    BETA = 8/3
    
    t0, tmax = 0, 100
    n_samples = 10000
    t = np.linspace(t0, tmax, n_samples)
    dt = t[1] - t[0]
    
    u0 = np.array([-8, 8, 27])
    
    result = solve_ivp(fun=lorenz,
                      t_span=(t0, tmax),
                      y0=u0,
                      t_eval=t,
                      args=(SIGMA, RHO, BETA))
    u = result.y.T
    
    true_derivatives = np.array([lorenz(0, u[i], SIGMA, RHO, BETA) 
                               for i in range(len(u))])
    
    lorenz_fig = create_lorenz_grid(u)
    
    noise_levels = [0.01, 0.1, 0.5]
    noise_results = []
    for variance in noise_levels:
        u_noisy = add_noise_to_derivatives(u, variance)
        optimizer = STLSQ(threshold=0.1, alpha=0.05)
        model = SINDy(optimizer=optimizer)
        model.fit(u_noisy, t=dt)
        noise_results.append(model)
    
    coef_fig = visualize_coefficients(noise_results, noise_levels)
    
    noise_acc_fig = plot_noise_accuracy(t, u)
    
    models = compare_derivative_methods(u, t)
    
    coefficient_analysis = analyze_coefficients(models)
    
    deriv_comp_fig = visualize_derivative_comparison(u, t, true_derivatives)
    
    coef_analysis_fig = visualize_coefficient_analysis(coefficient_analysis)
    
    lorenz_fig.show()
    coef_fig.show()
    noise_acc_fig.show()
    deriv_comp_fig.show()
    coef_analysis_fig.show()
    
    print_detailed_analysis(coefficient_analysis)
