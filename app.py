#!/usr/bin/env python
# coding: utf-8

import os
import base64
import subprocess

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, dash_table

import pandas as pd
import plotly.express as px
from io import StringIO

# -----------------------------------------------------------------------------
# 1. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def extract_subcatchments(file_content: bytes):
    """Parse SWMM INP file to extract subcatchment data."""
    lines = file_content.decode("utf-8").splitlines()
    in_section = False
    data = []
    for line in lines:
        l = line.strip()
        if l.startswith("[SUBCATCHMENTS]"):
            in_section = True
            continue
        if in_section and l.startswith("[") and l.endswith("]"):
            break
        if in_section and l and not l.startswith(";"):
            parts = l.split()
            if len(parts) >= 6:
                data.append((parts[0], parts[3], parts[4], parts[5]))
    return data

def parse_swmm_report(rpt_path: str):
    """Grab total & peak flow from the SWMM report."""
    total = peak = None
    section = False
    with open(rpt_path, "r") as f:
        for line in f:
            if "Outfall Node" in line:
                section = True
                continue
            if section:
                if line.strip().startswith("OF1"):
                    p = line.split()
                    if len(p) >= 5:
                        peak  = float(p[3])
                        total = float(p[4]) * 0.134 * 1_000_000
                    break
                if not line.strip():
                    section = False
    return total, peak

# -----------------------------------------------------------------------------
# 2. DASH SETUP
# -----------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
server = app.server

_upload_style = {
    "width": "50%", "height": "60px", "lineHeight": "60px",
    "borderWidth": "2px", "borderStyle": "dashed",
    "borderRadius": "5px", "textAlign": "center",
    "margin": "20px auto"
}

# -----------------------------------------------------------------------------
# 3. PAGE LAYOUTS
# -----------------------------------------------------------------------------
upload_layout = dbc.Container([
    html.H1("Upload Files"),
    html.H5("1) SWMM INP"),
    dcc.Upload(
        id="upload-inp",
        children=html.Div(["Drag & Drop or ", html.A("Select .inp")]),
        style={**_upload_style, "backgroundColor":"#add8e6"},
        multiple=False, accept=".inp"
    ),
    html.Div(id="inp-upload-status"),
    html.Hr(),

    html.H5("2) SWMM EXE"),
    dcc.Upload(
        id="upload-exe",
        children=html.Div(["Drag & Drop or ", html.A("Select .exe")]),
        style={**_upload_style, "backgroundColor":"#d3ffd3"},
        multiple=False, accept=".exe"
    ),
    html.Div(id="exe-upload-status"),
    html.Hr(),

    html.H5("3) SWMM DLL"),
    dcc.Upload(
        id="upload-dll",
        children=html.Div(["Drag & Drop or ", html.A("Select .dll")]),
        style={**_upload_style, "backgroundColor":"#ffd3d3"},
        multiple=False, accept=".dll"
    ),
    html.Div(id="dll-upload-status"),
], fluid=True)

subcatchments_layout = dbc.Container([
    html.H1("Subcatchment Data"),
    html.Div(id="file-info-subcatch"),
    dbc.Button("Extract Subcatchments", id="extract-btn", color="primary", className="mb-3"),
    html.Div(id="subcatchment-data"),
], fluid=True)

simulation_layout = dbc.Container([
    html.H1("SWMM Simulation"),
    html.Div(id="file-info-sim"),
    dbc.Button("Run SWMM", id="run-sim-btn", color="primary", className="mb-3"),
    html.Div(id="sim-results"),
], fluid=True)

# -----------------------------------------------------------------------------
# 4. MAIN LAYOUT WITH TABS
# -----------------------------------------------------------------------------
app.layout = dbc.Container([
    # stores for file paths and results
    dcc.Store(id="stored-inp"),
    dcc.Store(id="stored-exe"),
    dcc.Store(id="stored-dll"),
    dcc.Store(id="stored-original-total-flow"),
    dcc.Store(id="stored-original-peak-flow"),

    dbc.Tabs([
        dbc.Tab(label="Upload", tab_id="upload"),
        dbc.Tab(label="Subcatchments", tab_id="subcatchments"),
        dbc.Tab(label="Simulation", tab_id="simulation"),
    ], id="tabs", active_tab="upload",
       persistence=True, persistence_type="session",
       className="mb-4"),
    html.Div(id="page-content")
], fluid=True)

# -----------------------------------------------------------------------------
# 5. RENDER TABS
# -----------------------------------------------------------------------------
@app.callback(
    Output("page-content", "children"),
    Input("tabs", "active_tab")
)
def render_tab(tab):
    if tab == "upload":
        return upload_layout
    elif tab == "subcatchments":
        return subcatchments_layout
    elif tab == "simulation":
        return simulation_layout
    return html.Div("Unknown tab")

# -----------------------------------------------------------------------------
# 6A. SAVE INP
# -----------------------------------------------------------------------------
@app.callback(
    [Output("inp-upload-status","children"),
     Output("stored-inp","data")],
    Input("upload-inp","contents"),
    State("upload-inp","filename"),
    prevent_initial_call=True
)
def save_inp(contents, fname):
    if not contents:
        return dash.no_update, dash.no_update
    _, b64 = contents.split(",", 1)
    data = base64.b64decode(b64)
    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", fname)
    with open(path, "wb") as f:
        f.write(data)
    files = os.listdir("uploads")
    return f"Saved {fname}. Now have: {', '.join(files)}", path

# -----------------------------------------------------------------------------
# 6B. SAVE EXE
# -----------------------------------------------------------------------------
@app.callback(
    [Output("exe-upload-status","children"),
     Output("stored-exe","data")],
    Input("upload-exe","contents"),
    State("upload-exe","filename"),
    prevent_initial_call=True
)
def save_exe(contents, fname):
    if not contents:
        return dash.no_update, dash.no_update
    _, b64 = contents.split(",", 1)
    data = base64.b64decode(b64)
    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", fname)
    with open(path, "wb") as f:
        f.write(data)
    files = os.listdir("uploads")
    return f"Saved {fname}. Now have: {', '.join(files)}", path

# -----------------------------------------------------------------------------
# 6C. SAVE DLL
# -----------------------------------------------------------------------------
@app.callback(
    [Output("dll-upload-status","children"),
     Output("stored-dll","data")],
    Input("upload-dll","contents"),
    State("upload-dll","filename"),
    prevent_initial_call=True
)
def save_dll(contents, fname):
    if not contents:
        return dash.no_update, dash.no_update
    _, b64 = contents.split(",", 1)
    data = base64.b64decode(b64)
    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", fname)
    with open(path, "wb") as f:
        f.write(data)
    files = os.listdir("uploads")
    return f"Saved {fname}. Now have: {', '.join(files)}", path

# -----------------------------------------------------------------------------
# 6D. EXTRACT SUBCATCHMENTS
# -----------------------------------------------------------------------------
@app.callback(
    [Output("file-info-subcatch","children"),
     Output("subcatchment-data","children")],
    Input("extract-btn","n_clicks"),
    State("stored-inp","data"),
    prevent_initial_call=True
)
def extract_cb(n, inp_path):
    if not inp_path or not os.path.exists(inp_path):
        return "No INP file.", ""
    info = f"Using {os.path.basename(inp_path)}"
    with open(inp_path,"rb") as f:
        data = extract_subcatchments(f.read())

    df = pd.DataFrame(data, columns=["Sub","Area(ac)","%Imp","Width"])
    df["Area(ac)"] = pd.to_numeric(df["Area(ac)"], errors="coerce")
    df["%Imp"]    = pd.to_numeric(df["%Imp"],    errors="coerce")

    table = dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name":c,"id":c} for c in df.columns],
        page_size=10,
        style_table={"overflowX":"auto"}
    )
    fig1 = px.pie(df, names="Sub", values="Area(ac)", title="Areas")
    fig2 = px.treemap(df, path=["Sub"], values="%Imp", title="% Impervious")
    graphs = dbc.Row([
        dbc.Col(dcc.Graph(figure=fig1), width=6),
        dbc.Col(dcc.Graph(figure=fig2), width=6),
    ])
    return info, html.Div([table, html.Hr(), graphs])

# -----------------------------------------------------------------------------
# 6E. RUN SWMM
# -----------------------------------------------------------------------------
@app.callback(
    [Output("file-info-sim","children"),
     Output("sim-results","children"),
     Output("stored-original-total-flow","data"),
     Output("stored-original-peak-flow","data")],
    Input("run-sim-btn","n_clicks"),
    State("stored-inp","data"),
    State("stored-exe","data"),
    State("stored-dll","data"),
    prevent_initial_call=True
)
def run_swmm(n, inp_path, exe_path, dll_path):
    if not (inp_path and exe_path and dll_path):
        return "Upload INP, EXE & DLL first", "", None, None

    rpt = "swmm_report.rpt"
    cmd = [exe_path, inp_path, rpt]

    # put uploads folder (with exe+dll) on PATH so SWMM finds its DLL
    env = os.environ.copy()
    uploads = os.path.abspath("uploads")
    env["PATH"] = uploads + os.pathsep + env.get("PATH", "")

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}", "", None, None

    total, peak = parse_swmm_report(rpt)
    if total is None or peak is None:
        return "Report parse failed", "", None, None

    info = f"Ran {os.path.basename(exe_path)} → {rpt}"
    results = html.Div([
        html.P(f"Total Volume (ft³): {total:,.2f}"),
        html.P(f"Peak Flow   (cfs): {peak:,.2f}")
    ])
    return info, results, total, peak

# -----------------------------------------------------------------------------
# 7. RUN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run_server(debug=True, port=8059)
