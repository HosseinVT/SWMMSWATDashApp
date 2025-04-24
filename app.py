#!/usr/bin/env python
# coding: utf-8

import os
import base64
import subprocess
import pandas as pd
from io import StringIO

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objs as go
from flask import Flask

# ---- Flask server ----
server = Flask(__name__)

# ---- Helper functions ----

def extract_subcatchments(file_bytes):
    lines = file_bytes.decode("utf-8").splitlines()
    data = []; in_sec = False
    for L in lines:
        s = L.strip()
        if s.startswith("[SUBCATCHMENTS]"):
            in_sec = True; continue
        if in_sec and s.startswith("[") and s.endswith("]"):
            break
        if in_sec and s and not s.startswith(";"):
            parts = s.split()
            if len(parts) >= 6:
                data.append((parts[0], parts[3], parts[4], parts[5]))
    return data

allowed_lids = {27:["BR","IT","RB","GR","PP","VS"],
                18:["BR","IT","RB","GR","PP","VS"],
                11:["BR","IT","RB","GR","PP","VS"]}
all_lid_types = ["BR","IT","RB","GR","PP","VS"]

def update_LID(new_vals, inp_path):
    # (same as before, always writing to Update.inp in cwd)
    with open(inp_path,'r') as f: lines = f.readlines()
    sub_data = extract_subcatchments(open(inp_path,'rb').read())
    sums = {}
    for (s,_),v in new_vals.items():
        sums[s] = sums.get(s,0)+v

    out = []
    in_usage=False
    for line in lines:
        s = line.strip()
        if s=="[LID_USAGE]":
            in_usage = True
            out.append(line); continue
        if in_usage and s.startswith("[") and s.endswith("]"):
            in_usage = False
        if in_usage and s and not s.startswith(";"):
            P = s.split()
            if len(P)>=5:
                key = (P[0],P[1])
                if key in new_vals:
                    area = next((float(a) for n,a,_,_ in sub_data if n==P[0]),None)
                    if area is not None:
                        P[3] = str(new_vals[key]/100*area*43560)
                    line = " ".join(P)+"\n"
        out.append(line)

    # adjust impervious...
    for i,line in enumerate(out):
        s=line.strip()
        if s=="[SUBCATCHMENTS]":
            for j in range(i+1,len(out)):
                s2 = out[j].strip()
                if s2.startswith("[") and s2.endswith("]"): break
                if s2 and not s2.startswith(";"):
                    P = s2.split()
                    if len(P)>=5:
                        old_imp = next((float(im) for n,_,im,_ in sub_data if n==P[0]),None)
                        a = sums.get(P[0],0)/100
                        if old_imp is not None and a<1:
                            new_imp = (old_imp - a*100)/(1-a)
                            if new_imp<1: new_imp=1
                            P[4]=f"{new_imp:.2f}"
                            out[j] = " ".join(P)+"\n"
    with open("Update.inp",'w') as f:
        f.writelines(out)
    return "Update.inp"

def run_swmm(exe_path, dll_path, inp_path, rpt_name):
    """
    Launch runswmm.exe against inp_path producing rpt_name.
    Returns CompletedProcess plus working dir used.
    """
    exe_dir = os.path.dirname(exe_path) or os.getcwd()
    # make sure the dll folder is in PATH
    if dll_path:
        dll_dir = os.path.dirname(dll_path)
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH","")
    cmd = [exe_path, inp_path, rpt_name]
    cp = subprocess.run(cmd, cwd=exe_dir, capture_output=True, text=True)
    return cp, exe_dir

# ---- Layouts ----

upload_layout = dbc.Container([
    html.H1("Upload SWMM Files"),
    html.H3("1. SWMM Input (.inp)"),
    dcc.Upload(id="upload-inp", children=html.Div(["Drag & drop or ", html.A(".inp")]),
               style={"width":"50%","height":"60px","lineHeight":"60px","border":"2px dashed #add8e6","borderRadius":"5px","textAlign":"center","margin":"20px auto"},
               multiple=False, accept=".inp"),
    html.Div(id="upload-status"),
    html.Hr(),
    html.H3("2. runswmm.exe"),
    dcc.Upload(id="upload-runswmm", children=html.Div(["Drag & drop or ", html.A("runswmm.exe")]),
               style={"width":"50%","height":"60px","lineHeight":"60px","border":"2px dashed #d3ffd3","borderRadius":"5px","textAlign":"center","margin":"20px auto"},
               multiple=False, accept=".exe"),
    html.Div(id="runswmm-upload-status"),
    html.Hr(),
    html.H3("3. swmm5.dll"),
    dcc.Upload(id="upload-swmm5", children=html.Div(["Drag & drop or ", html.A("swmm5.dll")]),
               style={"width":"50%","height":"60px","lineHeight":"60px","border":"2px dashed #ffd3d3","borderRadius":"5px","textAlign":"center","margin":"20px auto"},
               multiple=False, accept=".dll"),
    html.Div(id="swmm5-upload-status"),
], fluid=True)

# (Define the other page containers just as before; e.g. subcatchments_layout, simulation_layout, etc.)

subcatchments_layout = dbc.Container([...], fluid=True)
simulation_layout     = dbc.Container([...], fluid=True)
lid_layout            = dbc.Container([...], fluid=True)
calculate_lid_area_layout   = dbc.Container([...], fluid=True)
total_lid_area_layout       = dbc.Container([...], fluid=True)
calculate_lid_cost_layout   = dbc.Container([...], fluid=True)
updated_simulation_layout   = dbc.Container([...], fluid=True)
pond_cost_layout            = dbc.Container([...], fluid=True)
swat_pond_layout            = dbc.Container([...], fluid=True)
total_cost_layout           = dbc.Container([...], fluid=True)

app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dcc.Store(id="stored-file-path"),
    dcc.Store(id="stored-runswmm-file"),
    dcc.Store(id="stored-swmm5-file"),
    dcc.Store(id="stored-lid-plan"),
    dcc.Store(id="stored-original-total-flow"),
    dcc.Store(id="stored-original-peak-flow"),
    dcc.Store(id="stored-updated-total-flow"),
    dcc.Store(id="stored-lid-cost"),
    dcc.Store(id="stored-pond-cost"),
    dbc.Tabs([
        dbc.Tab(label="Upload Files", tab_id="upload"),
        dbc.Tab(label="Subcatchment Data", tab_id="subcatchments"),
        dbc.Tab(label="SWMM Simulation", tab_id="simulation"),
        dbc.Tab(label="LIDs Plan", tab_id="lid_definition"),
        dbc.Tab(label="LID Area", tab_id="calculate_lid_area"),
        dbc.Tab(label="LID Type", tab_id="total_lid_area"),
        dbc.Tab(label="LID Cost", tab_id="calculate_lid_cost"),
        dbc.Tab(label="LID Simulation", tab_id="updated_simulation"),
        dbc.Tab(label="Pond Cost", tab_id="pond_cost"),
        dbc.Tab(label="Total Cost", tab_id="total_cost"),
        dbc.Tab(label="SWAT+ Simulation", tab_id="swat_pond"),
    ], id="tabs", active_tab="upload", persistence=True, persistence_type="session"),
    html.Div(id="page-content")
], fluid=True)

@app.callback(Output("page-content","children"), Input("tabs","active_tab"))
def render(tab):
    return {
        "upload": upload_layout,
        "subcatchments": subcatchments_layout,
        "simulation": simulation_layout,
        "lid_definition": lid_layout,
        "calculate_lid_area": calculate_lid_area_layout,
        "total_lid_area": total_lid_area_layout,
        "calculate_lid_cost": calculate_lid_cost_layout,
        "updated_simulation": updated_simulation_layout,
        "pond_cost": pond_cost_layout,
        "total_cost": total_cost_layout,
        "swat_pond": swat_pond_layout
    }.get(tab, "No tab selected.")

# ---- Callbacks ----

# 5A. Save .inp
@app.callback(
    [Output("upload-status","children"),
     Output("stored-file-path","data")],
    Input("upload-inp","contents"), State("upload-inp","filename")
)
def save_inp(c,fn):
    if c:
        _,b = c.split(",")
        data = base64.b64decode(b)
        os.makedirs("uploads", exist_ok=True)
        p = os.path.join("uploads", fn)
        with open(p,"wb") as f: f.write(data)
        return f".inp → {p}", p
    return "No .inp uploaded.", ""

# 5B. Save runswmm.exe
@app.callback(
    [Output("runswmm-upload-status","children"),
     Output("stored-runswmm-file","data")],
    Input("upload-runswmm","contents"), State("upload-runswmm","filename")
)
def save_exe(c,fn):
    if c and fn.lower().endswith(".exe"):
        _,b=c.split(","); data=base64.b64decode(b)
        os.makedirs("uploads", exist_ok=True)
        p=os.path.join("uploads",fn)
        with open(p,"wb") as f: f.write(data)
        return f"runswmm.exe → {p}", p
    return "Please upload a .exe", ""

# 5C. Save swmm5.dll
@app.callback(
    [Output("swmm5-upload-status","children"),
     Output("stored-swmm5-file","data")],
    Input("upload-swmm5","contents"), State("upload-swmm5","filename")
)
def save_dll(c,fn):
    if c and fn.lower().endswith(".dll"):
        _,b=c.split(","); data=base64.b64decode(b)
        os.makedirs("uploads", exist_ok=True)
        p=os.path.join("uploads",fn)
        with open(p,"wb") as f: f.write(data)
        return f"swmm5.dll → {p}", p
    return "Please upload a .dll", ""

# 5D. Subcatchment extraction (unchanged)
# ...

# 5E. Original SWMM Simulation
@app.callback(
    [Output("file-info-sim","children"),
     Output("sim-results","children"),
     Output("stored-original-total-flow","data"),
     Output("stored-original-peak-flow","data")],
    Input("run-sim-btn","n_clicks"),
    State("stored-file-path","data"),
    State("stored-runswmm-file","data"),
    State("stored-swmm5-file","data"),
)
def run_simulation(n, inp_path, exe_path, dll_path):
    if n and inp_path and exe_path:
        cp, workdir = run_swmm(exe_path, dll_path, inp_path, "swmm_report.rpt")
        print("RC", cp.returncode, "OUT", cp.stdout, "ERR", cp.stderr)
        if cp.returncode:
            return f"Using {inp_path}", html.Pre(f"SWMM failed:\n{cp.stderr}"), None, None

        # parse report absolute
        rpt = os.path.join(workdir, "swmm_report.rpt")
        total=peak=None; found=False
        with open(rpt) as f:
            for L in f:
                if "Outfall Node" in L:
                    found=True; continue
                if found:
                    if L.strip().startswith("OF1"):
                        p=L.split()
                        peak=float(p[3]); total=float(p[4])*0.134*1e6
                        break
                    if not L.strip():
                        found=False

        if peak is None:
            return "Parsing report","OF1 not found",None,None

        res = html.Div([
            html.H3("Results"),
            html.P(f"Total (ft³): {total:,.2f}"),
            html.P(f"Peak (cfs): {peak:,.2f}")
        ])
        return f"Using {inp_path}", res, total, peak

    return "", "", None, None

# 5F–5K. The rest of your callbacks (LID plan, updated sim, cost, SWAT+)—
# be sure in your updated simulation callback to call run_swmm(...) 
# with the right arguments just like above, parse Update.inp → updated_swmm_report.rpt,
# and never use os.chdir except via subprocess cwd.

###############################################################################
# 6. RUN
###############################################################################
if __name__ == "__main__":
    app.run_server(debug=True, port=8054)
