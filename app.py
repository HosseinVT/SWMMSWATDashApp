#!/usr/bin/env python
# coding: utf-8

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, dash_table
from flask import Flask
import plotly.express as px
import plotly.graph_objs as go
import base64
import pandas as pd
import subprocess
import os
from io import StringIO

# Flask server instance
server = Flask(__name__)

###############################################################################
# 1. HELPER FUNCTIONS
###############################################################################
def extract_subcatchments(file_content):
    lines = file_content.decode("utf-8").splitlines()
    in_section = False
    data = []
    for line in lines:
        line = line.strip()
        if line.startswith("[SUBCATCHMENTS]"):
            in_section = True
            continue
        if in_section and line.startswith("[") and line.endswith("]"):
            break
        if in_section and line and not line.startswith(";"):
            parts = line.split()
            if len(parts) >= 6:
                data.append((parts[0], parts[3], parts[4], parts[5]))
    return data

allowed_lids = {
    27: ["BR","IT","RB","GR","PP","VS"],
    18: ["BR","IT","RB","GR","PP","VS"],
    11: ["BR","IT","RB","GR","PP","VS"]
}
all_lid_types = ["BR","IT","RB","GR","PP","VS"]

def update_LID(new_values, inp_file_path=None):
    if inp_file_path is None:
        inp_file_path = "LID-Model.inp"
    output_file_path = "Update.inp"
    with open(inp_file_path,'r') as f:
        content = f.readlines()
    with open(inp_file_path,'rb') as f:
        sub_data = extract_subcatchments(f.read())
    sums = {}
    for (subc,_),v in new_values.items():
        sums[subc] = sums.get(subc,0) + v
    in_usage=False
    updated=[]
    for line in content:
        s=line.strip()
        if s.startswith("[LID_USAGE]"):
            in_usage=True; updated.append(line); continue
        if in_usage and s.startswith("[") and s.endswith("]"):
            in_usage=False
        if in_usage and s and not s.startswith(";"):
            parts=s.split()
            if len(parts)>=5:
                key=(parts[0],parts[1])
                if key in new_values:
                    area = next((float(a) for n,a,_,_ in sub_data if n==parts[0]), None)
                    if area is not None:
                        parts[3]=str(new_values[key]/100*area*43560)
                    line=" ".join(parts)+"\n"
        updated.append(line)
    # adjust impervious
    for i,line in enumerate(updated):
        s=line.strip()
        if s.startswith("[SUBCATCHMENTS]"):
            for j in range(i+1,len(updated)):
                s2=updated[j].strip()
                if s2.startswith("[") and s2.endswith("]"):
                    break
                if s2 and not s2.startswith(";"):
                    parts=s2.split()
                    if len(parts)>=5:
                        old_imp = next((float(im) for n,_,im,_ in sub_data if n==parts[0]), None)
                        a=sums.get(parts[0],0)/100
                        if old_imp is not None and a<1:
                            imp_new=(old_imp - a*100)/(1-a)
                            if imp_new<1: imp_new=1
                            parts[4]=f"{imp_new:.2f}"
                            updated[j]=" ".join(parts)+"\n"
    with open(output_file_path,'w') as f:
        f.writelines(updated)
    return output_file_path

def SWAT(swatworking_directory, UpsIn):
    coeff=0.0283168; unit_filter=275
    os.chdir(swatworking_directory)
    exc_file='exco_om.exc'
    try:
        lines=open(exc_file).readlines()
        header=lines[1].split()
        data=pd.read_csv(StringIO("".join(lines[2:])), delim_whitespace=True, header=None, names=header)
        if 'flo' in data:
            data['flo']=UpsIn*coeff
        else:
            raise ValueError("No 'flo' column")
        with open(exc_file,'w') as f:
            f.writelines(lines[:2])
            data.to_csv(f,sep='\t',index=False,header=False)
        subprocess.run(["rev61.0_64rel.exe"], check=True)
        df=pd.read_csv("channel_sd_day.txt", delim_whitespace=True, skiprows=1)
        df.to_csv("channel_sd_day.csv",index=False)
        df2=df[['unit','flo_out']].drop(index=[0])
        df2.to_csv("flo_out.csv",index=False)
        out=df2[df2.unit==unit_filter].flo_out.max()
        return out
    except Exception as e:
        return f"SWAT error: {e}"

###############################################################################
# 2. APP & SERVER
###############################################################################
app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP])

###############################################################################
# 3. PAGE LAYOUTS
###############################################################################
upload_layout = dbc.Container([
    html.H1("Upload SWMM Files"),
    html.H3("1. SWMM Input (.inp)"),
    dcc.Upload(id="upload-inp", children=html.Div(["Drag & drop or ", html.A(".inp file")]),
               style={"width":"50%","height":"60px","lineHeight":"60px",
                      "border":"2px dashed #add8e6","borderRadius":"5px",
                      "textAlign":"center","margin":"20px auto"},
               multiple=False, accept=".inp"),
    html.Div(id="upload-status"),
    html.Hr(),
    html.H3("2. SWMM Executable (runswmm.exe)"),
    dcc.Upload(id="upload-runswmm", children=html.Div(["Drag & drop or ", html.A("runswmm.exe")]),
               style={"width":"50%","height":"60px","lineHeight":"60px",
                      "border":"2px dashed #d3ffd3","borderRadius":"5px",
                      "textAlign":"center","margin":"20px auto"},
               multiple=False, accept=".exe"),
    html.Div(id="runswmm-upload-status"),
    html.Hr(),
    html.H3("3. SWMM Engine Library (swmm5.dll)"),
    dcc.Upload(id="upload-swmm5", children=html.Div(["Drag & drop or ", html.A("swmm5.dll")]),
               style={"width":"50%","height":"60px","lineHeight":"60px",
                      "border":"2px dashed #ffd3d3","borderRadius":"5px",
                      "textAlign":"center","margin":"20px auto"},
               multiple=False, accept=".dll"),
    html.Div(id="swmm5-upload-status"),
], fluid=True)

# ... define subcatchments_layout, simulation_layout, lid_layout, etc. as before ...

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
    ], id="tabs", active_tab="upload", persistence=True, persistence_type="session", className="mb-4"),
    html.Div(id="page-content")
], fluid=True)

@app.callback(Output("page-content","children"), Input("tabs","active_tab"))
def render_content(tab):
    if tab=="upload": return upload_layout
    if tab=="subcatchments": return subcatchments_layout
    if tab=="simulation":     return simulation_layout
    if tab=="lid_definition": return lid_layout
    if tab=="calculate_lid_area": return calculate_lid_area_layout
    if tab=="total_lid_area": return total_lid_area_layout
    if tab=="calculate_lid_cost": return calculate_lid_cost_layout
    if tab=="updated_simulation": return updated_simulation_layout
    if tab=="pond_cost": return pond_cost_layout
    if tab=="total_cost": return total_cost_layout
    if tab=="swat_pond": return swat_pond_layout
    return "No tab selected."

###############################################################################
# 5. CALLBACKS
###############################################################################

# 5A. save .inp
@app.callback(
    [Output("upload-status","children"), Output("stored-file-path","data")],
    Input("upload-inp","contents"), State("upload-inp","filename")
)
def save_inp(contents,fn):
    if contents:
        _,b64=contents.split(",")
        data=base64.b64decode(b64)
        os.makedirs("uploads", exist_ok=True)
        path=os.path.join("uploads",fn)
        with open(path,"wb") as f: f.write(data)
        return f".inp saved to {path}", path
    return "No .inp uploaded.", ""

# 5B. save runswmm.exe
@app.callback(
    [Output("runswmm-upload-status","children"), Output("stored-runswmm-file","data")],
    Input("upload-runswmm","contents"), State("upload-runswmm","filename")
)
def save_runswmm(contents,fn):
    if contents:
        if not fn.lower().endswith(".exe"):
            return "Only .exe allowed.", ""
        _,b64=contents.split(",")
        data=base64.b64decode(b64)
        os.makedirs("uploads", exist_ok=True)
        path=os.path.join("uploads",fn)
        with open(path,"wb") as f: f.write(data)
        return f"runswmm.exe saved to {path}", path
    return "No runswmm.exe uploaded.", ""

# 5C. save swmm5.dll
@app.callback(
    [Output("swmm5-upload-status","children"), Output("stored-swmm5-file","data")],
    Input("upload-swmm5","contents"), State("upload-swmm5","filename")
)
def save_swmm5(contents,fn):
    if contents:
        if not fn.lower().endswith(".dll"):
            return "Only .dll allowed.", ""
        _,b64=contents.split(",")
        data=base64.b64decode(b64)
        os.makedirs("uploads", exist_ok=True)
        path=os.path.join("uploads",fn)
        with open(path,"wb") as f: f.write(data)
        return f"swmm5.dll saved to {path}", path
    return "No swmm5.dll uploaded.", ""

# 5D. extract subcatchments
@app.callback(
    [Output("file-info-subcatch","children"), Output("subcatchment-data","children")],
    Input("extract-btn","n_clicks"), State("stored-file-path","data")
)
def extract_subcatchments_data(n,fp):
    if n>0:
        if not fp: return "No .inp.", ""
        raw=open(fp,"rb").read()
        data=extract_subcatchments(raw)
        if not data: return "No data found.",""
        df=pd.DataFrame(data,columns=["Subcatchment","Area","%Imp","Width"])
        df["Area"]=pd.to_numeric(df["Area"])
        df["%Imp"]=pd.to_numeric(df["%Imp"])
        table=dash_table.DataTable(data=df.to_dict("records"),
                                  columns=[{"name":c,"id":c} for c in df.columns],
                                  page_size=10,style_table={"overflowX":"auto"})
        fig1=px.pie(df,names="Subcatchment",values="Area",title="Areas")
        fig2=px.treemap(df,path=["Subcatchment"],values="%Imp",title="% Imperv")
        content=html.Div([dbc.Row(dbc.Col(table,width=12)), dbc.Row([dbc.Col(dcc.Graph(fig1),width=6), dbc.Col(dcc.Graph(fig2),width=6)])])
        return f"Using {fp}", content
    return "",""

# 5E. run original SWMM
@app.callback(
    [Output("file-info-sim","children"), Output("sim-results","children"),
     Output("stored-original-total-flow","data"), Output("stored-original-peak-flow","data")],
    Input("run-sim-btn","n_clicks"),
    State("stored-file-path","data"),
    State("stored-runswmm-file","data"),
    State("stored-swmm5-file","data"),
)
def run_swmm_simulation(n, inp_path, exe_path, dll_path):
    if n>0:
        if not inp_path: return "No .inp.", "", None, None
        if not exe_path: return "No runswmm.exe.", "", None, None

        # ensure SWMM finds its DLL
        exe_dir=os.path.dirname(exe_path)
        if dll_path:
            dll_dir=os.path.dirname(dll_path)
            os.environ["PATH"]=dll_dir+os.pathsep+os.environ.get("PATH","")
        os.chdir(exe_dir)

        cmd=[exe_path, inp_path, "swmm_report.rpt"]
        result=subprocess.run(cmd, capture_output=True, text=True)
        print("CMD:",cmd)
        print("RC:",result.returncode)
        print("OUT:",result.stdout)
        print("ERR:",result.stderr)
        if result.returncode!=0:
            return f"Using {inp_path}", html.Pre(f"SWMM error ({result.returncode}):\n{result.stderr}"), None, None

        # parse report
        total,peak=None,None
        found=False
        for line in open("swmm_report.rpt"):
            if "Outfall Node" in line:
                found=True; continue
            if found:
                if line.strip().startswith("OF1"):
                    p=line.split()
                    if len(p)>=5:
                        peak=float(p[3])
                        total=float(p[4])*0.134*1e6
                    break
                if not line.strip():
                    found=False

        if peak is None:
            return f"Using {inp_path}", "Could not find OF1.", None, None

        results=html.Div([html.H3("Results"), html.P(f"Total: {total:,.2f}"), html.P(f"Peak: {peak:,.2f}")])
        return f"Using {inp_path}", results, total, peak

    return "","",None,None

# 5F. LID plan callback
# ... unchanged ...

# 5G. updated SWMM simulation (apply same exe/dll logic) ...
@app.callback(
    [Output("file-info-updated","children"), Output("updated-sim-results","children"), Output("stored-updated-total-flow","data")],
    Input("run-updated-sim-btn","n_clicks"),
    State("stored-runswmm-file","data"),
    State("stored-swmm5-file","data"),
    State("stored-original-total-flow","data"),
    State("stored-original-peak-flow","data"),
)
def run_updated_simulation(n, exe_path, dll_path, orig_total, orig_peak):
    if n>0:
        if not exe_path: return "No runswmm.exe.", "", None
        # chdir & set PATH
        exe_dir=os.path.dirname(exe_path)
        if dll_path:
            os.environ["PATH"]=os.path.dirname(dll_path)+os.pathsep+os.environ.get("PATH","")
        os.chdir(exe_dir)

        cmd=[exe_path, "Update.inp", "updated_swmm_report.rpt"]
        result=subprocess.run(cmd, capture_output=True, text=True)
        print("UPDATED CMD:",cmd)
        print("RC:",result.returncode)
        print("ERR:",result.stderr)
        if result.returncode!=0:
            return "Update.inp", html.Pre(f"SWMM error ({result.returncode}):\n{result.stderr}"), None

        # parse same way...
        total_u, peak_u=None,None
        found=False
        for line in open("updated_swmm_report.rpt"):
            if "Outfall Node" in line:
                found=True; continue
            if found:
                if line.strip().startswith("OF1"):
                    p=line.split()
                    if len(p)>=5:
                        peak_u=float(p[3])
                        total_u=float(p[4])*0.134*1e6
                    break
                if not line.strip():
                    found=False

        if peak_u is None:
            return "Update.inp", "Could not find OF1 in updated.", None

        # build your combined graph...
        div=html.Div([html.P(f"Tot_u={total_u}"), html.P(f"Peak_u={peak_u}")])
        return "Update.inp", div, total_u

    return "","",None

# 5H-5K other callbacks unchanged...

###############################################################################
# 6. RUN THE APP
###############################################################################
if __name__=="__main__":
    app.run_server(debug=True, port=8064)
