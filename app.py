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

# Flask server instance (will be overwritten by app.server later)
server = Flask(__name__)

###############################################################################
# 1. HELPER FUNCTIONS
###############################################################################
def extract_subcatchments(file_content):
    """Parse SWMM INP file to extract subcatchment data."""
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
                subcatchment_name = parts[0]
                area = parts[3]
                imp0 = parts[4]
                width = parts[5]
                data.append((subcatchment_name, area, imp0, width))
    return data

# Allowed LIDs for each subcatchment
allowed_lids = {
    27: ["BR", "IT", "RB", "GR", "PP", "VS"],
    18: ["BR", "IT", "RB", "GR", "PP", "VS"],
    11: ["BR", "IT", "RB", "GR", "PP", "VS"]
}
all_lid_types = ["BR", "IT", "RB", "GR", "PP", "VS"]

def update_LID(new_values, inp_file_path=None):
    """Update the INP file with user-defined LID values, write to Update.inp."""
    if inp_file_path is None:
        inp_file_path = "LID-Model.inp"
    output_file_path = "Update.inp"
    with open(inp_file_path, 'r') as f:
        content = f.readlines()
    with open(inp_file_path, 'rb') as f:
        file_bytes = f.read()
    sub_data = extract_subcatchments(file_bytes)
    sub_lid_sums = {}
    for (subc, _), val in new_values.items():
        sub_lid_sums[subc] = sub_lid_sums.get(subc, 0) + val
    in_LID_USAGE = False
    updated = []
    for line in content:
        strip_line = line.strip()
        if strip_line.startswith("[LID_USAGE]"):
            in_LID_USAGE = True
            updated.append(line)
            continue
        if in_LID_USAGE and strip_line.startswith("[") and strip_line.endswith("]"):
            in_LID_USAGE = False
        if in_LID_USAGE and strip_line and not strip_line.startswith(";"):
            parts = strip_line.split()
            if len(parts) >= 5:
                sc_name = parts[0]
                lid_name = parts[1]
                key = (sc_name, lid_name)
                if key in new_values:
                    sc_area = next((float(s[1]) for s in sub_data if s[0] == sc_name), None)
                    if sc_area is not None:
                        parts[3] = str(new_values[key] / 100 * sc_area * 43560)
                line = " ".join(parts) + "\n"
        updated.append(line)
    for i, line in enumerate(updated):
        strip_line = line.strip()
        if strip_line.startswith("[SUBCATCHMENTS]"):
            for j in range(i + 1, len(updated)):
                sub_line = updated[j].strip()
                if sub_line.startswith("[") and sub_line.endswith("]"):
                    break
                if sub_line and not sub_line.startswith(";"):
                    parts = sub_line.split()
                    if len(parts) >= 4:
                        sc_name = parts[0]
                        old_imp = next((float(s[2]) for s in sub_data if s[0] == sc_name), None)
                        a = sub_lid_sums.get(sc_name, 0) / 100
                        if old_imp is not None and a < 1:
                            imp_new = (old_imp - a * 100) / (1 - a)
                            if imp_new < 1:
                                imp_new = 1
                            parts[4] = f"{imp_new:.2f}"
                            updated[j] = " ".join(parts) + "\n"
    with open(output_file_path, 'w') as f:
        f.writelines(updated)
    return output_file_path

def SWAT(swatworking_directory, UpsIn):
    """
    Modifies a SWAT input file, runs the SWAT executable,
    processes the output, and returns the peak outflow.
    """
    coeff = 0.0283168  # converting from ft³ to m³
    unit_filter = 275  # downstream river filter
    os.chdir(swatworking_directory)
    input_swatfile_path = 'exco_om.exc'
    output_swatfile_path = 'exco_om.exc'
    swat_executable = "rev61.0_64rel.exe"
    try:
        target_column = 'flo'
        with open(input_swatfile_path, 'r') as file:
            file_lines = file.readlines()
        header_line_index = 1
        data_start_index = 2
        header = file_lines[header_line_index].strip().split()
        data_lines = file_lines[data_start_index:]
        data_str = "\n".join(data_lines)
        data = pd.read_csv(StringIO(data_str), delim_whitespace=True, header=None, names=header)
        if target_column in data.columns:
            data[target_column] = UpsIn * coeff
        else:
            raise ValueError(f"Column '{target_column}' not found in the file.")
        with open(output_swatfile_path, 'w') as file:
            file.writelines(file_lines[:data_start_index])
            data.to_csv(file, sep='\t', index=False, header=False)
        subprocess.run([swat_executable], check=True, capture_output=True, text=True)
        initial_file_path = "channel_sd_day.txt"
        intermediate_csv_path = "channel_sd_day.csv"
        data = pd.read_csv(initial_file_path, delim_whitespace=True, skiprows=1)
        data.to_csv(intermediate_csv_path, index=False)
        columns_to_keep = ['unit', 'flo_out']
        selected_data = data[columns_to_keep]
        rows_to_remove = [0]
        data_without_rows = selected_data.drop(index=rows_to_remove)
        filtered_data = data_without_rows[data_without_rows['unit'] == unit_filter]
        outflow = filtered_data['flo_out'].max()
        return outflow
    except subprocess.CalledProcessError as e:
        return f"Error during subprocess execution: {e.stderr}"
    except Exception as e:
        return f"An error occurred: {e}"

###############################################################################
# 2. APP & SERVER
###############################################################################
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # expose the Flask server

###############################################################################
# 3. PAGE LAYOUTS
###############################################################################
# Page A: Upload Files
upload_layout = dbc.Container([
    html.H1("Upload Files"),

    # -- INP upload --
    html.H3("Upload SWMM INP File"),
    dcc.Upload(
        id="upload-inp",
        children=html.Div(["Drag & Drop / ", html.A("Select Input SWMM File")]),
        style={
            "width": "50%", "height": "60px", "lineHeight": "60px",
            "borderWidth": "2px", "borderStyle": "dashed", "borderRadius": "5px",
            "textAlign": "center", "margin": "20px auto", "backgroundColor": "#add8e6"
        },
        multiple=False,
        accept=".inp"
    ),
    html.Div(id="upload-status"),
    html.Hr(),

    # -- runswmm.exe upload --
    html.H3("Upload runswmm.exe"),
    dcc.Upload(
        id="upload-runswmm",
        children=html.Div(["Drag & Drop / ", html.A("Select runswmm.exe")]),
        style={
            "width": "50%", "height": "60px", "lineHeight": "60px",
            "borderWidth": "2px", "borderStyle": "dashed", "borderRadius": "5px",
            "textAlign": "center", "margin": "20px auto", "backgroundColor": "#d3ffd3"
        },
        multiple=False,
        accept=".exe"
    ),
    html.Div(id="runswmm-upload-status"),
    html.Hr(),

    # -- swmm5.exe upload --
    html.H3("Upload swmm5.exe"),
    dcc.Upload(
        id="upload-swmm5",
        children=html.Div(["Drag & Drop / ", html.A("Select swmm5.exe")]),
        style={
            "width": "50%", "height": "60px", "lineHeight": "60px",
            "borderWidth": "2px", "borderStyle": "dashed", "borderRadius": "5px",
            "textAlign": "center", "margin": "20px auto", "backgroundColor": "#ffd3d3"
        },
        multiple=False,
        accept=".exe"
    ),
    html.Div(id="swmm5-upload-status"),
    html.Hr(),

    # -- swmm6.exe upload --
    html.H3("Upload swmm6.exe"),
    dcc.Upload(
        id="upload-swmm6",
        children=html.Div(["Drag & Drop / ", html.A("Select swmm6.exe")]),
        style={
            "width": "50%", "height": "60px", "lineHeight": "60px",
            "borderWidth": "2px", "borderStyle": "dashed", "borderRadius": "5px",
            "textAlign": "center", "margin": "20px auto", "backgroundColor": "#e0d3ff"
        },
        multiple=False,
        accept=".exe"
    ),
    html.Div(id="swmm6-upload-status")
], fluid=True)

# ... (The rest of your page layouts B–K remain exactly as before) ...

###############################################################################
# 4. MAIN APP LAYOUT & TAB NAVIGATION
###############################################################################
app.layout = dbc.Container([
    dcc.Store(id="stored-file-path"),
    dcc.Store(id="stored-runswmm-file"),
    dcc.Store(id="stored-swmm5-file"),
    dcc.Store(id="stored-swmm6-file"),
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
        dbc.Tab(label="SWAT Simulation", tab_id="swat_pond"),
    ], id="tabs", active_tab="upload",
       persistence=True, persistence_type="session", className="mb-4"),
    html.Div(id="page-content")
], fluid=True)

# Callback to render page content remains unchanged
@app.callback(Output("page-content", "children"), Input("tabs", "active_tab"))
def render_content(tab):
    if tab == "upload": return upload_layout
    elif tab == "subcatchments": return subcatchments_layout
    elif tab == "simulation": return simulation_layout
    elif tab == "lid_definition": return lid_layout
    elif tab == "calculate_lid_area": return calculate_lid_area_layout
    elif tab == "total_lid_area": return total_lid_area_layout
    elif tab == "calculate_lid_cost": return calculate_lid_cost_layout
    elif tab == "updated_simulation": return updated_simulation_layout
    elif tab == "pond_cost": return pond_cost_layout
    elif tab == "total_cost": return total_cost_layout
    elif tab == "swat_pond": return swat_pond_layout
    return "No tab selected."

###############################################################################
# 5. UPLOAD CALLBACKS
###############################################################################

# 5A. Upload INP file
@app.callback(
    [Output("upload-status", "children"), Output("stored-file-path", "data")],
    Input("upload-inp", "contents"), State("upload-inp", "filename")
)
def save_inp(contents, filename):
    if contents:
        ctype, data = contents.split(',')
        decoded = base64.b64decode(data)
        os.makedirs("uploads", exist_ok=True)
        path = os.path.join("uploads", filename)
        with open(path, "wb") as f: f.write(decoded)
        files = os.listdir("uploads")
        return f".inp '{filename}' saved. uploads/: {', '.join(files)}", path
    return "No .inp uploaded.", None

# 5B. Upload runswmm.exe
@app.callback(
    [Output("runswmm-upload-status", "children"), Output("stored-runswmm-file", "data")],
    Input("upload-runswmm", "contents"), State("upload-runswmm", "filename")
)
def save_runswmm(contents, filename):
    if contents:
        ctype, data = contents.split(',')
        decoded = base64.b64decode(data)
        os.makedirs("uploads", exist_ok=True)
        path = os.path.join("uploads", filename)
        with open(path, "wb") as f: f.write(decoded)
        files = os.listdir("uploads")
        return f"runswmm.exe '{filename}' saved. uploads/: {', '.join(files)}", path
    return "No runswmm.exe uploaded.", None

# 5C. Upload swmm5.exe
@app.callback(
    [Output("swmm5-upload-status", "children"), Output("stored-swmm5-file", "data")],
    Input("upload-swmm5", "contents"), State("upload-swmm5", "filename")
)
def save_swmm5(contents, filename):
    if contents:
        ctype, data = contents.split(',')
        decoded = base64.b64decode(data)
        os.makedirs("uploads", exist_ok=True)
        path = os.path.join("uploads", filename)
        with open(path, "wb") as f: f.write(decoded)
        files = os.listdir("uploads")
        return f"swmm5.exe '{filename}' saved. uploads/: {', '.join(files)}", path
    return "No swmm5.exe uploaded.", None

# 5D. Upload swmm6.exe
@app.callback(
    [Output("swmm6-upload-status", "children"), Output("stored-swmm6-file", "data")],
    Input("upload-swmm6", "contents"), State("upload-swmm6", "filename")
)
def save_swmm6(contents, filename):
    if contents:
        ctype, data = contents.split(',')
        decoded = base64.b64decode(data)
        os.makedirs("uploads", exist_ok=True)
        path = os.path.join("uploads", filename)
        with open(path, "wb") as f: f.write(decoded)
        files = os.listdir("uploads")
        return f"swmm6.exe '{filename}' saved. uploads/: {', '.join(files)}", path
    return "No swmm6.exe uploaded.", None

# 5E–5K: (All your existing callbacks for subcatchments, simulation, LIDs, SWAT, etc.) remain exactly as before.

###############################################################################
# 6. RUN THE APP
###############################################################################
if __name__ == "__main__":
    app.run_server(debug=True, port=8054)

# Expose Flask server
server = app.server
