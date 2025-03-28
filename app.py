#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Urban + Rural


# In[3]:


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
        print("Coefficient:", coeff)
        print("UpsIn value:", UpsIn)
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
        selected_columns_path = "columns.csv"
        selected_data = data[columns_to_keep]
        selected_data.to_csv(selected_columns_path, index=False)
        rows_to_remove = [0]
        modified_columns_path = "columns1.csv"
        data_without_rows = selected_data.drop(index=rows_to_remove)
        data_without_rows.to_csv(modified_columns_path, index=False)
        filtered_file_path = "flo_out.csv"
        data_clean = pd.read_csv(modified_columns_path)
        filtered_data = data_clean[data_clean['unit'] == unit_filter]
        filtered_data.to_csv(filtered_file_path, index=False)
        data2 = pd.read_csv('flo_out.csv')
        outflow = max(data2['flo_out'])
        return outflow
    except subprocess.CalledProcessError as e:
        return f"Error during subprocess execution: {e.stderr}"
    except Exception as e:
        return f"An error occurred: {e}"

###############################################################################
# 2. APP & SERVER
###############################################################################
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

###############################################################################
# 3. PAGE LAYOUTS
###############################################################################
# Page A: Upload SWMM INP File
upload_layout = dbc.Container([
    html.H1("Upload SWMM File"),
    dcc.Upload(
        id="upload-inp",
        children=html.Div(["Drag and Drop / ", html.A("Select Input SWMM File")]),
        style={
            "width": "50%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "2px",
            "borderStyle": "dashed",
            "borderRadius": "5px",
            "textAlign": "center",
            "margin": "20px auto",
            "backgroundColor": "#add8e6"
        },
        multiple=False,
        accept=".inp"
    ),
    html.Div(id="upload-status")
], fluid=True)

# Page B: Subcatchment Data Extraction
subcatchments_layout = dbc.Container([
    html.H1("Subcatchments"),
    html.Div(id="file-info-subcatch"),
    dbc.Button("Subcatchments Data", id="extract-btn", color="primary", className="mb-3", n_clicks=0),
    html.Div(id="subcatchment-data")
], fluid=True)

# Page C: SWMM Simulation Runner (Original Simulation)
simulation_layout = dbc.Container([
    html.H1("SWMM Simulation"),
    html.Div(id="file-info-sim"),
    dbc.Button("Run SWMM", id="run-sim-btn", color="primary", className="mb-3", n_clicks=0),
    html.Div(id="sim-results")
], fluid=True)

# Page D: Define LIDs for Subcatchment (Three Groups)
lid_layout = dbc.Container([
    html.H1("BR=Bioretention Cell, IT=Infiltration Trench, RB=Rain Barrel, GR=Green Roof, PP=Permeable Pavement, VS=Vegetated Swale", style={"fontSize": "16px"}),
    html.Div(id="file-info-lid"),
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Subcatchment Group 1"),
                dbc.CardBody([
                    dcc.Dropdown(
                        id="dropdown-subc-1",
                        options=[{"label": str(sc), "value": sc} for sc in allowed_lids.keys()],
                        value=list(allowed_lids.keys())[0],
                        clearable=False
                    ),
                    html.Div([
                        html.Div([
                            html.Label(lid),
                            dcc.Input(
                                id={"type": "lid-input", "group": 1, "lid": lid},
                                type="number",
                                placeholder=f"Value for {lid}"
                            )
                        ], style={"margin": "10px", "display": "inline-block"})
                        for lid in all_lid_types
                    ])
                ])
            ], className="mb-3"),
            width=4
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Subcatchment Group 2"),
                dbc.CardBody([
                    dcc.Dropdown(
                        id="dropdown-subc-2",
                        options=[{"label": str(sc), "value": sc} for sc in allowed_lids.keys()],
                        value=list(allowed_lids.keys())[1],
                        clearable=False
                    ),
                    html.Div([
                        html.Div([
                            html.Label(lid),
                            dcc.Input(
                                id={"type": "lid-input", "group": 2, "lid": lid},
                                type="number",
                                placeholder=f"Value for {lid}"
                            )
                        ], style={"margin": "10px", "display": "inline-block"})
                        for lid in all_lid_types
                    ])
                ])
            ], className="mb-3"),
            width=4
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Subcatchment Group 3"),
                dbc.CardBody([
                    dcc.Dropdown(
                        id="dropdown-subc-3",
                        options=[{"label": str(sc), "value": sc} for sc in allowed_lids.keys()],
                        value=list(allowed_lids.keys())[2],
                        clearable=False
                    ),
                    html.Div([
                        html.Div([
                            html.Label(lid),
                            dcc.Input(
                                id={"type": "lid-input", "group": 3, "lid": lid},
                                type="number",
                                placeholder=f"Value for {lid}"
                            )
                        ], style={"margin": "10px", "display": "inline-block"})
                        for lid in all_lid_types
                    ])
                ])
            ], className="mb-3"),
            width=4
        )
    ]),
    dbc.Button("Submit LIDs Design", id="lid-submit-btn", color="primary", n_clicks=0),
    html.Div(id="lid-plan-output")
], fluid=True)

# Page E: Calculate LID Area for Each (Subcatchment, LID) Pair
calculate_lid_area_layout = dbc.Container([
    html.H1("LID Area"),
    html.Div(id="file-info-calc"),
    dbc.Button("LID Areas", id="calc-lid-btn", color="primary", className="mb-3", n_clicks=0),
    html.Div(id="calc-lid-output")
], fluid=True)

# Page F: Calculate Total LID Areas by Type
total_lid_area_layout = dbc.Container([
    html.H1("LID Type"),
    html.Div(id="file-info-total"),
    dbc.Button("LID Areas", id="calc-total-lid-btn", color="primary", className="mb-3", n_clicks=0),
    html.Div(id="calc-total-lid-output")
], fluid=True)

# Page G: Calculate LID Cost
calculate_lid_cost_layout = dbc.Container([
    html.H1("LID Cost"),
    html.Div(id="file-info-cost"),
    dbc.Button("LID Cost", id="calc-lid-cost-btn", color="primary", className="mb-3", n_clicks=0),
    html.Div(id="calc-lid-cost-output")
], fluid=True)

# Page H: Run Updated SWMM Simulation (LID Simulation)
updated_simulation_layout = dbc.Container([
    html.H1("LID Simulation"),
    html.Div(id="file-info-updated"),
    dbc.Button("Run LID Simulation", id="run-updated-sim-btn", color="primary", className="mb-3", n_clicks=0),
    html.Div(id="updated-sim-results")
], fluid=True)

# Page I: Calculate Pond Cost
pond_cost_layout = dbc.Container([
    html.H1("Pond Cost"),
    html.Label("Pond Depth (ft):"),
    dcc.Input(id="pond-cost-depth", type="number", placeholder="Enter pond depth (ft)", value=0),
    html.Br(),
    html.Label("Pond Area (acres):"),
    dcc.Input(id="pond-cost-area", type="number", placeholder="Enter pond area (acres)", value=0),
    html.Br(),
    dbc.Button("Calculate Pond Cost", id="calc-pond-cost-btn", color="primary", className="mb-3", n_clicks=0),
    html.Div(id="pond-cost-output")
], fluid=True)

# Page J: Define Pond & Run SWAT Outflow
swat_pond_layout = dbc.Container([
    html.H1("SWAT+ Simulation"),
    html.Label("Pond Depth (ft):"),
    dcc.Input(id="pond-depth", type="number", placeholder="Enter pond depth (ft)", value=0),
    html.Br(),
    html.Label("Pond Area (acres):"),
    dcc.Input(id="pond-area", type="number", placeholder="Enter pond area (acres)", value=0),
    html.Br(),
    html.Label("SWAT Working Directory:"),
    dcc.Input(id="swat-dir", type="text", placeholder="Enter SWAT working directory", value="TxtInOut"),
    html.Br(),
    dbc.Button("Run SWAT", id="run-swat-pond-btn", color="primary", className="mb-3", n_clicks=0),
    html.Div(id="swat-pond-output")
], fluid=True)

# Page K: Total Cost (LID Cost + Pond Cost)
total_cost_layout = dbc.Container([
    html.H1("Total Cost"),
    dbc.Button("Calculate Total Cost", id="total-cost-btn", color="primary", n_clicks=0),
    html.Div(id="total-cost-output")
], fluid=True)

###############################################################################
# 4. MAIN APP LAYOUT & TAB NAVIGATION
###############################################################################
app.layout = dbc.Container([
    dcc.Store(id="stored-file-path"),
    dcc.Store(id="stored-lid-plan"),
    dcc.Store(id="stored-original-total-flow"),
    dcc.Store(id="stored-original-peak-flow"),
    dcc.Store(id="stored-updated-total-flow"),
    dcc.Store(id="stored-lid-cost"),
    dcc.Store(id="stored-pond-cost"),
    dbc.Tabs(
        [
            dbc.Tab(label="Upload SWMM File", tab_id="upload"),
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
        ],
        id="tabs",
        active_tab="upload",
        persistence=True,
        persistence_type="session",
        className="mb-4"
    ),
    html.Div(id="page-content")
], fluid=True)

# Callback to render page content based on active tab
@app.callback(
    Output("page-content", "children"),
    Input("tabs", "active_tab")
)
def render_content(active_tab):
    if active_tab == "upload":
        return upload_layout
    elif active_tab == "subcatchments":
        return subcatchments_layout
    elif active_tab == "simulation":
        return simulation_layout
    elif active_tab == "lid_definition":
        return lid_layout
    elif active_tab == "calculate_lid_area":
        return calculate_lid_area_layout
    elif active_tab == "total_lid_area":
        return total_lid_area_layout
    elif active_tab == "calculate_lid_cost":
        return calculate_lid_cost_layout
    elif active_tab == "updated_simulation":
        return updated_simulation_layout
    elif active_tab == "pond_cost":
        return pond_cost_layout
    elif active_tab == "total_cost":
        return total_cost_layout
    elif active_tab == "swat_pond":
        return swat_pond_layout
    return "No tab selected."

###############################################################################
# 5. CALLBACKS
###############################################################################
# 5A. Upload Page Callback
@app.callback(
    [Output("upload-status", "children"),
     Output("stored-file-path", "data")],
    Input("upload-inp", "contents"),
    State("upload-inp", "filename")
)
def save_inp(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", filename)
        with open(file_path, "wb") as f:
            f.write(decoded)
        return f"File {filename} uploaded. Path: {file_path}", file_path
    return "No file uploaded yet.", ""

# 5B. Subcatchment Data Extraction Callback with Plotly Graph
@app.callback(
    [Output("file-info-subcatch", "children"),
     Output("subcatchment-data", "children")],
    Input("extract-btn", "n_clicks"),
    State("stored-file-path", "data")
)
def extract_subcatchments_data(n_clicks, file_path):
    if n_clicks > 0:
        if not file_path:
            return "No file selected.", ""
        info = f"Using file: {file_path}"
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        data = extract_subcatchments(file_bytes)
        if data:
            df = pd.DataFrame(data, columns=["Subcatchment", "Area", "%Imperv", "Width"])
            try:
                df["Area"] = pd.to_numeric(df["Area"])
                df["%Imperv"] = pd.to_numeric(df["%Imperv"])
            except Exception:
                pass
            table = dash_table.DataTable(
                data=df.to_dict("records"),
                columns=[{"name": col, "id": col} for col in df.columns],
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left"}
            )
            fig_area = px.pie(df, names="Subcatchment", values="Area", title="Subcatchment Areas (acres)")
            fig_imperv = px.treemap(df, path=["Subcatchment"], values="%Imperv", title="Subcatchment % Imperviousness")
            content = html.Div([
                dbc.Row([dbc.Col(table, width=12)], className="mb-4"),
                dbc.Row([dbc.Col(dcc.Graph(figure=fig_area), width=6),
                         dbc.Col(dcc.Graph(figure=fig_imperv), width=6)])
            ])
            return info, content
        else:
            return info, "No subcatchment data found."
    return "", ""

# 5C. SWMM Simulation Runner Callback (Original Simulation)
@app.callback(
    [Output("file-info-sim", "children"),
     Output("sim-results", "children"),
     Output("stored-original-total-flow", "data"),
     Output("stored-original-peak-flow", "data")],
    Input("run-sim-btn", "n_clicks"),
    State("stored-file-path", "data")
)
def run_swmm_simulation(n_clicks, file_path):
    if n_clicks > 0:
        if not file_path:
            return "No file selected.", "", None, None

        info = f"Using file: {file_path}"
        swmm_executable = "runswmm.exe"
        rpt_path = "swmm_report.rpt"
        cmd = [swmm_executable, file_path, rpt_path]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except Exception as e:
            return info, f"Error running SWMM: {e}", None, None

        section_found = False
        total_flow = None
        peak_flow = None

        try:
            with open(rpt_path, "r") as rep:
                for line in rep:
                    if "Outfall Node" in line:
                        section_found = True
                        continue
                    if section_found:
                        if line.strip().startswith("OF1"):
                            parts = line.split()
                            if len(parts) >= 5:
                                peak_flow = float(parts[3])
                                total_flow = float(parts[4]) * 0.134 * 1_000_000
                            break
                        if not line.strip():
                            section_found = False
        except Exception as e:
            return info, f"Error reading report: {e}", None, None

        if total_flow is None or peak_flow is None:
            return info, "Could not find total flow or peak flow in the report.", None, None

        results_content = [
            html.H3("Simulation Results:"),
            html.P(f"Total Flow Volume (ft³): {total_flow:,.2f}"),
            html.P(f"Peak Flow (cfs): {peak_flow:,.2f}")
        ]

        of1_csv_path = "OF1.csv"
        if os.path.exists(of1_csv_path):
            try:
                df = pd.read_csv(of1_csv_path)
                if df.shape[1] >= 3:
                    df.columns = ["Date", "Flow", "Depth"]
                    df["Date"] = pd.to_datetime(df["Date"])
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df["Date"], y=df["Flow"],
                        mode="lines",
                        name="Flow (cfs)",
                        line=dict(color="blue"),
                        yaxis="y1"
                    ))
                    fig.add_trace(go.Scatter(
                        x=df["Date"], y=df["Depth"],
                        mode="lines",
                        name="Depth (ft)",
                        line=dict(color="red", dash="dash"),
                        yaxis="y2"
                    ))
                    fig.update_layout(
                        xaxis=dict(title="Date/Time"),
                        yaxis=dict(
                            title=dict(text="Flow (cfs)", font=dict(color="black")),
                            tickfont=dict(color="black"),
                            side="left"
                        ),
                        yaxis2=dict(
                            title=dict(text="Depth (ft)", font=dict(color="black")),
                            tickfont=dict(color="black"),
                            overlaying="y",
                            side="right"
                        ),
                        legend=dict(x=0.02, y=0.98),
                        template="plotly_white"
                    )
                    original_graph = dcc.Graph(figure=fig)
                else:
                    original_graph = "Invalid format in OF1.csv. Ensure it has three columns (Date, Flow, Depth)."
            except Exception as e:
                original_graph = f"Error reading OF1.csv: {e}"
        else:
            original_graph = "OF1.csv file not found."

        results_content.append(html.H3(""))
        results_content.append(original_graph)

        return info, html.Div(results_content), total_flow, peak_flow

    return "", "", None, None

# 5D. LID Definition Submission Callback (Dynamic)
from dash.dependencies import ALL
@app.callback(
    [Output("file-info-lid", "children"),
     Output("lid-plan-output", "children"),
     Output("stored-lid-plan", "data")],
    Input("lid-submit-btn", "n_clicks"),
    State("stored-file-path", "data"),
    State({"type": "lid-input", "group": ALL, "lid": ALL}, "value"),
    State({"type": "lid-input", "group": ALL, "lid": ALL}, "id"),
    State("dropdown-subc-1", "value"),
    State("dropdown-subc-2", "value"),
    State("dropdown-subc-3", "value")
)
def submit_lid_plan(n_clicks, file_path, values, ids, sub1, sub2, sub3):
    if n_clicks > 0:
        if not file_path:
            return "No file selected.", "", None
        info = f"Using file: {file_path}"
        plan = {}
        for val, comp_id in zip(values, ids):
            group = comp_id["group"]
            if group == 1:
                subc = sub1
            elif group == 2:
                subc = sub2
            elif group == 3:
                subc = sub3
            else:
                subc = None
            lid = comp_id["lid"]
            if val is None:
                return info, f"Please enter a value for allowed LID {lid} for subcatchment {subc}.", None
            plan[f"{subc}_{lid}"] = val
        plan_for_update = { (str(k.split('_')[0]), k.split('_')[1]) : v for k, v in plan.items() }
        updated_file = update_LID(plan_for_update, inp_file_path=file_path)
        output_div = html.Div([])
        return info, output_div, plan
    return "", "", None

# 5E. Updated SWMM Simulation Callback (LID Simulation Page) with Combined Plot
@app.callback(
    [Output("file-info-updated", "children"),
     Output("updated-sim-results", "children"),
     Output("stored-updated-total-flow", "data")],
    [Input("run-updated-sim-btn", "n_clicks")],
    [State("stored-original-total-flow", "data"),
     State("stored-original-peak-flow", "data")]
)
def run_updated_simulation(n_clicks, original_total_flow, original_peak_flow):
    if n_clicks > 0:
        file_path = "Update.inp"
        if not os.path.exists(file_path):
            return "No updated file found.", "Please update the file using the LID Definition page first.", None
        
        info = f"Using updated file: {file_path}"
        swmm_executable = "runswmm.exe"
        rpt_path = "updated_swmm_report.rpt"
        cmd = [swmm_executable, file_path, rpt_path]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except Exception as e:
            return info, f"Error running updated SWMM: {e}", None
        
        section_found = False
        total_flow_updated = None
        peak_flow_updated = None
        try:
            with open(rpt_path, "r") as rep:
                for line in rep:
                    if "Outfall Node" in line:
                        section_found = True
                        continue
                    if section_found:
                        if line.strip().startswith("OF1"):
                            parts = line.split()
                            if len(parts) >= 5:
                                peak_flow_updated = float(parts[3])
                                total_flow_updated = float(parts[4]) * 0.134 * 1_000_000
                            break
                        if not line.strip():
                            section_found = False
        except Exception as e:
            return info, f"Error reading updated report: {e}", None
        
        if total_flow_updated is None or peak_flow_updated is None:
            return info, "Could not find total flow or peak flow in the updated report.", None
        
        if original_total_flow and original_total_flow != 0:
            total_reduction = ((total_flow_updated - original_total_flow) / original_total_flow) * 100
        else:
            total_reduction = None
        
        if original_peak_flow and original_peak_flow != 0:
            peak_reduction = ((peak_flow_updated - original_peak_flow) / original_peak_flow) * 100
        else:
            peak_reduction = None
        
        total_reduction_text = (
            f"Total Flow Volume Reduction: {total_reduction:.2f}%"
            if total_reduction is not None else "Total Flow reduction not available"
        )
        peak_reduction_text = (
            f"Peak Flow Reduction: {peak_reduction:.2f}%"
            if peak_reduction is not None else "Peak Flow reduction not available"
        )
        
        factor = 1 + (peak_reduction/100) if peak_reduction is not None else 1
        
        of1_csv_path = "OF1.csv"
        if os.path.exists(of1_csv_path):
            try:
                df = pd.read_csv(of1_csv_path)
                if df.shape[1] >= 3:
                    df.columns = ["Date", "Flow", "Depth"]
                    df["Date"] = pd.to_datetime(df["Date"])
                    df["ModifiedFlow"] = df["Flow"] * factor
                    df["ModifiedDepth"] = df["Depth"] * factor

                    combined_fig = go.Figure()
                    combined_fig.add_trace(go.Scatter(
                        x=df["Date"], y=df["Flow"],
                        mode="lines",
                        name="Original Flow",
                        line=dict(color="blue", dash="solid"),
                        yaxis="y1"
                    ))
                    combined_fig.add_trace(go.Scatter(
                        x=df["Date"], y=df["ModifiedFlow"],
                        mode="lines",
                        name="Modified Flow",
                        line=dict(color="green", dash="dash"),
                        yaxis="y1"
                    ))
                    combined_fig.add_trace(go.Scatter(
                        x=df["Date"], y=df["Depth"],
                        mode="lines",
                        name="Original Depth",
                        line=dict(color="red", dash="solid"),
                        yaxis="y2"
                    ))
                    combined_fig.add_trace(go.Scatter(
                        x=df["Date"], y=df["ModifiedDepth"],
                        mode="lines",
                        name="Modified Depth",
                        line=dict(color="orange", dash="dot"),
                        yaxis="y2"
                    ))
                    combined_fig.update_layout(
                        xaxis=dict(title="Date/Time"),
                        yaxis=dict(
                            title=dict(text="Flow (cfs)", font=dict(color="black")),
                            tickfont=dict(color="black"),
                            side="left"
                        ),
                        yaxis2=dict(
                            title=dict(text="Depth (ft)", font=dict(color="black")),
                            tickfont=dict(color="black"),
                            overlaying="y",
                            side="right"
                        ),
                        legend=dict(x=0.02, y=0.98),
                        template="plotly_white"
                    )
                    combined_graph = dcc.Graph(figure=combined_fig)
                else:
                    combined_graph = "Invalid format in OF1.csv. Ensure it has three columns (Date, Flow, Depth)."
            except Exception as e:
                combined_graph = f"Error reading OF1.csv: {e}"
        else:
            combined_graph = "OF1.csv file not found."

        results = html.Div([
            html.H3("LID Simulation Results:"),
            html.P(f"Total Flow Volume (ft³): {total_flow_updated:,.2f}"),
            html.P(f"Peak Flow (cfs): {peak_flow_updated:,.2f}"),
            html.P(total_reduction_text),
            html.P(peak_reduction_text),
            html.H3("Combined Flow and Depth Time Series:"),
            combined_graph
        ])
        
        return info, results, total_flow_updated
    return "", "", None

# 5F. Calculate LID Area Callback (Individual Calculations)
@app.callback(
    [Output("file-info-calc", "children"),
     Output("calc-lid-output", "children")],
    Input("calc-lid-btn", "n_clicks"),
    State("stored-file-path", "data"),
    State("stored-lid-plan", "data")
)
def calculate_lid_area(n_clicks, file_path, plan):
    if n_clicks > 0:
        if not file_path:
            return "No file selected.", "Please upload a file first."
        if not plan:
            return "No LID plan defined.", "Please define a LID plan first."
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        sub_data = extract_subcatchments(file_bytes)
        results = []
        for key, defined_val in plan.items():
            try:
                subc, lid = key.split("_")
            except Exception:
                continue
            area = None
            for entry in sub_data:
                if entry[0] == subc:
                    try:
                        area = float(entry[1])
                    except Exception:
                        area = None
                    break
            if area is None:
                calc_area = "Subcatchment not found"
            else:
                calc_area = defined_val / 100 * area * 43560
            results.append({
                "Subcatchment": subc,
                "LID Type": lid,
                "Defined Value (%)": defined_val,
                "Subcatchment Area (acres)": area if area is not None else "N/A",
                "LID Area (ft²)": calc_area
            })
        df = pd.DataFrame(results)
        if not df.empty:
            fig = px.bar(
                df,
                x="Subcatchment",
                y="LID Area (ft²)",
                color="LID Type",
                barmode="stack",
                title="LID Area by Subcatchment (Stacked)",
                labels={"LID Area (ft²)": "LID Area (ft²)", "Subcatchment": "Subcatchment"},
                template="plotly_white"
            )
            graph = dcc.Graph(figure=fig)
        else:
            graph = "No data available for plotting."
        table = dash_table.DataTable(
            data=df.to_dict("records"),
            columns=[{"name": col, "id": col} for col in df.columns],
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left"}
        )
        content = html.Div([
            dbc.Row([dbc.Col(table, width=12)], className="mb-4"),
            dbc.Row([dbc.Col(graph, width=12)])
        ])
        return "", content
    return "", ""

# 5G. Calculate Total LID Areas by Type Callback
@app.callback(
    [Output("file-info-total", "children"),
     Output("calc-total-lid-output", "children")],
    Input("calc-total-lid-btn", "n_clicks"),
    State("stored-file-path", "data"),
    State("stored-lid-plan", "data")
)
def calculate_total_lid_area(n_clicks, file_path, plan):
    if n_clicks > 0:
        if not file_path:
            return "No file selected.", "Please upload a file first."
        if not plan:
            return "No LID plan defined.", "Please define a LID plan first."
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        sub_data = extract_subcatchments(file_bytes)
        total_by_lid = {lid: 0 for lid in all_lid_types}
        for key, defined_val in plan.items():
            try:
                subc, lid = key.split("_")
            except Exception:
                continue
            area = None
            for entry in sub_data:
                if entry[0] == subc:
                    try:
                        area = float(entry[1])
                    except Exception:
                        area = None
                    break
            if area is not None:
                calc_area = defined_val / 100 * area * 43560
                total_by_lid[lid] += calc_area
        totals = []
        for lid, tot_area in total_by_lid.items():
            totals.append({
                "LID Type": lid,
                "Total LID Area (ft²)": tot_area
            })
        df = pd.DataFrame(totals)
        if not df.empty:
            fig = px.pie(
                df,
                names="LID Type",
                values="Total LID Area (ft²)",
                title="LID Area Distribution by Type",
                hole=0.4,
                template="plotly_white"
            )
            graph = dcc.Graph(figure=fig)
        else:
            graph = "No data available for plotting."
        table = dash_table.DataTable(
            data=df.to_dict("records"),
            columns=[{"name": col, "id": col} for col in df.columns],
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left"}
        )
        content = html.Div([
            dbc.Row([dbc.Col(table, width=12)], className="mb-4"),
            dbc.Row([dbc.Col(graph, width=12)])
        ])
        return "", content
    return "", ""

# 5H. Calculate LID Cost Callback
@app.callback(
    [Output("file-info-cost", "children"),
     Output("calc-lid-cost-output", "children"),
     Output("stored-lid-cost", "data")],
    Input("calc-lid-cost-btn", "n_clicks"),
    State("stored-file-path", "data"),
    State("stored-lid-plan", "data")
)
def calculate_lid_cost(n_clicks, file_path, plan):
    if n_clicks > 0:
        if not file_path:
            return "No file selected.", "Please upload a file first.", None
        if not plan:
            return "No LID plan defined.", "Please define a LID plan first.", None
        LocFactor = 0.907
        ENRCCI = 1.4208
        cost_formulas = {
            "BR": lambda x: 1.5691 * x + 3696,
            "IT": lambda x: 0.8473 * x + 3864,
            "PP": lambda x: 4.7209 * x + 1800,
            "RB": lambda x: 0.7697 * x + 3564,
            "VS": lambda x: 2.7125 * x + 2580.6,
            "GR": lambda x: 2.5009 * x + 3288,
        }
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        sub_data = extract_subcatchments(file_bytes)
        total_by_lid = {lid: 0 for lid in all_lid_types}
        for key, defined_val in plan.items():
            try:
                subc, lid = key.split("_")
            except Exception:
                continue
            area = None
            for entry in sub_data:
                if entry[0] == subc:
                    try:
                        area = float(entry[1])
                    except Exception:
                        area = None
                    break
            if area is not None:
                calc_area = defined_val / 100 * area * 43560
                total_by_lid[lid] += calc_area
        total_cost = 0
        cost_breakdown = []
        for lid, area in total_by_lid.items():
            if area == 0:
                cost = 0
            elif lid in cost_formulas:
                cost = cost_formulas[lid](area)
            else:
                cost = 0
            total_cost += cost
            cost_breakdown.append({
                "LID Type": lid,
                "Total LID Area (ft²)": area,
                "Cost (pre-adjustment)": cost
            })
        total_cost = total_cost * LocFactor * ENRCCI
        df_cost = pd.DataFrame(cost_breakdown)
        if not df_cost.empty:
            pie_fig = px.pie(
                df_cost,
                names="LID Type",
                values="Cost (pre-adjustment)",
                title="Cost Breakdown by LID Type",
                hole=0.4,
                template="plotly_white"
            )
            pie_chart = dcc.Graph(figure=pie_fig)
        else:
            pie_chart = "No data available for cost pie chart."
        table = dash_table.DataTable(
            data=df_cost.to_dict("records"),
            columns=[{"name": col, "id": col} for col in df_cost.columns],
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left"}
        )
        total_div = html.Div([
            table,
            html.H3(f"Overall LID Cost: ${total_cost:,.2f}"),
            html.Hr(),
            pie_chart
        ])
        return f"Cost calculations based on file: {file_path}", total_div, total_cost
    return "", "", None

# 5I. Calculate Pond Cost Callback
@app.callback(
    [Output("pond-cost-output", "children"),
     Output("stored-pond-cost", "data")],
    Input("calc-pond-cost-btn", "n_clicks"),
    State("pond-cost-depth", "value"),
    State("pond-cost-area", "value")
)
def calculate_pond_cost(n_clicks, depth, area):
    if n_clicks > 0:
        if depth is None or area is None:
            return "Please enter both pond depth and pond area.", None
        depth_ft = depth
        pond_area_ft2 = area * 43560
        bmp_area = pond_area_ft2 
        if bmp_area == 0:
            bmp_cost_pre = 0
        else:
            bmp_cost_pre = bmp_area * 4.6378 + 10052
        bmp_cost = bmp_cost_pre * 0.907 * 1.4208
        return f"Pond Cost: ${bmp_cost:,.2f}", bmp_cost
    return "", None

# 5J. Define Pond & Run SWAT Outflow Callback
@app.callback(
    Output("swat-pond-output", "children"),
    [Input("run-swat-pond-btn", "n_clicks")],
    [State("pond-depth", "value"),
     State("pond-area", "value"),
     State("swat-dir", "value"),
     State("stored-updated-total-flow", "data")]
)
def run_swat_pond(n_clicks, depth, area, swat_dir, updated_total_flow):
    if n_clicks > 0:
        if depth is None or area is None or not swat_dir:
            return "Please enter pond depth, pond area, and SWAT working directory."
        if updated_total_flow is None:
            return "Updated total flow is not available. Please run LID Simulation first."
      
        update_flow = (290000 - depth * 3.28084 * area * 4046.86) / 86400
        result = SWAT(swat_dir, updated_total_flow)
        
        # Check for SWAT errors
        if isinstance(result, str):
            return f"SWAT simulation error: {result}"
        
        peak_flow_ds = update_flow + result
        total_flow_vds = peak_flow_ds * 86400

        peak_reduction = ((peak_flow_ds - 3.7953) / 3.7953) * 100
        flow_reduction = ((total_flow_vds - 327912.32) / 327912.32) * 100

        text_output = html.Div([
            html.H3("SWAT Simulation Results:"),
            html.P(f"Downstream Peak Flow (cms): {peak_flow_ds:0.2f}"),
            html.P(f"Downstream Total Flow Volume (m³): {total_flow_vds:0.2f}"),
            html.P(f"Peak Flow Reduction (%): {peak_reduction:0.2f}"),
            html.P(f"Total Flow Volume Reduction (%): {flow_reduction:0.2f}")
        ])
        
        reduction_data = {
            "Reduction Type": ["Peak Flow Reduction (%)", "Total Flow Volume Reduction (%)"],
            "Reduction (%)": [peak_reduction, flow_reduction]
        }
        fig_reduction = px.bar(reduction_data,
                               x="Reduction Type",
                               y="Reduction (%)",
                               title="Flow Reduction Percentages",
                               text="Reduction (%)",
                               template="plotly_white")
        fig_reduction.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        
        graph_output = dcc.Graph(figure=fig_reduction)
        
        results_content = html.Div([
            text_output,
            html.Br(),
            graph_output
        ])
        
        return results_content
    return ""

# 5K. Total Cost Callback (LID Cost + Pond Cost)
@app.callback(
    Output("total-cost-output", "children"),
    Input("total-cost-btn", "n_clicks"),
    [State("stored-lid-cost", "data"), State("stored-pond-cost", "data")]
)
def calculate_total_cost(n_clicks, lid_cost, pond_cost):
    if n_clicks:
        if lid_cost is None or pond_cost is None:
            return "Please calculate both LID cost and Pond cost first."
        total_cost = lid_cost + pond_cost
        
        # Prepare data for the pie chart
        cost_data = {
            "Cost Type": ["LID Cost", "Pond Cost"],
            "Cost": [lid_cost, pond_cost]
        }
        # Create the pie chart; now showing percent values
        fig = px.pie(
            cost_data,
            names="Cost Type",
            values="Cost",
            title="Cost Breakdown",
            hole=0.4,
            template="plotly_white"
        )
        # Show percent values instead of raw values
        fig.update_traces(texttemplate='%{label}: %{percent:.2%}', textposition='inside')

        return html.Div([
            html.H3("Total Cost Calculation:"),
            html.P(f"LID Cost: ${lid_cost:,.2f}"),
            html.P(f"Pond Cost: ${pond_cost:,.2f}"),
            html.H3(f"Combined Total Cost: ${total_cost:,.2f}"),
            dcc.Graph(figure=fig)
        ])
    return "Click the button to calculate total cost."


###############################################################################
# 6. RUN THE APP
###############################################################################
if __name__ == "__main__":
    app.run_server(debug=True, port=8054)


# In[ ]:
server = app.server




