from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo
import numpy as np, pandas as pd
from pyreadr import read_r
from scipy.io import loadmat
from BootstrapReport import ObjectOfInterest

app_ui = ui.page_fixed(
    ui.div(style = "height:30px"),
    ui.panel_title("BootstrapReport"),

    ui.layout_sidebar(

      ui.panel_sidebar(
        ui.input_numeric("est", "Estimate", 0),
        ui.input_numeric("se", "Standard error", 1),
        ui.input_file("repcsv", "Choose data file", accept=[".csv", ".mat", ".dta", ".rds", ".xls", ".xlsx"], multiple=False),
        ui.output_table("summary"),
        ui.input_text("repname", "Name of column with replicates"),
        width = 5.2,
      ),

      ui.panel_main(
        ui.output_table("stats"),
        ui.output_plot("pp_plot"),
        width = 6.8,
      ),
      height = 900
    ),

    ui.tags.h3("Info"),
    ui.tags.div(
        "More information about the computation process can be found at ",
        ui.tags.a("https://github.com/JMSLab/BootstrapReport", href = "https://github.com/JMSLab/BootstrapReport",),
    ),
    ui.div(style = "height:30px"),
)

def server(input, output, session):
    @reactive.Calc
    def parsed_file() -> object:
        file: list[FileInfo] | None = input.repcsv()
        if file is None:
            return pd.DataFrame()
        elif file[0]["datapath"][-3:] == 'csv':
            return pd.read_csv(file[0]["datapath"])
        elif file[0]["datapath"][-3:] == 'mat':
            raw_m, out_m = loadmat(file[0]["datapath"]), dict()
            nonvars = ['__header__', '__version__', '__globals__']
            for key in raw_m.keys() - nonvars:
                num_columns = raw_m[key].shape[1]
                if num_columns > 1:
                    for i in range(1, num_columns + 1):
                        out_m[f'{key}_{i}'] = raw_m[key][:, i - 1]
                else:
                    out_m[key] = raw_m[key][:, 0]
            return pd.DataFrame(out_m)
        elif file[0]["datapath"][-3:] == 'dta':
            return pd.read_stata(file[0]["datapath"])
        elif file[0]["datapath"][-3:].lower() == 'rds':
            raw_r = read_r(file[0]["datapath"])
            return raw_r[None]
        elif file[0]["datapath"][-3:] == 'xls' or file[0]["datapath"][-4:] == 'xlsx':
            return pd.read_excel(file[0]["datapath"])
    
    @reactive.Calc
    def eval_ooi() -> object:
        df_rep = parsed_file()
        if not input.repname() == "":
            rep = df_rep.loc[:, input.repname()].values
        else:
            rep = df_rep.iloc[:, 0].values
        rep = rep[~np.isnan(rep)]
        ooi = ObjectOfInterest(input.est(), input.se(), rep)
        return ooi

    @output
    @render.table
    def summary() -> object:
        df_rep = parsed_file()
        if df_rep.empty:
            return pd.DataFrame()

        row_count = df_rep.shape[0]
        column_count = df_rep.shape[1]
        names = df_rep.columns.tolist()
        column_names = ", ".join(str(name) for name in names)
        info_df = pd.DataFrame({"Row Count": [row_count],
                                "Column Count": [column_count],
                                "Column Names": [column_names],})
        return info_df

    @output
    @render.plot(alt="pp plot", width = 670, height = 670)
    def pp_plot() -> object:
        ooi = eval_ooi()
        fig = ooi.pp_plot(fontsize = 13.5, legend_fontsize = 14.5, labelsize = 18)
        return fig
    
    @output
    @render.table
    def stats() -> object:
        ooi = eval_ooi()
        ooi.get_sk_min()
        df_stats = pd.DataFrame(columns = ['SK distance', 'Minimum SK distance', 'SK minimizing mean', 'SK minimizing SD'])
        df_stats.at[0, 'SK distance'] = ooi.sk_dist
        df_stats.at[0, 'Minimum SK distance'] = ooi.skmin
        df_stats.at[0, 'SK minimizing mean'] = ooi.skmin_mean
        df_stats.at[0, 'SK minimizing SD'] = ooi.skmin_sd
        return df_stats

app = App(app_ui, server)
