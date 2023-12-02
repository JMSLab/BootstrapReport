from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from src.BootstrapReport import ObjectOfInterest

app_ui = ui.page_fluid(
    ui.panel_title("Analyze a Report"),

    ui.layout_sidebar(

      ui.panel_sidebar(
        ui.input_numeric("est", "Estimate", 0),
        ui.input_numeric("se", "Standard error", 1),
        ui.input_file("repcsv", "Choose CSV File", accept=[".csv"], multiple=False),
        ui.input_checkbox_group("checkboxes", "Summary Stats",
        choices=["Row Count", "Column Count", "Column Names"],
        selected=["Row Count", "Column Count", "Column Names"],),
        ui.output_table("summary"),
      ),

      ui.panel_main(
        ui.output_plot("pp_plot"),
        ui.output_table("stats"),
      ),
    ),
)


def server(input, output, session):
    @reactive.Calc
    def parsed_file() -> object:
        file: list[FileInfo] | None = input.repcsv()
        if file is None:
            return pd.DataFrame()
        return pd.read_csv(  # pyright: ignore[reportUnknownMemberType]
            file[0]["datapath"]
        )
    
    @reactive.Calc
    def eval_ooi() -> object:
        df_rep = parsed_file()
        rep = df_rep['replicate_value'].values
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
        info_df = pd.DataFrame(
            {
                "Row Count": [row_count],
                "Column Count": [column_count],
                "Column Names": [column_names],
            }
        )

        # input.checkboxes() is a list of strings; subset the columns based on the selected
        # checkboxes
        return info_df.loc[:, input.checkboxes()]

    @render.plot(alt="pp plot")
    def pp_plot() -> object:
        ooi = eval_ooi()
        fig = ooi.pp_plot()
        return fig
    
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
