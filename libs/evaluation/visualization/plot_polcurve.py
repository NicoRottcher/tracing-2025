import datetime as dt
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

import evaluation.utils.db as db
from evaluation.visualization import plot


class exp_set_polcurve:
    def __init__(self,
                 debug=False,
                 by=None,
                 params=None,
                 name_user=None,
                 name_setup_sfc=None,
                 date=None,
                 id_ML=None,
                 data_add_cond=None,
                 get_exp_ec_data=True,
                 get_exp_compression_data=True,
                 cmaps=None,
                 cmap_min=0.3,
                 cmap_max=0.9,
                 cmap_group=None,
                 gradient_axis='id_exp_sfc',
                 add_colormaps=True,
                 ):
        self.exp_ec_polcurve = None
        self.data_ec_polcurve = None
        self.data_ec_polcurve_baddata = None
        self.exp_ec = None
        self.data_ec = None
        self.exp_compression = None
        self.data_compression = None

        self.j_geo_col = None  # "j__mA_cm2geo_fc_bottom_PTL"
        self.A_geo_col = None  # "j__mA_cm2geo_fc_bottom_PTL"

        self.axs = None
        self.ax_EIS = None
        self.ax_HFR_vs_j = None
        self.ax_HFR_vs_j_twinx = None
        self.ax_IV_vs_t_potential = None
        self.ax_tafel = None
        self.ax_IV_vs_t_normalized_potential = None
        self.ax_drift_potential = None
        self.ax_compression_zpos = None
        self.ax_compression_force = None
        self.debug = debug

        plot.manual_col_to_axis_label[
            'delta_tafel_fit_residuals__mV'] = '$\Delta E_\mathrm{tafel\ fit\ residuals}$ / mV'
        plot.manual_col_to_axis_label[
            'E_WE_uncompensated_step_normalized__VvsRHE'] = '$E_\mathrm{uncompensated,\ step\ norm.}$ / V'
        plot.manual_col_to_axis_label[
            'E_WE_uncompensated_drift__mV_min'] = '$E_\mathrm{uncompensated,\ drift}$ / mV min$^{-1}$'

        self.get_exp_ec_polcurve(
                            by=by,
                            params=params,
                            name_user=name_user,
                            name_setup_sfc=name_setup_sfc,
                            date=date,
                            id_ML=id_ML,
                            data_add_cond=data_add_cond,
                            )
        if get_exp_ec_data:
            self.get_exp_ec_data()
            if get_exp_compression_data:
                self.get_exp_compression_data()

        if add_colormaps:
            self.add_colormaps(
                cmaps=cmaps,
                cmap_min=cmap_min,
                cmap_max=cmap_max,
                cmap_group=cmap_group,
                gradient_axis=gradient_axis,
            )

    def get_exp_ec_polcurve(self,
                            by=None,
                            params=None,
                            name_user=None,
                            name_setup_sfc=None,
                            date=None,
                            id_ML=None,
                            data_add_cond=None,
                            ):
        if self.exp_ec_polcurve is not None:
            print('overwrite self.exp_ec_polcurve')
        if type(by) == str:
            if '%s' in by:
                if type(params) != list:
                    raise Exception('params must be list')
                if len(by.split('%s')) - 1 != len(params):
                    raise Exception('params must contain the same number of params in sql statement')
        elif by is None:
            id_ML = [id_ML] if type(id_ML) != list else id_ML
            id_ML_sql_str = ', '.join(['%s' for val in id_ML])
            by = f"""SELECT *
                        FROM exp_ec_polcurve_expanded
                        WHERE name_user= %s
                             AND name_setup_sfc = %s
                             AND DATE(t_start__timestamp) = %s
                             AND id_ML IN ({id_ML_sql_str})
                             ;"""
            params = [name_user, name_setup_sfc, date, *id_ML]
        self.exp_ec_polcurve = db.get_exp(by, params=params, debug=False).sort_values(by='t_start__timestamp')

        self.get_j_geo_col_from_exp_ec_polcurve()

        self.data_ec_polcurve = db.get_data(self.exp_ec_polcurve,
                                            'data_ec_polcurve_analysis',
                                            add_cond=data_add_cond)
        self.data_ec_polcurve_baddata = self.data_ec_polcurve.loc[lambda row: row.gooddata == 0, :]

        self.data_ec_polcurve.loc[:, 'E_WE_uncompensated_drift__mV_min'] \
            = self.data_ec_polcurve.E_WE_uncompensated_drift__V_s * 1000 * 60

        self.calc_geo_columns()

    def get_exp_ec_data(self, ):
        ids_exp_ec = pd.concat([self.data_ec_polcurve.id_exp_sfc_ghold.rename('id_exp_sfc'),
                                self.data_ec_polcurve.id_exp_sfc_geis.rename('id_exp_sfc')]) \
            .reset_index().set_index('id_exp_sfc').sort_index()
        # display(ids_exp_ec)
        self.exp_ec = db.get_exp(by=ids_exp_ec.loc[:, []],
                                 name_table='exp_ec_expanded') \
            .join(ids_exp_ec, on='id_exp_sfc')
        if self.A_geo_col not in self.exp_ec.columns:
            self.exp_ec = self.exp_ec.join(self.exp_ec_polcurve[self.A_geo_col], on='id_exp_ec_polcurve')

        self.data_ec = db.get_data_ec(self.exp_ec,
                                      add_data_eis=True, )

        self.data_ec = self.data_ec.join(ids_exp_ec.id_exp_ec_polcurve, on='id_exp_sfc')
        self.exp_ec = self.exp_ec.join(
            self.data_ec.groupby('id_exp_ec_polcurve').Timestamp.min().rename('t_start_exp_ec_polcurve__timestamp'),
            on='id_exp_ec_polcurve')
        self.data_ec_polcurve = self.data_ec_polcurve.join(
            self.data_ec.groupby('id_exp_sfc').Timestamp.min().rename(
                't_start_exp_ec_polcurve_step__timestamp'), on='id_exp_sfc_ghold')
        self.exp_ec.loc[:, 't_start_exp_ec_polcurve_step__timestamp'] = pd.concat([
            self.data_ec_polcurve.reset_index().set_index('id_exp_sfc_ghold').t_start_exp_ec_polcurve_step__timestamp,
            self.data_ec_polcurve.reset_index().set_index('id_exp_sfc_geis').t_start_exp_ec_polcurve_step__timestamp
        ]).sort_index()

        self.data_ec.loc[:, "t_synchronized__s"] = (
                pd.to_datetime(self.data_ec.Timestamp) - self.exp_ec.t_start_exp_ec_polcurve__timestamp
        ).dt.total_seconds()
        self.data_ec.loc[:, "t_synchronized_step__s"] = (
                pd.to_datetime(self.data_ec.Timestamp) - self.exp_ec.t_start_exp_ec_polcurve_step__timestamp
        # self.data_ec.groupby('id_exp_sfc').Timestamp.min()
        ).dt.total_seconds()

    def get_exp_compression_data(self, add_cond='force__N > 0'):
        if self.exp_ec is None:
            self.get_exp_ec_data()

        self.exp_compression = db.get_exp(by=self.exp_ec.loc[:, []],
                                          name_table='exp_compression_expanded')\
                                 .join(self.exp_ec.ec_name_technique, on='id_exp_sfc')

        self.data_compression = db.get_data(self.exp_compression,
                                            name_table='data_compression',
                                            add_cond=add_cond)

        self.data_compression.loc[:, "t_synchronized__s"] = (
                pd.to_datetime(self.data_compression.Timestamp) - self.exp_ec.t_start_exp_ec_polcurve__timestamp
        ).dt.total_seconds()
        self.data_compression.loc[:, "t_synchronized_step__s"] = (
                pd.to_datetime(self.data_compression.Timestamp) - self.exp_ec.t_start_exp_ec_polcurve_step__timestamp
        ).dt.total_seconds()

    def get_j_geo_col_from_exp_ec_polcurve(self,
                                           from_nth=0):
        if self.exp_ec_polcurve is None:
            raise Exception('None self.exp_ec_polcurve yet initiated. First run get_exp_ec_polcurve.')
        elif len(self.exp_ec_polcurve.chosen_j_geo_col.unique()) > 1:
            print('Different chosen_j_geo_col. Correct each experiment seperately and derive ..._geo_active_chosen')
            display(self.exp_ec_polcurve.chosen_j_geo_col)
            self.exp_ec_polcurve \
                = self.exp_ec_polcurve.join(plot.geo_columns.set_index('j_geo').A_geo.rename('chosen_A_geo_col'),
                                            on='chosen_j_geo_col') \
                .assign(
                A_geo_active_chosen__mm2=lambda exp: [row[row.chosen_A_geo_col] for index, row in exp.iterrows()])
            self.j_geo_col = 'j__mA_cm2geo_active_chosen'
        else:
            self.j_geo_col = self.exp_ec_polcurve.chosen_j_geo_col.iloc[from_nth]
        self.A_geo_col = plot.get_geo_column(name=self.j_geo_col, type_in='j_geo', type_out='A_geo')

    def calc_geo_columns(self, ):
        new_A_geo_cols = [row.A_geo for index, row in plot.geo_columns.iterrows() if
                          row.j_geo not in self.data_ec_polcurve.columns and row.A_geo in self.exp_ec_polcurve.columns]
        data_ec_polcurve_geos = self.data_ec_polcurve.join(self.exp_ec_polcurve.loc[:, new_A_geo_cols],
                                                           on='id_exp_ec_polcurve')
        self.data_ec_polcurve \
            = self.data_ec_polcurve.join(self.exp_ec_polcurve[self.A_geo_col], on='id_exp_ec_polcurve') \
            .assign(HFR__Ohm=lambda data: data.R_u__ohm,
                    HFR__mOhm_cm2=lambda data: data.HFR__Ohm * 1000 * data[self.A_geo_col] / 100,
                    **{plot.get_geo_column(new_A_geo_col, type_in='A_geo', type_out='j_geo'): lambda
                        data: data.I__A * 1000 / (data_ec_polcurve_geos[new_A_geo_col] / 100) for new_A_geo_col in
                       new_A_geo_cols}
                    )

    def calc_step_normalization(self, ):
        self.data_ec = self.data_ec \
            .join(self.data_ec_polcurve.set_index('id_exp_sfc_ghold') \
                  .loc[:, ['E_WE_uncompensated__VvsRHE', self.j_geo_col]] \
                  .rename(columns={'E_WE_uncompensated__VvsRHE': 'E_WE_uncompensated_step_mean__VvsRHE',
                                   self.j_geo_col: self.j_geo_col + '_step_mean'}),
                  on='id_exp_sfc',
                  ) \
            .assign(**{
            'E_WE_uncompensated_step_normalized__VvsRHE': lambda
                x: x.E_WE_uncompensated__VvsRHE - x.E_WE_uncompensated_step_mean__VvsRHE,
            self.j_geo_col + '_normalized': lambda x: x[self.j_geo_col] - x[self.j_geo_col + '_step_mean']
        }
                    )

    def data_ec_polcurve_tafel_fit(self, ):
        return self.data_ec_polcurve.join(
            self.exp_ec_polcurve[['tafel_fit_left_limit__j_geo', 'tafel_fit_right_limit__j_geo']],
            on='id_exp_ec_polcurve').loc[lambda row: ((row[self.j_geo_col] >= row.tafel_fit_left_limit__j_geo)
                                                      & (row[
                                                             self.j_geo_col] <= row.tafel_fit_right_limit__j_geo))
        ]

    def add_colormaps(self,
                      cmaps=None,
                      cmap_min=0.3,
                      cmap_max=0.9,
                      cmap_group=None,
                      gradient_axis='id_exp_sfc',
                      ):
        if cmaps is None:
            cmaps = ['Blues', 'Oranges', 'Greens', 'Reds', 'Greys', 'Purples',
                     'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                     'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

        # special case, each SGEIS hold gets a new color
        if gradient_axis == 'id_exp_sfc':

            if len(cmaps) < len(self.exp_ec_polcurve.index):
                warnings.warn('There are not enough cmaps given. Cmaps are used more tha once. '
                              'Give more cmaps to have unique colors '
                              'or adjust cmap_group/gradient_axis to control which experiments have the same color')
                while len(cmaps) < len(self.exp_ec_polcurve.index):
                    cmaps += cmaps

            self.exp_ec_polcurve.loc[:, 'cmap_choice'] = cmaps[:len(self.exp_ec_polcurve.index)]

            # half of cmap color as main color for experiment
            self.exp_ec_polcurve.dataset \
                .add_column('color',
                            values=self.exp_ec_polcurve.cmap_choice.tolist(),
                            rowindexers=[(val,) for val in self.exp_ec_polcurve.index.values],
                            cmap_min=0.5, cmap_max=0.5) \
                .return_dataset()

            cmap_pre = self.data_ec_polcurve.groupby('id_exp_ec_polcurve') \
                .apply(lambda row: pd.Series(plot.get_colormap(
                        cmap_name=self.exp_ec_polcurve.loc[row.index.get_level_values('id_exp_ec_polcurve')[0],
                                                           'cmap_choice'],
                        count_rows=len(row.index),
                        cmap_min=cmap_min,
                        cmap_max=cmap_max,
                    ),
                    index=row.index.get_level_values('id_data_ec_polcurve'),
                    name='cmap',
                )
               )
            if isinstance(cmap_pre,
                          pd.DataFrame):  # groupby-apply puts id_data_ec_polcurve into columns not index if all groups have same length, creating a dataframe, this cannot be handled afterward
                cmap_pre = cmap_pre.stack('id_data_ec_polcurve')
            display(cmap_pre) if self.debug else ''

            self.data_ec_polcurve.loc[:, 'cmap'] \
                = cmap_pre

        else:
            gradient_axis = [gradient_axis] if isinstance(gradient_axis,
                                                          str) else gradient_axis  # ensure gradient_axis is list
            # add index columns to regular columns in case these are selected
            for i, col in enumerate(gradient_axis):
                if col in self.exp_ec_polcurve.index.names:
                    col_from_index = col + '_from_index'
                    self.exp_ec_polcurve.loc[:, col_from_index] = self.exp_ec_polcurve.index.get_level_values(col)
                    gradient_axis[i] = col_from_index
            groups_val = self.exp_ec_polcurve[gradient_axis].drop_duplicates().values

            if len(cmaps) < len(groups_val):
                raise Exception('There are more cmap groups than cmaps. Please give more cmaps.')

            self.exp_ec_polcurve = self.exp_ec_polcurve.dataset \
                .add_column('color',
                            values=cmaps[:len(groups_val)],  # self.exp_ec_polcurve.cmap_choice.tolist(),
                            rowindexers=[(self.exp_ec_polcurve[gradient_axis] == group_val).all(axis=1) for group_val in
                                         groups_val],
                            # [(val,) for val in self.exp_ec_polcurve.index.values],
                            cmap_min=cmap_min,
                            cmap_max=cmap_max,
                            cmap_group=cmap_group) \
                .return_dataset()
            self.data_ec_polcurve = self.data_ec_polcurve.join(self.exp_ec_polcurve.color.rename('cmap'),
                                                               on='id_exp_ec_polcurve',
                                                               lsuffix='_old')
        # Transfer to exp_ec, exp_compression if needed
        if self.exp_ec is not None:
            self.exp_ec.loc[:, 'color'] = pd.concat([
                self.data_ec_polcurve.reset_index().set_index('id_exp_sfc_ghold').cmap,
                self.data_ec_polcurve.reset_index().set_index('id_exp_sfc_geis').cmap
            ])
            if self.exp_compression is not None:
                self.exp_compression = self.exp_compression.join(self.exp_ec.color, lsuffix='_old')

    ## plot templates
    def plot_scatter_line(
            self,
            exp,
            data,
            y_col,
            x_col,
            ax_plot_object_name,
            linestyle="-",
            marker="s",
            zorder=1,
            s=4,
            c='cmap',
            **kwargs,
    ):
        return (exp.dataset
                .plot(
                    x_col=x_col,
                    y_col=y_col,
                    data=data,
                    linestyle=linestyle,
                    zorder=zorder,
                    ax_plot_object_name=ax_plot_object_name + '_line',
                    **kwargs,
                )
                .scatter(
                    x_col=x_col,
                    y_col=y_col,
                    data=data,
                    zorder=zorder + 1,
                    s=s,
                    marker=marker,
                    label="",
                    c=c,
                    ax_plot_object_name=ax_plot_object_name + '_points',
                    plt_kwargs_extend_ignore_cols=['color'],
                    **{k: v for k, v in kwargs.items() if k not in ['label', ]},
                )
                .return_dataset()
                )

    # specific plots
    def plot_nyquist(self, **kwargs):
        self.exp_ec = (
            self.exp_ec.dataset
                .plot(
                    rowindexer=lambda row: row.ec_name_technique.isin(['exp_ec_geis', 'exp_ec_peis']),
                    x_col="Z_real__ohm",
                    y_col="minusZ_img__ohm",  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                    data=self.data_ec,
                    ax=self.ax_EIS,
                    # alpha=0.2
                    **kwargs,
                )
                .return_dataset()
        )

    def plot_E_vs_t(self,
                    **kwargs):
        self.exp_ec = (
            self.exp_ec.dataset
                .plot(
                    # rowindexer=lambda row: row.ec_name_technique.isin(['exp_ec_ghold', 'exp_ec_phold']),
                    x_col="t_synchronized__s",
                    y_col="E_WE_uncompensated__VvsRHE",
                    data=self.data_ec,
                    ax=self.ax_IV_vs_t_potential,
                    **kwargs,
                )
                .return_dataset()
        )

    def plot_tafel_uncompensated(self,
                                 alpha=0.2,
                                 linestyle="-",
                                 marker="s",
                                 markersize=2,
                                 zorder=0,
                                 **kwargs):
        self.exp_ec_polcurve = (
            self.exp_ec_polcurve.dataset
                .plot(
                    y_col="E_WE_uncompensated__VvsRHE",
                    x_col=self.j_geo_col,  # 'E_Signal__VvsRE',#'j__mA_cm2geo_fc_top_cell_Aideal',#'',
                    data=self.data_ec_polcurve,
                    ax=self.ax_tafel,
                    marker=marker,
                    markersize=markersize,
                    linestyle=linestyle,
                    label="",
                    axlabel_auto=False,
                    # color="gray",
                    alpha=alpha,
                    zorder=zorder,
                    ax_plot_object_name='tafel_E_raw_linepoints',
                    **kwargs,
                )
                .return_dataset()
        )

    def plot_tafel(self,
                   label='',
                   tafel_fit_method="scipy.odr",  # 'scipy.optimize.curve_fit',#
                   **kwargs):
        self.exp_ec_polcurve = self.plot_scatter_line(
            exp=self.exp_ec_polcurve,
            data=self.data_ec_polcurve,
            y_col="E_WE__VvsRHE",
            x_col=self.j_geo_col,
            ax_plot_object_name='tafel_E_HFR',
            ax=self.ax_tafel,
            label=label,
            **kwargs, )

        tafel_fit_params = dict(
            model=plot.tafel_func,
            x_col=self.j_geo_col,  # 'j__mA_cm2geo_fc_bottom_PTL',j__mA_mg_total_geo_fc_bottom_PTL
            y_col="E_WE__VvsRHE",
            # yerr_col='count_ratio_std',
            beta0=[1, 0],
            ifixb=[1, 1],
            method=tafel_fit_method,
            # odr has poorer performance when high hysteresis in for and back sweep
            linestyle=(0, (5, 7)),  # '--',
            # color="tab:orange",
            ax=self.ax_tafel,
            axlabel_auto=False,
            label_fit_overwrite=True,
            label_fit_style=plot.create_fit_label(
                description=False,
                params=True,
                params_selection=['m'],
                rsquared=False,
                err_considered=False,
            )
        )

        self.exp_ec_polcurve = self.exp_ec_polcurve.dataset.fit(
            **tafel_fit_params,
            data=self.data_ec_polcurve_tafel_fit(),
            ax_plot_object_name='tafel_E_HFR_fit',
        ).return_dataset()

    def plot_HFR_vs_j(self,
                      **kwargs):
        self.exp_ec_polcurve = self.plot_scatter_line(
            exp=self.exp_ec_polcurve,
            data=self.data_ec_polcurve,
            y_col="R_u__ohm",
            x_col=self.j_geo_col,
            ax_plot_object_name='tafel_HFR_vs_j',
            ax=self.ax_HFR_vs_j,
            **kwargs, )
        self.exp_ec_polcurve = self.plot_scatter_line(
            exp=self.exp_ec_polcurve,
            data=self.data_ec_polcurve,
            y_col="HFR__mOhm_cm2",
            x_col=self.j_geo_col,
            ax_plot_object_name='tafel_HFR_geo_vs_j',
            ax=self.ax_HFR_vs_j_twinx,
            **kwargs, )

    def plot_compression_zpos(self,
                              **kwargs):
        if self.exp_compression is None:
            self.get_exp_compression_data()
        self.exp_compression = (
            self.exp_compression.dataset
                .plot(
                    x_col="t_synchronized__s",
                    y_col="linaxis_z__mm",
                    data=self.data_compression,
                    ax=self.ax_compression_zpos,
                    **kwargs,
                )
                .return_dataset()
        )

    def plot_compression_force(self,
                               **kwargs):
        if self.exp_compression is None:
            self.get_exp_compression_data()
        self.exp_compression = (
            self.exp_compression.dataset
                .plot(
                    x_col="t_synchronized__s",
                    y_col="force__N",
                    data=self.data_compression,
                    ax=self.ax_compression_force,
                    **kwargs,
                )
                .return_dataset()
        )

    def plot_E_stepnorm_vs_t(self,
                             **kwargs):
        if 'E_WE_uncompensated_step_normalized__VvsRHE' not in self.data_ec.columns:
            self.calc_step_normalization()
        self.exp_compression = (
            self.exp_compression.dataset
                .plot(
                    rowindexer=lambda row: row.ec_name_technique.isin(['exp_ec_ghold', 'exp_ec_phold']),
                    x_col="t_synchronized__s",
                    y_col="E_WE_uncompensated_step_normalized__VvsRHE",
                    data=self.data_ec,
                    **kwargs,
                )
                .return_dataset()
        )

    def plot_drift_potential(self,
                             **kwargs):
        self.exp_ec_polcurve = self.plot_scatter_line(
            exp=self.exp_ec_polcurve,
            data=self.data_ec_polcurve,
            y_col="E_WE_uncompensated_drift__mV_min",
            x_col=self.j_geo_col,
            ax_plot_object_name='dritf_potential_vs_j',
            **kwargs, )

    def plot_overview(self,
                      plot_nyquist=True,
                      plot_HFR_vs_j=True,
                      plot_E_vs_t=True,
                      plot_tafel=True,
                      plot_tafel_uncompensated=True,
                      plot_E_stepnorm_vs_t=True,
                      plot_drift_potential=True,
                      plot_compression=True,
                      print_plot_info=True,
                      style="doubleColumn",  # "singleColumn",
                      interactive=True,
                      increase_fig_height=3.5,
                      add_margins_and_figsize=None,
                      add_margins_between_subplots=None,
                      add_params=None,
                      scale_fontsize=0.8,
                      export_name='polcurve_overview',
                      **kwargs_get_style,
                      ):

        if add_params is None:
            add_params = {"figure.dpi": 150}
        if add_margins_between_subplots is None:
            add_margins_between_subplots = {"hspace": 6, "wspace": 4}
        if add_margins_and_figsize is None:
            add_margins_and_figsize = {"right": 0.8, "left": 0.5, "bottom": 0}

        with plt.rc_context(plot.get_style(style=style,  # "singleColumn",
                                           interactive=interactive,
                                           increase_fig_height=increase_fig_height,
                                           add_margins_and_figsize=add_margins_and_figsize,
                                           add_margins_between_subplots=add_margins_between_subplots,
                                           add_params=add_params,
                                           scale_fontsize=scale_fontsize,
                                           **kwargs_get_style)):
            plot_storage = plot.PlotDataStorage(
                export_name, overwrite_existing=True
            )
            fig = plt.figure()
            # ax_EIS = fig.add_subplot(111)
            self.axs = []

            gs = gridspec.GridSpec(5, 2)  # n_plots, 1)

            if plot_nyquist:
                print('plot_nyquist') if print_plot_info else ''
                self.ax_EIS = fig.add_subplot(gs[0, 0])  # n_plot, 0])
                self.ax_EIS.axis('equal')  # same magnitude for x and y axis - important for Nyquist plots
                self.axs += [self.ax_EIS]
                self.plot_nyquist()

            if plot_HFR_vs_j:
                print('plot_HFR_vs_j') if print_plot_info else ''
                self.ax_HFR_vs_j = fig.add_subplot(gs[0, 1])
                self.ax_HFR_vs_j_twinx = self.ax_HFR_vs_j.twinx()
                self.ax_HFR_vs_j.set_xscale("log")
                self.axs += [self.ax_HFR_vs_j]
                self.plot_HFR_vs_j()

            if plot_E_vs_t:
                print('plot_E_vs_t') if print_plot_info else ''
                self.ax_IV_vs_t_potential = fig.add_subplot(gs[1, 0])
                self.axs += [self.ax_IV_vs_t_potential]
                self.plot_E_vs_t()

            if plot_tafel:
                print('plot_tafel') if print_plot_info else ''
                self.ax_tafel = fig.add_subplot(gs[1, 1], sharex=self.ax_HFR_vs_j)
                self.ax_tafel.set_xscale('log')
                self.axs += [self.ax_tafel]
                if plot_tafel_uncompensated:
                    print('plot_tafel_uncompensated') if print_plot_info else ''
                    self.plot_tafel_uncompensated()
                self.plot_tafel()
                self.ax_tafel.legend()

            if plot_E_stepnorm_vs_t:
                print('plot_E_stepnorm_vs_t') if print_plot_info else ''
                self.ax_IV_vs_t_normalized_potential = fig.add_subplot(gs[2, 0])  # n_plot, 0])
                self.axs += [self.ax_IV_vs_t_normalized_potential]
                self.plot_E_stepnorm_vs_t()
                self.ax_IV_vs_t_normalized_potential.set_ylim([-0.002, 0.002])

            if plot_drift_potential:
                print('plot_drift_potential') if print_plot_info else ''
                self.ax_drift_potential = fig.add_subplot(gs[2, 1], sharex=self.ax_tafel)
                self.ax_drift_potential.set_xscale("log")
                self.axs += [self.ax_drift_potential]
                self.plot_drift_potential()

                if 'label' not in self.exp_ec_polcurve.columns:
                    self.exp_ec_polcurve = self.exp_ec_polcurve.assign(
                        label=lambda row: 'id_ML: ' + row.id_ML.astype(str))
                self.ax_drift_potential.legend(loc='upper left', bbox_to_anchor=(0,-0.3))

            if plot_compression:
                print('plot_compression: zpos') if print_plot_info else ''
                self.ax_compression_zpos = fig.add_subplot(gs[3, 0], sharex=self.ax_IV_vs_t_potential)
                self.axs += [self.ax_compression_zpos]
                self.plot_compression_zpos()

                print('plot_compression: force') if print_plot_info else ''
                self.ax_compression_force = fig.add_subplot(gs[4, 0], sharex=self.ax_IV_vs_t_potential)
                self.axs += [self.ax_compression_force]
                self.plot_compression_force()

            plot_storage.export(fig, plot_format='pdf')
            plt.show()


def plot_template(kwargs_add_exp_ec_polcurve,
                  kwargs_plot=None,
                  ):
    if kwargs_plot is None:
        kwargs_plot = {}
    set_polcurves = exp_set_polcurve(**kwargs_add_exp_ec_polcurve)
    set_polcurves.plot_overview(**kwargs_plot)


