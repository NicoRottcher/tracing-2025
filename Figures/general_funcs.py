import pandas as pd
def background_correction(icpms_x_col,
                          icpms_y_col,
                          data_icpms_overlay,
                          y_col_background='_background',
                          y_col_background_corrected='_background_corrected',
                          debug=False,
                          ):
    icpms_y_col_background_correction = icpms_y_col
    icpms_y_col_background = icpms_y_col_background_correction + y_col_background
    icpms_y_col_background_corrected =  icpms_y_col_background_correction + y_col_background_corrected
    
    
    if icpms_y_col_background in data_icpms_overlay.columns:
        print('Already corrected, it is nor recalculated')
        return data_icpms_overlay, icpms_y_col_background, icpms_y_col_background_corrected
    
    # datapoints of ocp to be averaged  - assuming 1 datapoint per secon
    n_ocp_select = 20

    # y-values for baseline
    data_icpms_overlay_select_for_background = data_icpms_overlay.loc[((data_icpms_overlay.is_during_ocp)
                                                 ), 
                                                 [icpms_x_col, 
                                                                                icpms_y_col_background_correction, 
                                                                                'id_exp_sfc']]

    # y-values
    # Take the last n_ocp_select datapoints for each ocp technique
    data_icpms_overlay_background_ocp_data = data_icpms_overlay_select_for_background.groupby(data_icpms_overlay.index.names[:-1]+['id_exp_sfc']).tail(n=n_ocp_select)
    data_icpms_overlay_background_ocp = data_icpms_overlay_background_ocp_data.groupby(data_icpms_overlay.index.names[:-1]+['id_exp_sfc'])[icpms_y_col_background_correction].mean()#\

    
    # Special
    # For the second ocp after activation hold, take datapoints around a specific synchronized time: t_center_2__s (instead of values at start or end of ocp)
    t_center_2__s = 650
    data_icpms_overlay_background_2_ocp = \
        data_icpms_overlay_select_for_background.loc[((data_icpms_overlay_select_for_background.loc[:, icpms_x_col] < t_center_2__s+n_ocp_select/2)
                                                      & (data_icpms_overlay_select_for_background.loc[:, icpms_x_col] > t_center_2__s-n_ocp_select/2)), 
                                                     :].groupby(data_icpms_overlay.index.names[:-1]+['id_exp_sfc'])[icpms_y_col_background_correction].mean()

    # For the last ocp after polcurve, take datapoints around a specific synchronized time: t_center_3__s (instead of values at start or end of ocp)
    t_center_3__s = 3500
    data_icpms_overlay_background_3_ocp = \
        data_icpms_overlay_select_for_background.loc[((data_icpms_overlay_select_for_background.loc[:, icpms_x_col] < t_center_3__s+n_ocp_select/2)
                                                      & (data_icpms_overlay_select_for_background.loc[:, icpms_x_col] > t_center_3__s-n_ocp_select/2)), 
                                                     :].groupby(data_icpms_overlay.index.names[:-1]+['id_exp_sfc'])[icpms_y_col_background_correction].mean()


    # x-values for baseline
    data_icpms_overlay_ocp_first = data_icpms_overlay_select_for_background.groupby(data_icpms_overlay.index.names[:-1]+['id_exp_sfc'])[icpms_x_col].first()
    data_icpms_overlay_ocp_last = data_icpms_overlay_select_for_background.groupby(data_icpms_overlay.index.names[:-1]+['id_exp_sfc'])[icpms_x_col].last()
    
    only_activation = data_icpms_overlay_ocp_last.groupby(data_icpms_overlay.index.names[:-1]).count().max() <= 2
    if debug:
        display(data_icpms_overlay_ocp_last.groupby(data_icpms_overlay.index.names[:-1]).count())
    x_values = pd.concat([#activation
                          data_icpms_overlay_ocp_first.groupby(data_icpms_overlay.index.names[:-1]).nth(0),
                          data_icpms_overlay_ocp_last.groupby(data_icpms_overlay.index.names[:-1]).nth(0),
                          data_icpms_overlay_ocp_first.groupby(data_icpms_overlay.index.names[:-1]).nth(1),
                          data_icpms_overlay_ocp_last.groupby(data_icpms_overlay.index.names[:-1]).nth(1),
                          # polarization
                          data_icpms_overlay_ocp_last.groupby(data_icpms_overlay.index.names[:-1]).nth(2),
                          data_icpms_overlay_ocp_first.groupby(data_icpms_overlay.index.names[:-1]).nth(3),
                          data_icpms_overlay_ocp_last.groupby(data_icpms_overlay.index.names[:-1]).nth(3)
                         ]).sort_index()
    y_values = pd.concat([
                   # first baseline niveau end of ocp 0 (before activation)
                   data_icpms_overlay_background_ocp.groupby(data_icpms_overlay.index.names[:-1]).nth(0),
                   data_icpms_overlay_background_ocp.groupby(data_icpms_overlay.index.names[:-1]).nth(0),

                    # second baseline niveau: 
                    # for only activation around t_center_2__s
                    # for polarization curve: end of ocpd before polarization curve starts
                    data_icpms_overlay_background_2_ocp.groupby(data_icpms_overlay.index.names[:-1]).nth(0),
                    data_icpms_overlay_background_2_ocp.groupby(data_icpms_overlay.index.names[:-1]).nth(0),


                    data_icpms_overlay_background_ocp.groupby(data_icpms_overlay.index.names[:-1]).nth(2),
                    

                   # third baseline niveau: at specific synchronied time point as defined by t_center_3__s
                    data_icpms_overlay_background_3_ocp.groupby(data_icpms_overlay.index.names[:-1]).nth(0),
                    data_icpms_overlay_background_3_ocp.groupby(data_icpms_overlay.index.names[:-1]).nth(0),
                  ]).sort_index()
    if debug:
        display(x_values)
        display(y_values)
    data_icpms_overlay_background=pd.concat([x_values, y_values], axis=1)\
                                            .reset_index().reset_index()\
                                            .set_index(data_icpms_overlay.index.names[:-1]+[icpms_x_col])\
                                            .rename(columns={icpms_y_col_background_correction:icpms_y_col_background})



    data_icpms_overlay_background_joined = data_icpms_overlay.join(data_icpms_overlay_background, on=data_icpms_overlay.index.names[:-1]+[icpms_x_col])

    data_icpms_overlay.loc[:, icpms_y_col_background] =  data_icpms_overlay.join(data_icpms_overlay_background, 
                                                                                 on=data_icpms_overlay.index.names[:-1]+[icpms_x_col])\
                                        .groupby(data_icpms_overlay.index.names[:-1])\
                                        .apply(lambda group: pd.Series(group.set_index(icpms_x_col)[icpms_y_col_background].interpolate(method='index', ).to_numpy(),
                                                                       index=group.index))\
                                        .reset_index(level=list(range(0,len(data_icpms_overlay.index.names)-1)), drop=True)



    data_icpms_overlay.loc[:, icpms_y_col_background_corrected] = data_icpms_overlay.loc[:, icpms_y_col_background_correction] - data_icpms_overlay.loc[:, icpms_y_col_background]
    
    return data_icpms_overlay, icpms_y_col_background, icpms_y_col_background_corrected