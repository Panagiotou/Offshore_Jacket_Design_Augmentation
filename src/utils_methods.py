import json
import matplotlib.lines as mlines
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sys
sys.path.append("src")


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))

def plot_discr(out, train_original, x_lim=[], y_lim=[], elipse=True):
    df_data = pd.DataFrame()
    df_generated = pd.DataFrame()
    
    df_data["x"] = out['Ez.y'][:,0]
    df_data["y"] = out['Ez.y'][:,1]
    
    zz = out['zz']
    
#     zz = famd.transform(pred_famd)
    
    df_generated["x"],  df_generated["y"] = zz[:,0], zz[:,1]
    
    df_generated["method"] = "synthetic (MIAMI)"
    
    df_data["method"] = train_original["label"]
    
    df_scatter = pd.concat([df_data, df_generated], axis=0, ignore_index=True)
    

    # Create a scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.scatterplot(data=df_scatter, x="x", y="y", style="method")
    # Add a title and axis labels
    ax.set_title('Scatter plot grouped by type')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    # Create a legend
    legend = ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), title="Data", borderaxespad=0.)
    if len(x_lim)>0:
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[0], y_lim[1])
        
    if elipse:
        weights_ = out['w_s']  
        means_ = out['mu'][0]
        covariances_ = out['sigma'][0]
        
        w_factor = 0.2 / weights_.max()
        for pos, covar, w in zip(means_, covariances_, weights_):
            draw_ellipse(pos, covar, alpha=w * w_factor)
    plt.show()

def sample_points_in_circle(center_x, center_y, radius, num_points):
    # Generate random polar coordinates
    theta = 2 * np.pi * np.random.rand(num_points)
    r = radius * np.sqrt(np.random.rand(num_points))
    
    # Convert polar coordinates to Cartesian coordinates
    x = center_x + r * np.cos(theta)
    y = center_y + r * np.sin(theta)
    
    return np.array(list(zip(x, y)))

def plot_near(z_points, item, out, x_lim=[], y_lim=[], elipse=True):
    df_data = pd.DataFrame()
    df_generated = pd.DataFrame()
    
    
    df_data["x"] = out['Ez.y'][:,0]
    df_data["y"] = out['Ez.y'][:,1]
    
    
#     zz = famd.transform(pred_famd)
    
    
    
    df_data["method"] = "real"
    df_data["size"] = 20
    
    df_scatter = pd.concat([df_data], axis=0, ignore_index=True)
    
    df_item = pd.concat([df_scatter.loc[item:item]], axis=0, ignore_index=True) 
    
    df_item["method"] = "selected"
    df_item["size"] = 80


    points = z_points
    df_points = pd.DataFrame(points, columns=["x", "y"])
    df_points["method"] = "sampled"
    df_points["size"] = 80
    
    all_dfs = [df_scatter, df_item, df_points]

    df_all = pd.concat(all_dfs, axis=0, ignore_index=True )

    # Create a scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = sns.scatterplot(data=df_all, x="x", y="y", hue="method", style="method", size="size")

    # Add a title and axis labels
    ax.set_title('Scatter plot grouped by type')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    methods = df_all["method"].unique()
    # Customize the legend labels
    handles, labels = ax.get_legend_handles_labels()
    # indices = np.where(np.isin(labels, methods))[0].tolist()
    # handles = list(np.array(handles)[indices])
    # labels = list(np.array(labels)[indices])
    handles = handles[1:4]
    labels = labels[1:4]
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), title="Data", borderaxespad=0., handles=handles, labels=labels)
    if len(x_lim)>0:
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[0], y_lim[1])
        
    if elipse:
        weights_ = out['w_s']  
        means_ = out['mu'][0]
        covariances_ = out['sigma'][0]
        
        w_factor = 0.2 / weights_.max()
        for pos, covar, w in zip(means_, covariances_, weights_):
            draw_ellipse(pos, covar, alpha=w * w_factor)
    plt.savefig("sampling.png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.show()

    return points

from .oversample import draw_new_bin, draw_new_ord,\
                       draw_new_categ, draw_new_cont

def generate_pseudo(point, out, nj, var_distrib, le_dict, num_points=10, r=0.5):  
    
    x_val = out['Ez.y'][point,0]
    y_val = out['Ez.y'][point,1]
    
    

    
    samples_total = 5*num_points # sample 5 times more than wanted because we loose some
    z = sample_points_in_circle(x_val, y_val, r, samples_total)
    
    lambda_bin = np.array(out['lambda_bin']) 
    lambda_ord = out['lambda_ord'] 
    lambda_categ = out['lambda_categ'] 
    lambda_cont = np.array(out['lambda_cont'])
    
    nj_bin = nj[pd.Series(var_distrib).isin(['bernoulli', 'binomial'])].astype(int)
    nj_ord = nj[var_distrib == 'ordinal'].astype(int)
    nj_categ = nj[var_distrib == 'categorical'].astype(int)

    y_std = out['y_std']
    
    y_bin_new = []
    y_categ_new = []
    y_ord_new = []
    y_cont_new = []


    y_bin_new.append(draw_new_bin(lambda_bin, z, nj_bin))
    y_categ_new.append(draw_new_categ(lambda_categ, z, nj_categ))
    y_ord_new.append(draw_new_ord(lambda_ord, z, nj_ord))
    y_cont_new.append(draw_new_cont(lambda_cont, z))
    # Stack the quantities
    y_bin_new = np.vstack(y_bin_new)
    y_categ_new = np.vstack(y_categ_new)
    y_ord_new = np.vstack(y_ord_new)
    y_cont_new = np.vstack(y_cont_new)

    # "Destandardize" the continous data
    y_cont_new = y_cont_new * y_std

    # Put them in the right order and append them to y
    type_counter = {'count': 0, 'ordinal': 0,\
                    'categorical': 0, 'continuous': 0} 

    y_new = np.full((samples_total, out['y_shape'][1]), np.nan)
    # Quite dirty:
    for j, var in enumerate(var_distrib):
        if (var == 'bernoulli') or (var == 'binomial'):
            y_new[:, j] = y_bin_new[:, type_counter['count']]
            type_counter['count'] =  type_counter['count'] + 1
        elif var == 'ordinal':
            y_new[:, j] = y_ord_new[:, type_counter[var]]
            type_counter[var] =  type_counter[var] + 1
        elif var == 'categorical':
            y_new[:, j] = y_categ_new[:, type_counter[var]]
            type_counter[var] =  type_counter[var] + 1
        elif var == 'continuous':
            y_new[:, j] = y_cont_new[:, type_counter[var]]
            type_counter[var] =  type_counter[var] + 1
        else:
            raise ValueError(var, 'Type not implemented')
    cols = out['cols']
    layer_cols = np.array([i for i, s in enumerate(cols) if s.startswith('layer_height')])
    brace_cols = np.array([i for i, s in enumerate(cols) if s.startswith('brace')])
    other_continuous_cols = list(set(np.where(var_distrib == 'continuous')[0]) - set(layer_cols))
    ranges_cont = [out['ranges'][other_continuous_cols], other_continuous_cols]
    
    mask = np.apply_along_axis(lambda x: check_constraints(x, var_distrib, le_dict, cols, layer_cols, brace_cols, continuous_ranges=ranges_cont), axis=1, arr=y_new)
    filtered_rows = y_new[mask]
    filtered_z = z[mask]
    filtered_rows = filtered_rows[:num_points,:]
    filtered_z = filtered_z[:num_points,:]
    return filtered_rows, filtered_z

  
# from https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


def encode(d, max_layers, brace_dict, N_BRACES, one_hot=True, native=False, normalize_layer_heights=False):
    basics = [d["legs"], d["total_height"], d["radius_bottom"], d["radius_top"], d["n_layers"]]
    
    # fill design's braces according to max_layers with dummies ("NONE")
    braces = d["connection_types"]
    if native:
        braces = np.array([b for b in braces] + ["nan"] * (max_layers - 1 - len(braces)))
    else:
        braces = np.array([brace_dict[b] for b in braces] + [brace_dict["nan"]] * (max_layers - 1 - len(braces)))
        if one_hot:
            braces = get_one_hot(braces, N_BRACES)
    
    # fill design's layer_heights according to max_layers with dummies
    layer_heights = d["layer_heights"]
    if normalize_layer_heights:
        layer_heights = np.array(layer_heights + [d["total_height"]] * (max_layers - 2 - len(layer_heights))) / d["total_height"]
    else:
        layer_heights = np.array(layer_heights + [1] * (max_layers - 2 - len(layer_heights)))

    
    
    # return a flat encoding
    return np.array([*basics, *braces.flatten(), *layer_heights])

def transform_df_to_json(input_df):
    output = []
    id_counter = 1

    for index, row in input_df.iterrows():
        n_layers = row["n_layers"]
        connection_types = [row[f"brace{i}"] for i in range(n_layers-1) if row[f"brace{i}"] != "nan"]
        layer_heights = [row[f"layer_height{i}"]*row["total_height"] for i in range(n_layers-2)]

        output.append({
            "legs": row["legs"],
            "total_height": row["total_height"],  
            "connection_types": connection_types,
            "radius_bottom": row["radius_bottom"],
            "radius_top": row["radius_top"],
            "n_layers": row["n_layers"],
            "layer_heights": layer_heights,
            "id": id_counter
        })

        id_counter += 1

    return output

def generate_plots(data_to_plot, train_original, pred_post, zz_train, zz_synthetic, var_distrib, le_dict, brace_cols, unique_braces, res_folder, percentage=False):

    x_lim = []
    y_lim = []
    s_K=2 # number of variances from mean

    df_data = pd.DataFrame()
    if zz_train is not None:
        df_data["x"], df_data["y"] = zz_train[:,0], zz_train[:,1]

    df_generated = pd.DataFrame()
    if zz_synthetic is not None:
        df_generated["x"],  df_generated["y"] = zz_synthetic[:,0], zz_synthetic[:,1]
        
        
        
        
    continuous_position = 0
    for col_idx, colname in enumerate(train_original.columns[:-1]):
        df_data[colname] = train_original[colname]
        df_data["Data"] = train_original["label"]
        
        if pred_post is not None:
            df_generated[colname] = pred_post[colname]


            df_generated["Data"] = "synthetic"
            
            df_scatter = pd.concat([df_data, df_generated], axis=0, ignore_index=True)
        else:
            df_scatter = df_data
        
        df_scatter = df_scatter[df_scatter["Data"].isin(data_to_plot)]
    
        if var_distrib[col_idx] in ['categorical', 'bernoulli', "ordinal"]: 

            unique_labels = le_dict[colname].classes_
            le_name_mapping = dict(zip(le_dict[colname].classes_, le_dict[colname].transform(le_dict[colname].classes_)))
            if colname in brace_cols:
                unique_vals = unique_braces
            else:  
                unique_vals = df_scatter[colname].to_numpy().flatten()
                unique_vals = list(set(unique_vals))
                unique_vals.sort()

            if zz_train is not None:
                # Create a scatter plot
                fig, ax = plt.subplots(figsize=(16, 6), nrows=1, ncols=2)

                    
                sns.scatterplot(data=df_scatter, x="x", y="y", hue=colname, style="Data", hue_order=unique_vals, ax=ax[1], palette=sns.color_palette("tab10"))
                # Add a title and axis labels
                ax[1].set_title('Scatter plot grouped by ' + colname)
                ax[1].set_xlabel('PC1')
                ax[1].set_ylabel('PC2')
                if len(x_lim)>0:
                    ax[1].set_xlim(x_lim[0], x_lim[1])
                    ax[1].set_ylim(y_lim[0], y_lim[1])
                # Create a legend
                legend = ax[1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)
                
                distribution_ax = ax[0]
            else:
                fig, ax = plt.subplots(figsize=(6, 6), nrows=1, ncols=1)
                distribution_ax = ax
            # Count the occurrences of each number of legs per cluster
            items_per_cluster = df_scatter.groupby(['Data', colname]).size().unstack().fillna(0)
            
            rev_unique_vals = unique_vals
            change_cols_list = [col for col in rev_unique_vals if col in items_per_cluster.columns]
            
            # Reorder the columns based on the sorted labels
            items_per_cluster = items_per_cluster[change_cols_list]
            df = df_scatter.astype(str)
            # Calculate percentages
            # Calculate percentages per 'Data' category
            if percentage:
                df['Percentage'] = df.groupby(['Data', colname])['Data'].transform('count') / df.groupby(['Data'])['Data'].transform('count') * 100

                # Aggregate by taking the mean
                df_aggregated = df.groupby(['Data', colname])['Percentage'].mean().reset_index()

                # Pivot the DataFrame to get 'real' and 'synthetic' columns
                df_pivot = df_aggregated.pivot(index='Data', columns=colname, values='Percentage').fillna(0)

                # Normalize the values to make sure they sum to 100
                df_pivot = df_pivot.div(df_pivot.sum(axis=1), axis=0) * 100

                ordering = [str(x) for x in unique_vals]

                custom_colors = sns.color_palette("tab10")
                

                df_pivot.plot(kind='bar', stacked=True, ax=distribution_ax, color=custom_colors, width=1, edgecolor='black')
            else:
                order = [str(x) for x in unique_vals]
                g = sns.histplot(df_scatter.astype(str), x='Data', hue=colname, hue_order=order, multiple='stack', ax=distribution_ax, palette=sns.color_palette("tab10"))
    #         g.legend(handles, labels)
    #         ax.margins(x=0.5)
            distribution_ax.xaxis.set_tick_params(rotation=0)
            if res_folder is not None:
                plt.savefig(res_folder + "figures/" + "synthetic_" + colname + ".png", 
                    bbox_inches='tight', 
                    transparent=True,
                    pad_inches=0, dpi=200)
            plt.show()
            
        else:
            if zz_train is not None:
                # Create a scatter plot
                fig, ax = plt.subplots(figsize=(16, 6), nrows=1, ncols=2)
                # # calculate summary statistics
                # data_mean, data_std = np.mean(train_original[colname]), np.std(train_original[colname])
                # # identify outliers
                # cut_off = data_std * s_K
                # lower, upper = data_mean - cut_off, data_mean + cut_off
                # identify outliers
        #         outliers = [x for x in train_original[colname] if x < lower or x > upper]
        #         print('Identified outliers: %d' % len(outliers))
                
                # Compare woman, 60+ y.o and people presenting both modalities


        #         scatter = ax[1].scatter(x=df_scatter['x'], y=df_scatter['y'], c=train_original[colname].astype(int), cmap='RdBu_r', s=10, vmin=lower, vmax=upper)
                scatter = sns.scatterplot(data=df_scatter, x="x", y="y", hue=colname, style="Data", palette="RdBu_r", ax=ax[1])
                
                norm = plt.Normalize(df_scatter[colname].min(), df_scatter[colname].max())
                sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
                sm.set_array([])

                # Remove the legend and add a colorbar
                ax[1].get_legend().remove()
                ax_cbar = fig.add_axes([0.85, 0.1, 0.03, 0.8])  # Adjust the position as needed
                # ax_cbar.figure.colorbar(sm)
                cbar = plt.colorbar(sm, ax=ax_cbar)

                
                # Add a title and axis labels
                ax[1].set_title('Scatter plot grouped by ' + colname)
                ax[1].set_xlabel('PC1')
                ax[1].set_ylabel('PC2') 
                distribution_ax = ax[0]

            else:
                fig, ax = plt.subplots(figsize=(6, 6), nrows=1, ncols=1)
                distribution_ax = ax
    #        box_to_anchor=(1.05, 0.5), title=colname, borderaxespad=0.)
            continuous_position += 1
            if len(x_lim)>0:
                ax[1].set_xlim(x_lim[0], x_lim[1])
                ax[1].set_ylim(y_lim[0], y_lim[1])
                
            sns.kdeplot(data=df_scatter, x=colname, hue="Data", ax=distribution_ax, bw_adjust=0.3, common_norm=not percentage)
            if res_folder is not None:
                plt.savefig(res_folder + "figures/" + "synthetic_" + colname + ".png", 
                bbox_inches='tight', 
                transparent=True,
                pad_inches=0, dpi=200)
            plt.show()


def check_constraints(y, var_distrib, le_dict, colnames, layer_height_cols, brace_cols, continuous_ranges=None, useful_cols=None):
    cont = y[var_distrib == 'continuous']

    if not useful_cols:
        useful_cols = ['radius_bottom', 'radius_top', 'n_layers']

    
    
    for j, colname in enumerate(colnames):
        if colname in le_dict.keys():
            if y[j].astype(int) not in list(range(len(le_dict[colname].classes_))):
                return False
    
#     y_trans = le_dict[colname].inverse_transform(pred[colname].astype(int))
    
    if (cont<0).any():
        return False
    if continuous_ranges is not None:
        ranges = continuous_ranges[0]
        column_ranges = continuous_ranges[1]
        values = y[column_ranges]
        
        within_ranges = np.logical_and(values >= ranges[:, 0], values <= ranges[:, 1])

        if not within_ranges.all():
            return False

    if y[colnames == useful_cols[0]] < y[colnames == useful_cols[1]]:
        # radius bottom bigger than radius top
        return False
    
    n_layers = int(y[colnames == useful_cols[2]])
    n_layers_decode = int(le_dict[useful_cols[2]].inverse_transform([n_layers])[0])
    n_layer_corrected = n_layers_decode - 2
    braces_corrected = n_layers_decode - 1
    relevant_brace = brace_cols[:braces_corrected]
    relevant_brace_cols = zip(y[relevant_brace], colnames[relevant_brace])

    transf_brace = []
    for brace_col, brace_col_name in relevant_brace_cols:
        if int(brace_col) in list(range(len(le_dict[brace_col_name].classes_))):
            transf_brace.append(le_dict[brace_col_name].inverse_transform([int(brace_col)])[0])
        else:
            return False
    # [le_dict[brace_col_name].inverse_transform([int(brace_col)])[0] for brace_col, brace_col_name in relevant_brace_cols]
    if "nan" in transf_brace:
        return False
    
    if n_layer_corrected>0:
        layer_heights = y[layer_height_cols]
        relevant_layer_heights = layer_heights[:n_layer_corrected]
        
        if (relevant_layer_heights>=1).any():
            return False
        is_ascending = all(relevant_layer_heights[i] < relevant_layer_heights[i + 1] for i in range(len(relevant_layer_heights) - 1))
        if not is_ascending:
            return False
    return True


from sklearn.preprocessing import LabelEncoder 
from copy import deepcopy

def get_var_metadata(train, train_original, brace_cols, BERNOULLI):
    #*****************************************************************
    # Formating the data
    #*****************************************************************
    # 
    unique_counts = train.nunique()
    le_dict = {}

    var_distrib = []     
    var_transform_only = []     
    # Encode categorical datas
    for colname, dtype, unique in zip(train.columns, train.dtypes, train.nunique()):
        if unique < 2:
            print("Dropped", colname, "because of 0 var")
            train.drop(colname, axis=1, inplace=True)
            continue

        if (dtype==int or dtype == "object") and unique==2:
            #bool
            
            var_distrib.append('bernoulli')
            if colname in BERNOULLI:
                var_transform_only.append('bernoulli')
            else:
                var_transform_only.append('categorical')
                
            le = LabelEncoder()
            # Convert them into numerical values               
            train[colname] = le.fit_transform(train[colname]) 
            le_dict[colname] = deepcopy(le)
        elif dtype==int and unique > 2:
            # ordinal
            
            var_distrib.append('ordinal')
            var_transform_only.append('ordinal')
            
            le = LabelEncoder()
            # Convert them into numerical values               
            train[colname] = le.fit_transform(train[colname]) 
            le_dict[colname] = deepcopy(le)
        elif dtype == "object":
            var_distrib.append('categorical')
            var_transform_only.append('categorical')
            
            le = LabelEncoder()
            # Convert them into numerical values               
            train[colname] = le.fit_transform(train[colname]) 
            le_dict[colname] = deepcopy(le)
        elif dtype == float:
            var_distrib.append('continuous')
            var_transform_only.append('continuous')
            
        
    var_distrib = np.array(var_distrib)
    unique_braces = train_original[brace_cols].to_numpy().flatten()
    unique_braces = list(set(unique_braces))
    unique_braces.sort()
    return var_distrib, var_transform_only, le_dict, brace_cols, unique_braces


def post_process(pred, train_columns, le_dict, discrete_features, brace_cols, dtype, layer_height_cols, layer_col="n_layers"):

    pred_trans = pred.copy()

    for j, colname in enumerate(train_columns):
        if colname in le_dict.keys():
            pred_trans[colname] = le_dict[colname].inverse_transform(pred[colname].astype(int))
        
    pred_post = pred_trans.copy()    
    pred_famd = pred.copy()    

    pred_famd[discrete_features] = pred_famd[discrete_features].astype(int)

    layer_height_cols = train_columns[layer_height_cols]
    # Define a function to apply to each row
    def replace_layer_height(row, brace_cols, layer_height_cols):
        n_layers = int(row[layer_col])
        layers_set_one = layer_height_cols[n_layers-2:]
        braces_set_nan = brace_cols[n_layers-1:]
        # Set the layer_height columns to 1.0 for all rows after the nth row
        row[layers_set_one] = 1.0
        row[braces_set_nan] = "nan"
        
        # Return the modified row
        return row

    # Apply the function to each row of the DataFrame
    pred_post = pred_post.apply(lambda x: replace_layer_height(x, brace_cols, layer_height_cols), axis=1)

    pred_famd = pred_famd.astype(dtype, copy=True)

    pred_post_famd = pred_famd.copy()
    pred_post_famd[layer_height_cols] = pred_post[layer_height_cols]



    pred_post_famd[discrete_features] = pred_post_famd[discrete_features].astype(int)

    pred_post_famd = pred_post_famd.astype(dtype, copy=True)

    return pred_post, pred_post_famd