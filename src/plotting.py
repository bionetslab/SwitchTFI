
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import graph_tool.all as gt
import networkx as nx

from typing import *
from pdf2image import convert_from_path
from PIL import Image, ImageChops
from .tf_ranking import grn_to_nx


def plot_grn(grn: pd.DataFrame,
             gene_centrality_df: Union[pd.DataFrame, None] = None,
             plot_folder: Union[str, None] = None,
             weight_key: str = 'weight',
             pval_key: str = 'pvals_wy',
             tf_target_keys: Tuple[str, str] = ('TF', 'target'),
             axs: Union[plt.Axes, None] = None,
             **kwargs):
    """
    Plot the gene regulatory network (GRN) and visualize transcription factor (TF) rankings.

    This function visualizes the GRN by plotting TF-target gene relationships using the `graph_tool` library.
    Vertices marked with green edging are TFs. A darker shade of blue for the fill color signifies a higher relevance
    of the TF to the cell state transition (ranking is based on the centrality values of the TFs).
    Edge thickness is proportional to the edge weight, and edge opacity is proportional to the empirical p-value
    of the edge (high opacity indicates low/significant p-values).
    The plot is saved as a PDF and optionally displayed in a matplotlib axis.

    Args:
        grn (pd.DataFrame): The GRN DataFrame containing TF-target gene pairs.
        gene_centrality_df (Union[pd.DataFrame, None], optional): A DataFrame of gene centrality rankings. Defaults to None.
        plot_folder (Union[str, None], optional): Folder to save the plot. Defaults to None.
        weight_key (str, optional): Column name representing edge weights in the GRN. Defaults to 'weight'.
        pval_key (str, optional): Column name representing p-values for edges in the GRN. Defaults to 'pvals_wy'.
        tf_target_keys (Tuple[str, str], optional): Column names for TFs and targets in the GRN. Defaults to ('TF', 'target').
        axs (Union[plt.Axes, None], optional): Matplotlib axis to display the plot. Defaults to None.
        **kwargs: Additional arguments, such as `fn_prefix` for filename prefixes.

    Returns:
        None: The function saves the plot to the specified folder and optionally displays it in a matplotlib axis.
    """

    g = grn_to_gt(grn=grn,
                  gene_centrality_df=gene_centrality_df,
                  weight_key=weight_key,
                  pval_key=pval_key,
                  tf_target_keys=tf_target_keys)

    if plot_folder is None:
        plot_folder = './'

    prefix = kwargs.get('fn_prefix')
    if prefix is None:
        prefix = ''
    plot_p = os.path.join(plot_folder, f'{prefix}grn.pdf')

    # ### Vertex position
    pos = gt.sfdp_layout(g, C=0.01, p=2, r=10)  # r=6)
    # pos = gt.fruchterman_reingold_layout(g)
    # pos = gt.arf_layout(g, max_iter=100, epsilon=10**(-4))

    # ### Edge properties
    # Get rounded edge weights as string ep for plotting weight on edges
    rounded_weights = g.new_ep('string')
    for e in g.edges():
        rounded_weights[e] = str(round(g.ep[weight_key][e], ndigits=3))

    # Define edge width proportional to edge weight
    ewidth = g.new_ep('long double')
    a = g.ep[weight_key].a
    if (a.max() - a.min()) != 0:
        a = (a - a.min()) / (a.max() - a.min()) * 3
    else:
        a = np.ones(shape=a.shape) * 1.5
    ewidth.a = a

    # Define colors of edges based on p-values
    p = g.ep[pval_key].a
    if (p.max() - p.min()) != 0:
        p = (p - p.min()) / (p.max() - p.min())
    else:
        p = np.ones(shape=p.shape) * 0.5

    p = get_rgba_color_gradient(rgba1=[0, 0, 0, 0.7],
                                rgba2=[0, 0, 0, 0.1],
                                values=np.array(p))

    e_color = g.new_ep('vector<long double>')
    for i, e in enumerate(g.edges()):
        e_color[e] = p[i, :]

    # ### Vertex properties
    # Define green outline of TF vertices
    vert_outline_col = g.new_vp('vector<long double>')
    for v in g.vertices():
        if g.vp['tf'][v]:
            vert_outline_col[v] = [0, 193/255, 0, 0.9]
        else:
            vert_outline_col[v] = [0, 0, 0, 0.3]

    # Define fill color vertices, based on TF ranking
    vert_fill_col = g.new_vp('vector<long double>')
    if gene_centrality_df is not None:

        tf_bool = g.vp['tf'].a.astype('bool')
        sorted_indices = np.argsort(g.vp[gene_centrality_df.columns[1]].a)[::-1]  # vertex index = index
        tf_indices = np.where(tf_bool)[0]
        sorted_tfs = sorted_indices[np.isin(sorted_indices, tf_indices)]

        tf_rgba0 = [19, 0, 255, 0.9]
        tf_rgba1 = [0, 255, 231, 0.3]
        tf_cols = get_rgba_color_gradient(rgba1=tf_rgba0,
                                          rgba2=tf_rgba1,
                                          values=int(tf_bool.sum()))

    for v in g.vertices():
        if not g.vp['tf'][v]:
            vert_fill_col[v] = [255, 147, 0, 0.5]
        else:
            if gene_centrality_df is not None:
                vert_fill_col[v] = tf_cols[np.where(sorted_tfs == int(v))[0][0], :]
            else:
                vert_fill_col[v] = [39/255, 123/255, 245/255, 0.5]

    gt.graph_draw(g,
                  pos=pos,
                  nodesfirst=False,
                  vprops={'text': g.vp['gene_name'],
                          'text_position': -2,
                          'font_size': 3,
                          'text_color': 'black',
                          'color': vert_outline_col,
                          'fill_color': vert_fill_col},
                  eprops={# 'text': rounded_weights,
                          # 'font_size': 3,
                          'pen_width': ewidth,
                          'color': e_color},
                  output=plot_p)

    if axs is not None:
        # Plot pdf in matplotlib figure
        conda_env_path = sys.prefix
        poppler_path = os.path.join(conda_env_path, 'bin')

        def trim_whitespace(img):
            bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
            diff = ImageChops.difference(img, bg)
            bbox = diff.getbbox()
            if bbox:
                return img.crop(bbox)
            return img

        pages = convert_from_path(plot_p, first_page=1, last_page=1, poppler_path=poppler_path)

        axs.imshow(trim_whitespace(pages[0]))
        # axs.imshow(pages[0])
        axs.axis('off')

        '''# Add colorbars
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.patches as patches

        sorted_tf_names = [g.vp['gene_name'][tf] for tf in sorted_tfs]

        # Create a divider for the existing axes instance
        divider = make_axes_locatable(axs)

        # Append a new axes below the imshow plot with 20% height of ax
        cax = divider.append_axes("bottom", size="20%", pad=0.6)

        # Define the positions for each circle
        positions = np.linspace(0.1, 0.9, len(sorted_tf_names))

        for i in range(len(sorted_tf_names)):
            circle = patches.Circle((positions[i], 0.5), radius=0.1, edgecolor='black', facecolor=tf_cols[i],
                                    linewidth=2)
            cax.add_patch(circle)

            # Add the text inside the circle
            cax.text(positions[i], 0.5, sorted_tf_names[i], fontsize=12, ha='center', va='center', color='black')

        # Adjust limits and remove axes for the circle plot
        cax.set_xlim(0, 1)
        cax.set_ylim(0, 1)
        cax.axis('off')'''


def plot_regulon(grn: pd.DataFrame,
                 tf: str,
                 sort_by: Union[str, None] = None,
                 top_k: Union[int, None] = None,
                 out: bool = True,
                 weight_key: str = 'weight',
                 pval_key: str = 'pvals_wy',
                 show: bool = False,
                 title: Union[str, None] = None,
                 dpi: int = 100,
                 node_size: int = 600,
                 font_size: int = 9,
                 tf_target_keys: Tuple[str, str] = ('TF', 'target'),
                 axs: Union[plt.Axes, None] = None):
    """
    Plots a transcription factor (TF) and its targets as a subnetwork of the gene regulatory network.

    Args:
        grn (pd.DataFrame): The GRN DataFrame with TF-target gene pairs and additional values such as weights and P-values.
        tf (str): The transcription factor.
        sort_by (Union[str, None], optional): Column to sort edges by (e.g., 'score'). Defaults to None.
        top_k (Union[int, None], optional): Number of top edges to display. Defaults to None (display all).
        out (bool, optional): If True, plots outgoing edges from TF. If False, plots incoming edges to TF. Defaults to True.
        weight_key (str, optional): Column for edge weights. Defaults to 'weight'.
        pval_key (str, optional): Column for p-values of edges. Defaults to 'pvals_wy'.
        show (bool, optional): If True, displays the plot. Defaults to False.
        title (Union[str, None], optional): Title for the plot. Defaults to None.
        dpi (int, optional): Resolution of the plot. Defaults to 100.
        node_size (int, optional): Size of the network nodes. Defaults to 600.
        font_size (int, optional): Font size for node labels. Defaults to 9.
        tf_target_keys (Tuple[str, str], optional): Column names for TF and target in the GRN. Defaults to ('TF', 'target').
        axs (Union[plt.Axes, None], optional): Existing matplotlib axis to plot on. Defaults to None.

    Returns:
        None: The function generates and optionally displays the plot.
    """

    if out:
        # Keep edges where the TF is the desired TF
        keep_bool = np.isin(grn[tf_target_keys[0]].to_numpy(), [tf])
    else:
        # Keep edges where the target is the desired TF
        keep_bool = np.isin(grn[tf_target_keys[1]].to_numpy(), [tf])

    regulon = grn[keep_bool].copy().reset_index(drop=True)

    if sort_by is not None:
        if sort_by not in regulon.columns and sort_by == 'score':
            weights = regulon[weight_key].to_numpy()
            pvals = regulon[pval_key].to_numpy()
            pvals += np.finfo(np.float64).eps
            regulon['score'] = -np.log10(pvals) * weights

        regulon = regulon.sort_values(sort_by, axis=0, ascending=False)

    if top_k is not None:
        regulon = regulon[0:top_k]

    reg = grn_to_nx(grn=regulon, edge_attributes=True, tf_target_keys=tf_target_keys)

    if axs is None:
        fig, axs = plt.subplots(dpi=dpi)

    pos = nx.spring_layout(reg)

    tf_bool = (np.array(reg.nodes) == tf)
    node_cols = np.empty(reg.number_of_nodes(), dtype=object)
    node_cols[tf_bool] = '#277bf580'  # rgba(39, 123, 245, 0.5)
    node_cols[~tf_bool] = '#f5c82780'  # rgba(245, 200, 39, 0.5)
    node_edge_cols = np.empty(reg.number_of_nodes(), dtype=object)
    node_edge_cols[tf_bool] = '#00c100e6'
    node_edge_cols[~tf_bool] = '#0000004d'
    node_edge_widths = np.ones(reg.number_of_nodes())
    node_edge_widths[tf_bool] *= 3

    if regulon.shape[0] > 1:
        edge_cols = regulon[sort_by].to_numpy()
        edge_cols = - np.log10(edge_cols)
        edge_cols = 0.1 + (edge_cols - edge_cols.min()) / (edge_cols.max() - edge_cols.min()) * (0.9 - 0.1)
        ec_list = []
        for ec in edge_cols:
            ec_list.append((0, 0, 0, ec))
    else:
        ec_list = (0, 0, 0, 0.9)

    nx.draw_networkx_nodes(reg, pos, node_size=node_size, node_color=node_cols, linewidths=node_edge_widths,
                           edgecolors=node_edge_cols, alpha=0.8, ax=axs)
    nx.draw_networkx_labels(reg, pos, font_size=font_size, font_color='black', ax=axs)
    nx.draw_networkx_edges(reg, pos, width=2.0, edge_color=ec_list, node_size=node_size, ax=axs)

    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['left'].set_visible(False)
    axs.spines['bottom'].set_visible(False)

    if title is not None:
        axs.set_title(title)

    plt.tight_layout()

    if show:
        plt.show()


# Auxiliary ############################################################################################################
def grn_to_gt(grn: pd.DataFrame,
              gene_centrality_df: Union[pd.DataFrame, None],
              weight_key: str = 'weight',
              pval_key: str = 'pvals_wy',
              tf_target_keys: Tuple[str, str] = ('TF', 'target')):

    # Get edge list and corresponding int -> gene name mapping
    edge_list, idx_gn_dict = build_edge_list(grn=grn, tf_target_keys=tf_target_keys)

    # Add attribute values to edge list
    weights = np.reshape(grn[weight_key].to_numpy(), newshape=(grn.shape[0], 1))
    pvals = np.reshape(grn[pval_key].to_numpy(), newshape=(grn.shape[0], 1))
    edge_list = np.hstack((edge_list, weights, pvals))

    # Initialize graph
    g = gt.Graph(directed=True)

    # Add edges with respective weight and p-value
    eweight = g.new_ep('long double')
    epval = g.new_ep('long double')
    g.add_edge_list(edge_list=edge_list, eprops=[eweight, epval])
    g.ep[weight_key] = eweight
    g.ep[pval_key] = epval

    # Add vertex properties
    gene_names = g.new_vertex_property('string')
    is_tf = g.new_vertex_property('bool')
    tfs = np.unique(grn[tf_target_keys[0]].to_numpy())

    if gene_centrality_df is not None:
        gene_centrality_df = gene_centrality_df.copy()
        gene_centrality_df.set_index(keys=['gene'], drop=True, inplace=True)
        centrality = g.new_vertex_property('double')

    for v in g.vertices():
        gene_name = idx_gn_dict[int(v)]
        gene_names[v] = gene_name
        is_tf[v] = bool(np.isin(gene_name, tfs))
        if gene_centrality_df is not None and bool(np.isin(gene_name, tfs)):
            centrality[v] = gene_centrality_df.at[gene_name, gene_centrality_df.columns[0]]

    g.vp['gene_name'] = gene_names
    g.vp['tf'] = is_tf
    if gene_centrality_df is not None:
        g.vp[gene_centrality_df.columns[0]] = centrality

    return g


def build_edge_list(grn: pd.DataFrame,
                    tf_target_keys: Tuple[str, str] = ('TF', 'target')) -> Tuple[np.ndarray, dict]:
    tf_key = tf_target_keys[0]
    target_key = tf_target_keys[1]
    # Get unique gene names from GRN
    gene_names = np.unique(grn[[tf_key, target_key]].to_numpy())
    # Map gene name to integer indices, store mapping in dict
    gn_idx_dict = {}
    idx_gn_dict = {}
    for i, gn in enumerate(gene_names):
        gn_idx_dict[gn] = i
        idx_gn_dict[i] = gn
    # Iterate over GRN and successively create edge list with integer entries
    edge_list = np.empty((grn.shape[0], 2))
    for i in range(grn.shape[0]):
        edge_list[i, 0] = gn_idx_dict[grn[tf_key].iloc[i]]
        edge_list[i, 1] = gn_idx_dict[grn[target_key].iloc[i]]

    return edge_list, idx_gn_dict


def get_rgba_color_gradient(rgba1: List[float],
                            rgba2: List[float],
                            values: Union[int, np.ndarray]) -> np.ndarray:

    if isinstance(values, int):
        points = np.linspace(0, 1, values)

        out = interpolation_function(rgba1=rgba1, rgba2=rgba2, points=points)
    else:
        out = interpolation_function(rgba1=rgba1, rgba2=rgba2, points=values)

    return out


def interpolation_function(rgba1: List[float],
                           rgba2: List[float],
                           points: np.ndarray) -> np.ndarray:
    # Scale to [0,1] range
    rgba1 = np.array([a / 255 if i < 3 else a for i, a in enumerate(rgba1)])
    rgba2 = np.array([a / 255 if i < 3 else a for i, a in enumerate(rgba2)])

    # Calculate linear interpolation for each value in array -> n-values x 4 matrix
    # out = np.minimum(rgba1, rgba2) + np.expand_dims(points, 1) * (np.maximum(rgba1, rgba2) - np.minimum(rgba1, rgba2))

    out = rgba1 + np.expand_dims(points, 1) * (rgba2 - rgba1)

    return out
