
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from PIL import Image
import io
import numpy as np

def figure_to_array(fig):
    image_bytes = fig.to_image(format='png')

    picture_stream = io.BytesIO(image_bytes)

    img = np.array(Image.open(picture_stream))
    
    return img

def plot_orginal_heatmaps(y, pred, dataset, BATCH_ID=0, loss=None, return_fig=False):
    eq = (y == pred).astype(int)
    

    desc_y = [i.split(" ") for i in dataset.decode_pred(y, BATCH_ID).split("\n")]
    desc_y_other = [i.split(" ") for i in dataset.decode_pred(pred, BATCH_ID).split("\n")]

    padded_desc_y = [i + [None] * (y.shape[-1] - len(i)) for i in desc_y]
    padded_desc_y_other = [i + [None] * (y.shape[-1] - len(i)) for i in desc_y_other]


    
    
    batch_eq = np.flip(eq[BATCH_ID, :, :], 0)
    text = np.flip(np.array(padded_desc_y)[:-1, :], 0)
    
    fig1 = ff.create_annotated_heatmap(batch_eq, annotation_text=text)
    fig1.update_layout(height=800, width=500, 
                    margin=dict(l=20, r=20, t=20, b=20),
    )
    
    if loss is None:
        loss = eq
    batch_eq = np.flip(loss[BATCH_ID, :, :], 0)
    text = np.flip(np.array(padded_desc_y_other)[:-1, :], 0)
    
    fig2 = ff.create_annotated_heatmap(batch_eq, annotation_text=text)
    fig2.update_layout(height=800, width=500, 
                    margin=dict(l=20, r=20, t=20, b=20),
    )

    fig = make_subplots(rows=1, cols=2, subplot_titles = ['True', 'Pred'])
    fig.add_trace(fig1.data[0], 1, 1)
    fig.add_trace(fig2.data[0], 1, 2)

    annot1 = list(fig1.layout.annotations)
    annot2 = list(fig2.layout.annotations)
    for k  in range(len(annot2)):
        annot2[k]['xref'] = 'x2'
        annot2[k]['yref'] = 'y2'
    fig.update_layout(annotations=annot1+annot2, 
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=800, width=1200)  
    
    if return_fig:
        return fig

    

    img = figure_to_array(fig)
    return img



def plot_orginal_heatmaps_ar(y, pred, dataset, inputs, BATCH_ID=0, loss=None, return_fig=False):
    eq = (y == pred).astype(int)
    
    inputs = inputs[:, :, :y.shape[2]] # cut off the parameter only featuers

    desc_y = [i.split(" ") for i in dataset.decode_pred(y, BATCH_ID).split("\n")]
    desc_pred = [i.split(" ") for i in dataset.decode_pred(pred, BATCH_ID).split("\n")]
    desc_inputs = [i.split(" ") for i in dataset.decode_pred(inputs, BATCH_ID).split("\n")]

    # add padding to align everything
    padded_desc_y = [i + [None] * (y.shape[-1] - len(i)) for i in desc_y]
    padded_desc_pred = [i + [None] * (y.shape[-1] - len(i)) for i in desc_pred]
    padded_desc_inputs = [i + [None] * (y.shape[-1] - len(i)) for i in desc_inputs]

    
    # Targets
    batch_eq = np.flip(eq[BATCH_ID, :, :], 0)
    text = np.flip(np.array(padded_desc_y)[:-1, :], 0)
    
    fig1 = ff.create_annotated_heatmap(batch_eq, annotation_text=text)
    fig1.update_layout(height=1500, width=500, 
                    margin=dict(l=20, r=20, t=20, b=20),
    )
    
    # Predictions
    if loss is None:
        loss = eq
    loss_mags = np.flip(loss[BATCH_ID, :, :], 0)
    text = np.flip(np.array(padded_desc_pred)[:-1, :], 0)
    
    fig2 = ff.create_annotated_heatmap(loss_mags, annotation_text=text)
    fig2.update_layout(height=1500, width=500, 
                    margin=dict(l=20, r=20, t=20, b=20),
    )
    
    # inputs
    text = np.flip(np.array(padded_desc_inputs)[:-1, :], 0)
    fig3 = ff.create_annotated_heatmap(np.zeros_like(loss_mags), annotation_text=text)
    fig3.update_layout(height=1500, width=500, 
                    margin=dict(l=20, r=20, t=20, b=20),
    )

    fig = make_subplots(rows=1, cols=3, subplot_titles = ['Inputs', 'Pred', 'True'])
    fig.add_trace(fig3.data[0], 1, 1)
    fig.add_trace(fig2.data[0], 1, 2)
    fig.add_trace(fig1.data[0], 1, 3)

    annot1 = list(fig1.layout.annotations)
    annot2 = list(fig2.layout.annotations)
    annot3 = list(fig3.layout.annotations)
    for k  in range(len(annot2)):
        annot2[k]['xref'] = 'x2'
        annot2[k]['yref'] = 'y2'
    
    for k  in range(len(annot1)):
        annot1[k]['xref'] = 'x3'
        annot1[k]['yref'] = 'y3'
    fig.update_layout(annotations=annot1+annot2+annot3, 
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=1500, width=1800)  
    
    if return_fig:
        return fig

    

    img = figure_to_array(fig)
    return img

