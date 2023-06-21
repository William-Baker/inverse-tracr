
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

