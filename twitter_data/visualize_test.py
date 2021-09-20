import numpy as np
import matplotlib.pyplot as plt
import graphvite as gv
import graphvite.application as gap

app = gap.VisualizationApplication(dim=2)

with open("sub_6_model.pkl", "rb") as fin:
    model = pickle.load(fin)
embeddings = model.solver.vertex_embeddings

app.load(vectors=embeddings, perplexity=6)

app.build()

app.save_model('sub_6_vis_model.pkl')

app.visualization(save_file='sub_6.png')