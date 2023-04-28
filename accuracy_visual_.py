import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler
metrics = ['RMSE', 'R2', 'MAPE', 'MAE', 'SMAPE', 'TIC']
name=[
'Qua+RF',
'Qua+Linear',
'Bi-LSTM+Att',
'LSTM+Att',
'LSTM',
'Our method'
]
data_source='UCSXXXD'
if data_source=='UCI':
      data_=[
          [356.2482294,	0.967389638,	1.172504529,	152.8578384,	1.190800972,	0.005973423], #rf
          [933.7570202,	0.705109778,	3.49614904,	462.8763725,	3.613802344,	0.018027479],
          [456.2894702,  0.938083561,    2.744712211,    348.0646047,    2.768788393,    0.01358911],
          [426.6429054,	0.946441012,	2.551748318,	324.0951888,	2.574052275,	0.012651278],
          [394.1564586,	0.954759029,	2.381031571,	302.0568648,	2.39942637,	0.01178532],#lstm
          [271.4036291,	0.98142264,	0.709524859,	95.73145459,	0.699135403,	0.003749034]#our
          ]
elif data_source=='UCSD':
      data_=[
          [350.4765337,	0.97748179,	0.529555226,	175.7251914,	0.531489843,	0.002644894], #rf
          [785.8859374,	0.877451721,	1.122164664,	378.6781256,	1.146848179,	0.005674981],
          [468.4881911,	0.955428876,	1.069915067,	354.565125,	1.070519097,	0.005340817],
          [453.6652429,	0.959001139,	1.015072449,	335.7537493,	1.016085205,	0.005056756],
          [428.0758146,	0.966299441,	0.917872381,	302.5239004,	0.91994809,	0.004553344],#lstm
          [275.1510284,	0.986599734,	0.373561323,	124.381575,	0.373733682,	0.001873066]#our
          ]
else:
      data_ = [
          [107.0690153,    0.993009025,    0.754976514,    57.95122312,    0.759327229,    0.003889631],#rf
          [501.4744032,	0.828742743,	4.256073002,	320.7704741,	4.136111124,	0.021459298],
          # [39.73728892,	0.999039334,	0.226188716,	16.24029333,	0.22613366,	0.001091136],
           [199.4604197,	0.973665438,	2.116218892,	155.3117766,	2.108377193,	0.010467743],
           [190.2474464,	0.975825737,	2.009970416,	148.0310974,	2.00013627,	0.009983767],
           [197.9818109,	0.97337913,	2.079835359,	155.4764447,	2.05850662,	0.01051664],#lstm
           [46.1688609,	0.998698613,	0.163326615,	11.94685876,	0.161998774,	0.000802942]#our
      ]
data_=np.array(data_)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset_=np.zeros([6,6])
dataset=[]
for i in range (0, 6):

  dataset= scaler.fit_transform(data_[:,i].reshape(-1, 1))
  dataset=np.reshape(dataset, [len(dataset)])
  dataset_[:,i]=np.array(dataset)
  if(i!=1):
      dataset_[:, i]= 1-dataset_[:,i]
  #print (i)

fig = go.Figure()
fig.add_trace(go.Scatterpolar(
      r= dataset_[5],
      marker=dict(color='pink', ),
      theta=metrics,
      name=name[5]),
)

fig.add_trace(go.Scatterpolar(
      r= dataset_[0],
      marker=dict(color='  blue', ),
      theta=metrics,
      name=name[0]),
)

fig.add_trace(go.Scatterpolar(
      r= dataset_[4],
      marker=dict(color=' yellow', ),
      theta=metrics,
      name=name[4]),
)

fig.add_trace(go.Scatterpolar(
      r= dataset_[3],
      marker=dict(color=' grey', ),
      theta=metrics,
      name=name[3]),
)

# fig.add_trace(go.Scatterpolar(
#       r= dataset_[3],
#       marker=dict(color=' yellowgreen', ),
#       theta=metrics,
#       name=name[3]),
# )
fig.add_trace(go.Scatterpolar(
      r= dataset_[2],
      marker=dict(color='orange', ),
      theta=metrics,
      name=name[2]),
)
fig.add_trace(go.Scatterpolar(
      r= dataset_[1],
      marker=dict(color='green', ),
      theta=metrics,
      name=name[1]),
)

# fig.add_trace(go.Scatterpolar(
#       r= dataset_[0],
#       marker=dict(color=' lightgray', ),
#       theta=metrics,
#       name=name[0]),
# )
#
# fig.add_trace(go.Scatterpolar(
#       r= dataset_[4],
#       marker=dict(color=' deepskyblue', ),
#       theta=metrics,
#       name=name[4]),
# )
# fig.add_trace(go.Scatterpolar(
#       r= dataset_[6],
#       marker=dict(color=' blue', ),
#       theta=metrics,
#       name=name[6]),
# )
# fig.add_trace(go.Scatterpolar(
#       r= dataset_[8],
#       marker=dict(color=' lightblue', ),
#       theta=metrics,
#       name=name[8]),
# )
# fig.add_trace(go.Scatterpolar(
#       r= dataset_[7],
#       marker=dict(color=' deeppink', ),
#       theta=metrics,
#       name=name[7]),
# )


fig.update_traces(fill='toself')
fig.update_layout(
      #title="Classification performance",
      polar=dict( radialaxis=dict(visible=True,  range=[0, 1] )),
      font=dict(
      family="Arial, monospace",
      size=44,
      color="Black") ,
      showlegend=True
)
#fig.update_layout(height=500, width=500)
# Path_sorce_="C:\\Users\\Xinlin\\Desktop\\3rd_predictior\\prediction_results.jpeg"
fig.show()
# fig.write_image(Path_sorce_)