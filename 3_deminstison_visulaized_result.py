import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler
metrics = ['Accuracy', 'Runtime', 'Flexibility']
name=[
'Qua+RF(49d)',
'Qua+Linear(49d)',
'Qua+Xgboost(7d)',
'Our method(7d)'
]
data_source='UCI'


if data_source=='UCI':
      a1=356.2482294+  (1-0.967389638)+ 1.172504529+152.8578384+1.190800972+0.005973423
      a2=933.7570202+  (1-0.705109778)  +	3.49614904+	462.8763725+	3.613802344+	0.018027479
      a3=203.9331779+  (1-0.989923386) + 0.796745349+ 103.0728218+ 0.797558405+0.004031568
      a4=229.6107355+  (1-0.986857902)  +	0.759254168+	101.1585724+	0.753878613+ 0.003960462
      accuracy=[a1,a2,a3,a4]
      t1=0.672022962
      t2=0.022535326
      t3=1.490212877
      t4=1.947284079
      time=[t1,t2,t3,t4]
      f1=0.034334065
      f2=0.066040452
      f3=0.030935954
      f4=0.026972598
      flexibility=[f1,f2,f3,f4]
      data_=[
          [a1,t1,f1],
          [a2,t2,f2],
          [a3,t3,f3],
          [a4,t4,f4]
          ]
else:
      data_ = [
          [107.0690153,    0.993009025,    0.754976514,    57.95122312,    0.759327229,    0.003889631],
          [501.4744032,	0.828742743,	4.256073002,	320.7704741,	4.136111124,	0.021459298],
          [39.73728892,	0.999039334,	0.226188716,	16.24029333,	0.22613366,	0.001091136],
           [38.11524204,	0.999112591,	0.075359187,	5.767043028,	0.07432221,	0.000387572]
      ]
data_=np.array(data_)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset_=np.zeros([4,3])
dataset=[]
for i in range (0, 3):

  dataset= scaler.fit_transform(data_[:,i].reshape(-1, 1))
  # dataset = (data_[:, i].reshape(-1, 1))
  dataset=np.reshape(dataset, [len(dataset)])
  dataset_[:,i]=np.array(dataset)
  # if(i!=2):
  dataset_[:, i]= 1-dataset_[:,i]
  #print (i)

fig = go.Figure()
fig.add_trace(go.Scatterpolar(
      r= dataset_[3],
      marker=dict(color='pink', ),
      theta=metrics,
      name=name[3]),
)
fig.add_trace(go.Scatterpolar(
      r= dataset_[2],
      marker=dict(color='mediumseagreen', ),
      theta=metrics,
      name=name[2]),
)
fig.add_trace(go.Scatterpolar(
      r= dataset_[1],
      marker=dict(color='  darkorchid', ),
      theta=metrics,
      name=name[1]),
)

fig.add_trace(go.Scatterpolar(
      r= dataset_[0],
      marker=dict(color=' tomato', ),
      theta=metrics,
      name=name[0]),
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