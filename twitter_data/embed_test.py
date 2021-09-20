import graphvite as gv
import graphvite.application as gap
app = gap.GraphApplication(dim=128)

#as_directed=Trueがデフォルト、有向でやりたいがとりあえず無効で
app.load(file_name='sub_6_ncol')

app.build()

app.train()

app.save_model('sub_6_model.pkl')
