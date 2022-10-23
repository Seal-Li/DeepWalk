# DeepWalk
DeepWalk: Online Learning of Social Representations


## Task
Link predict, AUC: 97.39%


## parameter
set parameter as file utlis.arg_parser!

Watch out: it's not best parameter, we do not take fine-tune.


## dataset(Amazon)
train_edges.txt: src_node dst_node, seperated with " "(space)

test_edges.txt: src_node dst_node label seperated with " "(space)


## output
embeddings.txt: node_id dim_1 dim_2 ... dim_n, seperated with " "(space)

model.pt: all of model parameter, you can get the same results by loading model.pt file.