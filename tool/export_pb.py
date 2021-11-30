from keras import backend as K
from tensorflow.python.saved_model import builder as saved_model_builder
from keras.models import load_model

import tensorflow as tf


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph





#keras model file load
# keras_model = load_model('../checkpoint/model-201905240942-42-loss:0.0329-val_loss:0.0565-val_acc:0.9817.hdf5')
# keras_model = load_model('../checkpoint/model-201906091502-47-loss:0.0281-val_loss:0.0766-val_acc:0.9754.hdf5')
# keras_model = load_model('../checkpoint/20190619-430.hdf5')
keras_model = load_model('../checkpoint/model-201906201611-52-loss:0.0216-val_loss:0.0953-val_acc:0.9750.hdf5')
print(keras_model.inputs)
print(keras_model.outputs)
frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in keras_model.outputs])
#pb file export
tf.train.write_graph(frozen_graph, "/home/hclee/workspace/git_adev2/lane_localization/serving/", "440.pb", as_text=False)