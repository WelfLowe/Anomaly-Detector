from torch import nn

from anomaly_detector.NF.freia_funcs import (
    permute_layer,
    glow_coupling_layer,
    F_fully_connected,
    ReversibleGraphNet,
    OutputNode,
    InputNode,
    Node,
)


def nf_head(n_feat, n_coupling_blocks, clamp_alpha, fc_internal, dropout):
    nodes = list()
    nodes.append(InputNode(n_feat, name="input"))
    #print("input_size: {}".format(n_feat))
    #print("n_coupling_blocks: {}".format(n_coupling_blocks))
    #print("clamp_alpha: {}".format(clamp_alpha))
    #print("fc_internal: {}".format(fc_internal))
    #print("dropout: {}".format(dropout))

    for k in range(n_coupling_blocks):
        nodes.append(
            Node([nodes[-1].out0], permute_layer, {"seed": k}, name=f"permute_{k}")
        )
        nodes.append(
            Node(
                [nodes[-1].out0],
                glow_coupling_layer,
                {
                    "clamp": clamp_alpha,
                    "F_class": F_fully_connected,
                    "F_args": {"internal_size": fc_internal, "dropout": dropout},
                },
                name=f"fc_{k}",
            )
        )
    nodes.append(OutputNode([nodes[-1].out0], name="output"))
    coder = ReversibleGraphNet(nodes)
    # print number of parameters
    #n_params = sum([p.numel() for p in coder.parameters()])
    #print("n_params: {}".format(n_params))

    return coder


class DifferNet(nn.Module):
    def __init__(self, n_feat, n_coupling_blocks, clamp_alpha, fc_internal, dropout):
        super(DifferNet, self).__init__()
        self.nf = nf_head(n_feat, n_coupling_blocks, clamp_alpha, fc_internal, dropout)

    def forward(self, x):
        z = self.nf(x)
        return z
