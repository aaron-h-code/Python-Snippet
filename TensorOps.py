def batched_index_select(input, dim, index):
    """
    :param input: B x * x ... x *
    :param dim: 0 < scalar
    :param index: B x M
    :return:
    """
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)

    return torch.gather(input, dim, index)
