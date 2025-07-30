from torch.nn import init


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        if len(list(m.children())) > 0:
            for name, submodule in m.named_children():
                classname = submodule.__class__.__name__
                if hasattr(submodule, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                    if init_type == 'normal':
                        init.normal_(submodule.weight.data, 0.0, gain)
                    elif init_type == 'xavier':
                        init.xavier_normal_(submodule.weight.data, gain=gain)
                    elif init_type == 'kaiming':
                        init.kaiming_normal_(submodule.weight.data, a=0, mode='fan_in')
                    elif init_type == 'orthogonal':
                        init.orthogonal_(submodule.weight.data, gain=gain)
                    else:
                        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(submodule, 'bias') and submodule.bias is not None:
                    init.constant_(submodule.bias.data, 0.0)
                elif classname.find('BatchNorm2d') != -1:
                    init.normal_(submodule.weight.data, 1.0, gain)
                    init.constant_(submodule.bias.data, 0.0)
        else:
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
