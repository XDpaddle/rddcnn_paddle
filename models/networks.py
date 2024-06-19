import paddle
import models.archs.classSR_rcan_arch as classSR_rcan_arch
import models.archs.ADNet as ADNet
import models.archs.rddcnn as rddcnn


# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    #scale = opt_net['scale']
    print('*'*100,opt_net)

    # image restoration
    if which_model == 'rddcnn':
        netG = rddcnn.DnCNN()


    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG