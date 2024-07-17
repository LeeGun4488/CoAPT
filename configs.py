import copy


base = dict(
    # dataset configs
    data = dict(
        root='/workspace/dataset',
        datasets_base_to_new=['oxford_flowers'],
        # datasets_base_to_new=['dtd', 'caltech101', 'eurosat', 'ucf101', 'oxford_flowers', 
                            #   'oxford_pets', 'stanford_cars', 'fgvc_aircraft', 'food101', 'sun397'],
        datasets_cross_dataset=['caltech101', 'oxford_pets', 'stanford_cars', 'oxford_flowers', 'food101',
                                'fgvc_aircraft', 'sun397', 'dtd', 'eurosat', 'ucf101',
                                'imagenetv2', 'imagenet_sketch', 'imagenet_a', 'imagenet_r'],
    ),

    # mail configs
    mail = dict(
        username='somebody@example.com',
        password='password here',
        host='host here',
        to='somebody@example.com',
    ),
)

##########################################################

coop = dict(
    # GPU ids, if you have multiple GPUs, it can be setted to [0, 1, 2, ...]
    # number of GPU ids is recommanded to be a multiple of 3
    # because seeds are 1, 2, 3
    gpu_ids = [0],
    # gpu_ids = [0, 1, 2],
    # training and eval mode
    # 'b2n' means base to new, or 'xd' means cross dataset and domain generalization
    mode='b2n',
    
    # training configs
    train = dict(
        trainer='CoOp',              # trainer, please see trainers
        cfg='vit_b16_ep10_bs4_lr35', # config, please see configs/
        seeds=[1, 2, 3],             # seeds
        loadep=-1,                   # load epoch, -1 to load the last epoch
        shots=16,                    # num of shots
        opts=[],                     # extra opts, if you have, please add, such as [OPTIM.MAX_EPOCH, 10]
    ),
    
    # grid search configs, if enable=False, grid search will not be used
    grid_search = dict(enable=False),
    
    # output configs
    output = dict(
        root='outputs/coop',   # output root
        result='results/coop', # result root 
        remove_dirs=['root'],  # which directorys will be removed before training task starts
    ),
)

cocoop = dict(
    gpu_ids = [0],
    mode='b2n',
    
    train = dict(
        trainer='CoCoOp',              
        cfg='vit_b16_c4_ep10_batch1_ctxv1', 
        seeds=[1, 2, 3],             
        loadep=-1,                   
        shots=16,                   
        opts=[],      
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/cocoop',  
        result='results/cocoop',
        remove_dirs=['root'],       
    ),
)

kgcoop = dict(
    gpu_ids = [0],
    mode='b2n',
    
    train = dict(
        trainer='KgCoOp',              
        cfg='vit_b16_ep10_ctxv1_bs4_lr35', 
        seeds=[1, 2, 3],             
        loadep=-1,                   
        shots=16,                   
        opts=[],      
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/kgcoop',  
        result='results/kgcoop',
        remove_dirs=['root'],       
    ),
)

maple = dict(
    gpu_ids = [0],
    mode='b2n',
    
    train = dict(
        trainer='MaPLe',              
        cfg='vit_b16_c2_ep10_batch4_2ctx', 
        seeds=[1, 2, 3],             
        loadep=-1,                   
        shots=16,                   
        opts=[],      
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/maple',  
        result='results/maple',
        remove_dirs=['root'],       
    ),
)

promptsrc = dict(
    gpu_ids = [0],
    mode='b2n',
    
    train = dict(
        trainer='PromptSRC',              
        cfg='vit_b16_c2_ep20_batch4_4+4ctx', 
        seeds=[1, 2, 3],             
        loadep=-1,                   
        shots=16,                   
        opts=[],      
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/promptsrc',  
        result='results/promptsrc',
        remove_dirs=['root'],       
    ),
)

dept = dict(
    gpu_ids = [0],
    mode='b2n',
    
    train = dict(
        trainer='DePT',              
        cfg='vit_b16_c2_ep20_batch4_4+4ctx', 
        seeds=[1, 2, 3],             
        loadep=-1,                   
        shots=16,                   
        opts=['TRAINER.LINEAR_PROBE.WEIGHT', 0.7,
              'TRAINER.LINEAR_PROBE.CLS_WEIGHT', 0.6,
              'OPTIM.LR_EXP', 6.0],      
        # opts=['TRAINER.LINEAR_PROBE.WEIGHT', 0.5,
        #       'TRAINER.LINEAR_PROBE.CLS_WEIGHT', 0.8,
        #       'OPTIM.LR_EXP', 4.0],
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/dept',  
        result='results/dept',
        remove_dirs=['root'],       
    ),
)

coop_coapt = dict(
    gpu_ids = [0],
    mode='b2n',
    
    train = dict(
        trainer='CoOp_CoAPT',      
        cfg='vit_b16_ep10_bs4_lr35',
        seeds=[1, 2, 3],    
        loadep=-1,         
        shots=16,   
        opts=['NUM_A', '32',
              'VOCAB', 'gpt-L',
              'layer1', '128',
              'layer2', '256',
              'layer3', '2048'
              ],    
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/coop_coapt',   
        result='results/coop_coapt', 
        remove_dirs=['root'],  
    ),
)

cocoop_coapt = dict(
    gpu_ids = [0],
    mode='b2n',
    
    train = dict(
        trainer='CoCoOp_CoAPT',              
        cfg='vit_b16_c4_ep10_batch1_ctxv1', 
        seeds=[1, 2, 3],             
        loadep=-1,                   
        shots=16,                   
        opts=['NUM_A', '32',
              'VOCAB', 'gpt-L',
              'layer1', '128',
              'layer2', '256',
              'layer3', '512'
              ],       
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/cocoop_coapt',  
        result='results/cocoop_coapt',
        remove_dirs=['root'],       
    ),
)

kgcoop_coapt = dict(
    gpu_ids = [0],
    mode='b2n',
    
    train = dict(
        trainer='KgCoOp_CoAPT',              
        cfg='vit_b16_ep10_ctxv1_bs4_lr35', 
        seeds=[1, 2, 3],             
        loadep=-1,                   
        shots=16,                   
        opts=['NUM_A', '32',
              'VOCAB', 'gpt-L',
              'layer1', '128',
              'layer2', '2048',
              'layer3', '128'
              ],       
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/kgcoop_coapt',  
        result='results/kgcoop_coapt',
        remove_dirs=['root'],       
    ),
)

maple_coapt = dict(
    gpu_ids = [0],
    mode='b2n',
    
    train = dict(
        trainer='MaPLe_CoAPT',              
        cfg='vit_b16_c2_ep10_batch4_2ctx', 
        seeds=[0],             
        loadep=-1,                   
        shots=16,                   
        opts=['NUM_A', '32',
              'VOCAB', 'gpt-L',
              'layer1', '128',
              'layer2', '128',
              'layer3', '256'
              ],         
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/maple_coapt',  
        result='results/maple_coapt',
        remove_dirs=['root'],       
    ),
)

promptsrc_coapt = dict(
    gpu_ids = [0],
    mode='b2n',
    
    train = dict(
        trainer='PromptSRC_CoAPT',              
        cfg='vit_b16_c2_ep20_batch4_4+4ctx', 
        seeds=[1, 2, 3],             
        loadep=-1,                   
        shots=16,                   
        opts=['NUM_A', '32',
              'VOCAB', 'gpt-L',
              'layer1', '128',
              'layer2', '128',
              'layer3', '256'
              ],       
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/promptsrc_coapt',  
        result='results/promptsrc_coapt',
        remove_dirs=['root'],       
    ),
)

dept_coapt = dict(
    gpu_ids = [0],
    mode='b2n',
    
    train = dict(
        trainer='DePT_CoAPT',              
        cfg='vit_b16_c2_ep20_batch4_4+4ctx', 
        seeds=[1, 2, 3],             
        loadep=-1,                   
        shots=16,
        opts=['NUM_A', '32',
              'VOCAB', 'gpt-L',
              'layer1', '128',
              'layer2', '512',
              'layer3', '64',                   
              'TRAINER.LINEAR_PROBE.WEIGHT', 0.7,
              'TRAINER.LINEAR_PROBE.CLS_WEIGHT', 0.6,
              'OPTIM.LR_EXP', 6.0],      
            #   'TRAINER.LINEAR_PROBE.WEIGHT', 0.5,
            #   'TRAINER.LINEAR_PROBE.CLS_WEIGHT', 0.8,
            #   'OPTIM.LR_EXP', 4.0],
    ),
    
    grid_search = dict(enable=False),
    
    output = dict(
        root='outputs/dept_coapt',  
        result='results/dept_coapt',
        remove_dirs=['root'],       
    ),
)

def get_config(name):
    cfg = copy.deepcopy(base)
    extend_cfg = copy.deepcopy(globals()[name])
    cfg.update(extend_cfg)
    return cfg
