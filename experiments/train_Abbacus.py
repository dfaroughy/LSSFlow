from data.datasets import AbbacusData
from utils.experiments import Configs, RunExperiment

#==================

config = Configs(
                num_nodes = 4,
                tags = ['Perlmutter', 'train'],
                datafile = '/pscratch/sd/d/dfarough/LSS_data/halo_positions.npy',
                data_name = 'Abbacus',
                radius = None,              # choose 'Universe' box size in Mpc/h
                source_support = 'ball',    # unit 'ball' or 'cube'
                gen_model = 'ConditionalFlowMatching',
                flow = 'UniformFlow',
                network = 'AutoCompressingNet',
                dim = 3,
                dim_fourier = 256,
                n_embd = 256,
                num_blocks = 10,
                dropout = 0.2,
                activation = 'GeLU',
                sigma = 1e-5,               # cfm hyperparameter
                gamma = 1.0,                # fourier feature hyperparameter
                use_OT = True,
                batch_size = 2048,
                max_epochs = 20000,
                train_split = 1.0,
                lr = 0.001,
                lr_final = 0.0001,
                warmup_epochs = 5
                )
                
#==================

abbacus = AbbacusData(datafile=config.datafile, radius=config.radius)
target = abbacus.sample()
experiment = RunExperiment(config=config, target=target)
experiment.train()