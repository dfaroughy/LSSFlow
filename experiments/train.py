from data.datasets import UchuuVoxelizedData
from utils.experiments import Configs, RunExperiment

#==================

config = Configs(
                num_nodes = 4,
                tags = ['Perlmutter', 'train'],
                datafile = '/pscratch/sd/d/dfarough/LSS_data/Uchuu1000-Pl18_z0p00_hlist_4.h5',
                data_name = 'Uchuu',
                box_size = 128,          # choose 'Universe' box size in Mpc/h
                voxel_size = 64,         # choose voxelization size in Mpc/h
                voxel_idx = 0,           # choose which voxel index to train on
                gen_model = 'ActionMatching',
                network = 'ResNet',
                dim = 3,
                dim_fourier = 256,
                n_embd = 256,
                num_blocks = 10,
                dropout = 0.1,
                activation = 'LeakyReLU',
                sigma = 1e-5,               # cfm hyperparameter
                gamma = 1.0,                # fourier feature hyperparameter
                mass_reg = 1.0,
                use_mass_reg = True,
                use_OT = False,
                batch_size = 2048,
                max_epochs = 20000,
                train_split = 1.0,
                lr = 0.001,
                lr_final = 0.0001,
                warmup_epochs = 5
                )
                
#==================

lss = UchuuVoxelizedData(datafile=config.datafile, box_size=config.box_size, voxel_size=config.voxel_size)
target = lss.sample(voxel_idx=config.voxel_idx, normalize=True)
experiment = RunExperiment(config=config, target=target)
experiment.train()