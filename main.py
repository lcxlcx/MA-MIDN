from config.multiple_instance_config import get_config
from dataloader.bag_loader import get_test_loader, get_train_loader
from task.train_Ind import Trainer_Ind

from OICR.task.train import Trainer_MIL as train
config, unparsed = get_config()


train_dataset = get_train_loader(config.traindata_dir, config.batch_size,config.num_workers)

test_dataset = get_test_loader(config.testdata_dir,config.batch_size,config.num_workers)

data_loader = (train_dataset, test_dataset)

# trainer = Trainer_MIL_Softmax(config, data_loader)
# trainer = train(config, data_loader)

trainer = Trainer_Ind(config, data_loader)
# trainer = Trainer_MIL_Resnet(config, data_loader)
# trainer = Trainer_DMIL(config, data_loader)
# trainer = Trainer_MIL_SeResnet(config, data_loader)
# trainer = Trainer_out1_DMIL(config, data_loader)
trainer.train()