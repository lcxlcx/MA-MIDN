from config.multiple_instance_config import get_config
from dataloader.bag_loader import get_test_loader, get_train_loader
from task.train_Ind import Trainer_Ind

config, unparsed = get_config()


train_dataset = get_train_loader(config.traindata_dir, config.batch_size,config.num_workers)

test_dataset = get_test_loader(config.testdata_dir,config.batch_size,config.num_workers)

data_loader = (train_dataset, test_dataset)



trainer = Trainer_Ind(config, data_loader)
# trainer = Trainer_DML(config, data_loader)

trainer.train()
