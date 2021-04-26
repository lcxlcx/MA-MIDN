from dataloader.instance_data import PatchMethod
import torch.utils.data as data_utils


def get_train_loader(data_path_train, batch_size, num_workers):
    data = PatchMethod(root=data_path_train)
    train_loader = data_utils.DataLoader(data, shuffle=True, num_workers=num_workers, batch_size=batch_size)
    return train_loader

def get_test_loader(data_path_test, batch_size, num_workers):
    val_data = PatchMethod(root=data_path_test, mode='test')
    test_loader = data_utils.DataLoader(val_data, shuffle = False, num_workers=num_workers, batch_size=batch_size)
    return test_loader


