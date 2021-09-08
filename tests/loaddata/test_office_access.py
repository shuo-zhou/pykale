from numpy import testing

from kale.loaddata.multi_domain import MultiDomainAdapDataset, MultiDomainDatasets
from kale.loaddata.office_access import Office31, OfficeAccess, OfficeCaltech


def test_office31(office_path):
    office_access = Office31(root=office_path, download=True, return_domain_label=True)
    testing.assert_equal(len(office_access.class_to_idx), 31)
    testing.assert_equal(len(office_access.domain_to_idx), 3)
    dataset = MultiDomainAdapDataset(office_access)
    dataset.prepare_data_loaders()
    domain_labels = list(dataset.domain_to_idx.values())
    for split in ["train", "valid", "test"]:
        dataloader = dataset.get_domain_loaders(split=split)
        x, y, z = next(iter(dataloader))
        for domain_label_ in domain_labels:
            testing.assert_equal(y[z == domain_label_].shape[0], 10)


def test_office_caltech(office_path):
    office_access = OfficeCaltech(root=office_path, download=True)
    testing.assert_equal(len(office_access.class_to_idx), 10)
    testing.assert_equal(len(office_access.domain_to_idx), 4)


def test_custom_office(office_path):
    source = OfficeAccess(root=office_path, download=True, sub_domain_set=["dslr"], split_train_test=True)
    target = OfficeAccess(root=office_path, download=True, sub_domain_set=["webcam"], split_train_test=True)
    dataset = MultiDomainDatasets(source_access=source, target_access=target)
    dataset.prepare_data_loaders()
    dataloader = dataset.get_domain_loaders()
    testing.assert_equal(len(next(iter(dataloader))), 2)
