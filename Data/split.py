import glob
import json
import random
from sklearn import model_selection
from Lib.lib import write_json

# meta = {}
# root = "/ai/mnt/data/dicom/pair"
# dcms = glob.glob(root + "/*.dcm")
# patients = [i.split("/")[-1].split("_X_")[0] for i in dcms]
# patients = list(set(patients))
# print(len(patients))
# meta["ALL_PATIENTS"] = patients
# num_patients = len(patients)
# random.shuffle(patients)
# random.shuffle(patients)
# f1 = int(num_patients * 0.33)
# f2 = f1 + f1
# meta["PATIENTS_FOLD1"] = dict(train=patients[:f1], test=patients[f1:])
# meta["PATIENTS_FOLD2"] = dict(train=patients[f1:f2], test=patients[:f1] + patients[f2:])
# meta["PATIENTS_FOLD3"] = dict(train=patients[f2:], test=patients[:f2])
# meta["PATIENTS_FOLD1_SINGLE_VIEW"] = dict(CORONAL_VIEW={}, SAGITTAL_VIEW={})
# meta["PATIENTS_FOLD2_SINGLE_VIEW"] = dict(CORONAL_VIEW={}, SAGITTAL_VIEW={})
# meta["PATIENTS_FOLD3_SINGLE_VIEW"] = dict(CORONAL_VIEW={}, SAGITTAL_VIEW={})

# ### FOLD1 ###
# cor_view_train = []
# cor_view_test = []
# sag_view_train = []
# sag_view_test = []
# dual_view_train = []
# dual_view_test = []
# for train_patient in meta["PATIENTS_FOLD1"]["train"]:
#     cor_view_train += glob.glob(
#         "/ai/mnt/data/dicom/pair/" + train_patient + "*_C_*.dcm"
#     )
# for test_patient in meta["PATIENTS_FOLD1"]["test"]:
#     cor_view_test += glob.glob("/ai/mnt/data/dicom/pair/" + test_patient + "*_C_*.dcm")

# for train_patient in meta["PATIENTS_FOLD1"]["train"]:
#     sag_view_train += glob.glob(
#         "/ai/mnt/data/dicom/pair/" + train_patient + "*_S_*.dcm"
#     )
# for test_patient in meta["PATIENTS_FOLD1"]["test"]:
#     sag_view_test += glob.glob("/ai/mnt/data/dicom/pair/" + test_patient + "*_S_*.dcm")

# meta["PATIENTS_FOLD1_SINGLE_VIEW"]["CORONAL_VIEW"] = dict(
#     train=cor_view_train, test=cor_view_test
# )
# meta["PATIENTS_FOLD1_SINGLE_VIEW"]["SAGITTAL_VIEW"] = dict(
#     train=sag_view_train, test=sag_view_test
# )


# ### FOLD1 ###

# ### FOLD2 ###
# cor_view_train = []
# cor_view_test = []
# sag_view_train = []
# sag_view_test = []
# for train_patient in meta["PATIENTS_FOLD2"]["train"]:
#     cor_view_train += glob.glob(
#         "/ai/mnt/data/dicom/pair/" + train_patient + "*_C_*.dcm"
#     )
# for test_patient in meta["PATIENTS_FOLD2"]["test"]:
#     cor_view_test += glob.glob("/ai/mnt/data/dicom/pair/" + test_patient + "*_C_*.dcm")

# for train_patient in meta["PATIENTS_FOLD2"]["train"]:
#     sag_view_train += glob.glob(
#         "/ai/mnt/data/dicom/pair/" + train_patient + "*_S_*.dcm"
#     )
# for test_patient in meta["PATIENTS_FOLD2"]["test"]:
#     sag_view_test += glob.glob("/ai/mnt/data/dicom/pair/" + test_patient + "*_S_*.dcm")

# meta["PATIENTS_FOLD2_SINGLE_VIEW"]["CORONAL_VIEW"] = dict(
#     train=cor_view_train, test=cor_view_test
# )
# meta["PATIENTS_FOLD2_SINGLE_VIEW"]["SAGITTAL_VIEW"] = dict(
#     train=sag_view_train, test=sag_view_test
# )
# ### FOLD2 ###

# ### FOLD3 ###
# cor_view_train = []
# cor_view_test = []
# sag_view_train = []
# sag_view_test = []
# for train_patient in meta["PATIENTS_FOLD3"]["train"]:
#     cor_view_train += glob.glob(
#         "/ai/mnt/data/dicom/pair/" + train_patient + "*_C_*.dcm"
#     )
# for test_patient in meta["PATIENTS_FOLD3"]["test"]:
#     cor_view_test += glob.glob("/ai/mnt/data/dicom/pair/" + test_patient + "*_C_*.dcm")

# for train_patient in meta["PATIENTS_FOLD3"]["train"]:
#     sag_view_train += glob.glob(
#         "/ai/mnt/data/dicom/pair/" + train_patient + "*_S_*.dcm"
#     )
# for test_patient in meta["PATIENTS_FOLD3"]["test"]:
#     sag_view_test += glob.glob("/ai/mnt/data/dicom/pair/" + test_patient + "*_S_*.dcm")

# meta["PATIENTS_FOLD3_SINGLE_VIEW"]["CORONAL_VIEW"] = dict(
#     train=cor_view_train, test=cor_view_test
# )
# meta["PATIENTS_FOLD3_SINGLE_VIEW"]["SAGITTAL_VIEW"] = dict(
#     train=sag_view_train, test=sag_view_test
# )
# ### FOLD3 ###


# write_json(
#     meta,
#     outfile="/ai/mnt/code/DSFNet_MTICI/Data/meta.json",
# )

# root = "/ai/mnt/data/erase_renamed_pair_relabel_RS"
# dcms = (
#     glob.glob(root + r"/*_C_*.dcm")
#     + glob.glob(root + r"/*C#b*.dcm")
#     + glob.glob(root + r"/*C#a*.dcm")
#     + glob.glob(root + r"/*C#m*.dcm")
# )
# dcms = list(set(dcms))
# random.shuffle(dcms)
# random.shuffle(dcms)
# split_index = int(len(dcms) * 0.636)
# train_list = dcms[:split_index]
# val_list = dcms[split_index:]
# meta = {}
# meta["CORONAL_VIEW"] = {}
# meta["CORONAL_VIEW"]["train"] = train_list
# meta["CORONAL_VIEW"]["val"] = val_list
# write_json(
#     meta,
#     outfile="/ai/mnt/code/DSFNet_MTICI/Data/ReNamedAll.json",
# )
