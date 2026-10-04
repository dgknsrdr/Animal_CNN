from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights

def Get_Transform():

    #---------------------------------------------model:1 train transform---------------------------------------------#
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.TrivialAugmentWide(),
        transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4508, 0.4392, 0.3964], std=[0.2799, 0.2714, 0.2758])
    ])

    # ---------------------------------------------model:1 test transform---------------------------------------------#

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4508, 0.4392, 0.3964], std=[0.2799, 0.2714, 0.2758])
    ])

    # ---------------------------------------model:EfficientNet transform---------------------------------------#

    weights = EfficientNet_B0_Weights.DEFAULT

    model1_train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.03),
        weights.transforms()
    ])

    model1_test_transform = weights.transforms()

    return train_transform , test_transform ,model1_train_transform ,model1_test_transform

    # ----------------------------------------------------------------------------------------------------------------#