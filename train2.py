if __name__ == '__main__':
    from train import *
    train_on_folds(
        session_name='FSRCNNSRCNN-5000-64',
        output_dir=r'E:\my_files\programmes\python\super_resolution_outputs',
        x_size=(64, 64),
        y_size=(128, 128),
        model_select=ModelSelect.FSRCNNSRCNN,
        discriminator_select=DiscriminatorSelect.NO_DISCRIMINATOR,
        dataset_dirs=[
            r'E:\my_files\programmes\python\super_resolution_images\fold0',
            r'E:\my_files\programmes\python\super_resolution_images\fold1',
            r'E:\my_files\programmes\python\super_resolution_images\fold2',
            r'E:\my_files\programmes\python\super_resolution_images\fold3',
            r'E:\my_files\programmes\python\super_resolution_images\fold4',
            # r'E:\my_files\programmes\python\super_resolution_images\srclassic\SR_training_datasets\T91',
            r'E:\my_files\programmes\python\super_resolution_images\srclassic\SR_testing_datasets\Set5',
        ],
        epochs=70,
        batch_size_train=8,
        batch_size_val=1,
        learning_rate=0.0001,
        num_workers_train=8,
        num_workers_val=0,
        shuffle=True,
        run_tests=1,
        train_crop='random_scale_crop',
        # choose_detaset='resize',
    )
